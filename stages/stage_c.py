"""
Stage C: 文风指纹与感官映射提取
使用 qwen2.5:7b 模型，多线程提取文风指纹、感官映射、经典摘录
"""

import logging
import math
import re
import os
from typing import List, Dict, Any
from stages.base import BaseStage
from core.ollama_client import ollama_chat, safe_parse_json
from core.utils import generate_id
from config.settings import (
    STAGE_C_WORKERS,
    STAGE_SAMPLE_BASE,
    STAGE_SAMPLE_MULTIPLIER,
    STAGE_SAMPLE_DENOMINATOR,
)

logger = logging.getLogger(__name__)


def process_single_chapter_c(
    chap: Dict, book_name: str, category: str
) -> Dict[str, Any]:
    """处理单章的文风指纹提取"""
    text = chap["text"]

    prompt = f"""你是顶尖文学编辑。请深度拆解本章原文的"文风指纹"、"情绪感官映射"，并【原封不动】摘录经典段落，输出纯JSON。
【书名】{book_name} 【分类】{category}
【正文】{text}
输出JSON：{{
  "author_fingerprint": {{
    "preferred_verbs": ["本章特色动词(非作者全部偏好)，限5个，必须是纯字符串"],
    "preferred_adjectives": ["本章特色形容词(非作者全部偏好)，限5个，必须是纯字符串"],
    "environmental_imagery": ["本章环境描写意象(非全书统计)，限5个，必须是纯字符串"],
    "signature_transitions": ["本章观察到的过渡手法(非全书标志性)或修辞手法，限2个，必须是纯字符串，绝对禁止使用对象或字典嵌套！"],
    "narrative_perspective": "叙事视角(如:第一人称限制视角/全知上帝视角/多视角切换/意识流，限20字)",
    "sentence_rhythm": "句式节奏偏好(如:偏爱绵密的长句与从句/冷峻短促的白描/大量使用破折号与省略号，限30字)"
  }},
  "sensory_mappings": [
    {{
      "emotion": "核心情绪",
      "show_not_tell": "原著中展示该情绪的生理反应/动作/环境细节(限50字)",
      "analysis": "为什么这种描写比直接写情绪更有质感(20字内)"
    }}
  ],
  "classic_excerpts": [
    {{
      "excerpt_text": "从原文中原封不动地摘录 1 段最能代表该作者文风的完整段落（严格控制在300到400字之间，包含标点）。必须是原汁原味的原文，禁止修改任何字词！必须保持句子完整，绝不能在句子中间截断（必须以句号、问号、叹号或省略号结尾）。优先选择包含完整'环境铺垫+动作冲突+情绪反馈'的段落。",
      "scene_type": "场景类型(如:战斗/环境/对话/心理)",
      "style_tag": "风格标签(如:肃杀/幽默/细腻/宏大)"
    }}
  ]
}} (如果没有明显特征或情绪，对应数组留空。classic_excerpts必须严格摘录原文，禁止使用反引号)"""

    raw_resp = ollama_chat(prompt, 0.3, "C")
    res = safe_parse_json(raw_resp)
    if not res:
        if raw_resp.count("{") > raw_resp.count("}"):
            res = safe_parse_json(raw_resp + "}")
        if not res:
            raise Exception("阶段C JSON解析失败")

    res.setdefault("author_fingerprint", {})
    res.setdefault("sensory_mappings", [])
    res.setdefault("classic_excerpts", [])

    # 清洗 author_fingerprint，过滤非字符串项
    fp = res.get("author_fingerprint", {})
    if isinstance(fp, dict):
        for key in [
            "preferred_verbs",
            "preferred_adjectives",
            "environmental_imagery",
            "signature_transitions",
        ]:
            val = fp.get(key, [])
            if isinstance(val, list):
                fp[key] = [
                    str(v) for v in val if isinstance(v, (str, int, float, bool))
                ]
            else:
                fp[key] = []
    else:
        fp = {}

    fp["narrative_perspective"] = str(fp.get("narrative_perspective", ""))
    fp["sentence_rhythm"] = str(fp.get("sentence_rhythm", ""))
    res["author_fingerprint"] = fp

    res.update({"chapter_id": chap["id"], "book_name": book_name, "category": category})
    return res


class StageC(BaseStage):
    """Stage C: 文风指纹与感官映射提取"""

    def __init__(self, book_name: str, category: str):
        super().__init__("C", book_name, category)

    def run(self, chapters: List[Dict], **kwargs) -> List[Dict]:
        """执行 Stage C"""
        print("=== 阶段三：多线程提取文风指纹与感官映射 ===")

        cache = self.load_cache()
        success_list = cache.get("data", []) if cache else []
        completed_ids = {x["chapter_id"] for x in success_list}
        if completed_ids:
            print(f"✅ [阶段C] 恢复断点：已完成 {len(completed_ids)} 章")

        # 均匀间隔采样（--full 模式下全量处理）
        if not os.environ.get("NOVEL_KB_FULL_SAMPLE"):
            import math

            total = len(chapters)
            sample_count = max(
                STAGE_SAMPLE_BASE,
                min(
                    total,
                    int(
                        STAGE_SAMPLE_BASE
                        + STAGE_SAMPLE_MULTIPLIER
                        * math.sqrt(total / STAGE_SAMPLE_DENOMINATOR)
                    ),
                ),
            )
            if total > sample_count:
                step = total / sample_count
                sampled = [chapters[int(i * step)] for i in range(sample_count)]
                logger.info(f"[阶段C] 均匀采样: {sample_count}/{total} 章")
                chapters = sampled

        pending = [c for c in chapters if c["id"] not in completed_ids]
        if not pending:
            return success_list

        def worker_task(chap):
            return process_single_chapter_c(chap, self.book_name, self.category)

        new_results, _ = self.run_parallel(
            pending, worker_task, STAGE_C_WORKERS, "阶段C进度"
        )
        success_list.extend(new_results)
        return success_list

    def insert(self, results: List[Dict]) -> Dict[str, int]:
        """将 Stage C 结果写入数据库"""
        cursor = self.db.connect().cursor()
        stats = {
            "fingerprints_db": 0,
            "sensory_db": 0,
            "sensory_chroma": 0,
            "excerpts_chroma": 0,
        }

        blacklist = {"无", "未知", "暂无", "没有", "null", "none", "未提供"}

        def clean_list(fp, key):
            val = fp.get(key, [])
            return ",".join([w for w in val if w and w not in blacklist])

        for res in results:
            fp = res.get("author_fingerprint", {})
            fp_id = generate_id(res["book_name"], res["chapter_id"], "fingerprint")

            cursor.execute(
                "INSERT OR REPLACE INTO author_fingerprints VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    fp_id,
                    res["book_name"],
                    res["category"],
                    clean_list(fp, "preferred_verbs"),
                    clean_list(fp, "preferred_adjectives"),
                    clean_list(fp, "environmental_imagery"),
                    clean_list(fp, "signature_transitions"),
                    fp.get("negative_prompts", ""),
                    fp.get("narrative_perspective", ""),
                    fp.get("sentence_rhythm", ""),
                ),
            )
            stats["fingerprints_db"] += 1

            # 感官映射入库
            for sm in res.get("sensory_mappings", []):
                sm_id = generate_id(
                    res["book_name"], res["chapter_id"], sm.get("emotion", "")
                )
                cursor.execute(
                    "INSERT OR REPLACE INTO sensory_mappings VALUES (?,?,?,?,?,?,?)",
                    (
                        sm_id,
                        res["book_name"],
                        res["chapter_id"],
                        res["category"],
                        sm.get("emotion", ""),
                        sm.get("show_not_tell", ""),
                        sm.get("analysis", ""),
                    ),
                )
                stats["sensory_db"] += 1

        # ChromaDB: 感官映射
        sen_ids, sen_docs, sen_metas = [], [], []
        for res in results:
            for sm in res.get("sensory_mappings", []):
                sid = generate_id(
                    res["book_name"], res["chapter_id"], sm.get("emotion", "")
                )
                sen_ids.append(sid)
                sen_docs.append(
                    f"情绪:{sm.get('emotion', '')}\n展示:{sm.get('show_not_tell', '')}\n分析:{sm.get('analysis', '')}"
                )
                sen_metas.append(
                    {
                        "book_name": res["book_name"],
                        "category": res["category"],
                        "emotion": sm.get("emotion", ""),
                    }
                )
        if sen_ids:
            self.chroma.upsert_batch("sensory_details", sen_ids, sen_docs, sen_metas)
            stats["sensory_chroma"] = len(sen_ids)

        # ChromaDB: 经典摘录（含乱码拦截）
        exc_ids, exc_docs, exc_metas = [], [], []
        # GBK 误读为 UTF-8 的典型乱码特征
        _garbled_re = re.compile(r"(?:[\x80-\xff]{3,}|[\xc0-\xff]{2,})")
        for res in results:
            for exc in res.get("classic_excerpts", []):
                excerpt_text = exc.get("excerpt_text", "")
                if not excerpt_text:
                    continue
                # 乱码拦截：跳过包含典型乱码特征的摘录
                if _garbled_re.search(excerpt_text):
                    logger.warning(
                        f"跳过乱码摘录: chapter={res.get('chapter_id', '未知')}, text={excerpt_text[:50]}..."
                    )
                    continue
                eid = generate_id(
                    res["book_name"], res["chapter_id"], exc.get("style_tag", "")
                )
                exc_ids.append(eid)
                exc_docs.append(exc["excerpt_text"])
                exc_metas.append(
                    {
                        "book_name": res["book_name"],
                        "category": res["category"],
                        "scene_type": exc.get("scene_type", ""),
                        "style_tag": exc.get("style_tag", ""),
                    }
                )
        if exc_ids:
            self.chroma.upsert_batch("classic_excerpts", exc_ids, exc_docs, exc_metas)
            stats["excerpts_chroma"] = len(exc_ids)

        self.db.commit()
        logger.info(
            f"   ✅ [阶段C战报] 文风指纹: {stats['fingerprints_db']} 条 | "
            f"感官DB: {stats['sensory_db']} | 感官Chroma: {stats['sensory_chroma']} | "
            f"摘录Chroma: {stats['excerpts_chroma']}"
        )
        return stats
