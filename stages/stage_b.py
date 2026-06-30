"""
Stage B: 写作技法与高潮点提取
使用 qwen2.5:7b 模型，多线程提取每章的叙事技法、高潮/张力点、场景类型
"""
import logging
from typing import List, Dict, Any
from stages.base import BaseStage
from core.ollama_client import ollama_chat, safe_parse_json
from core.utils import compress_state_to_text, find_quote_position_fast, generate_id
from config.settings import STAGE_B_WORKERS

logger = logging.getLogger(__name__)


def process_single_chapter_b(
    chap: Dict, book_name: str, category: str
) -> Dict[str, Any]:
    """处理单章的技法提取"""
    text = chap["text"]
    state_text = compress_state_to_text(chap.get("character_state", {}))

    prompt = f"""你是专业的小说技法分析师。基于原文提取写作模板，输出纯JSON。
【书名】{book_name} 【章节】{chap["id"]} 【分类】{category}
【摘要】{chap.get("summary", "")} 【人物状态】{state_text}
【正文】{text}
输出JSON：{{
  "scene_type": "场景类型(根据{category}分类自适应，如：冲突/日常/高潮/铺垫/转折/揭秘/情感/动作等)", 
  "narrative_skills": [{{"skill_name": "", "original_example": "", "analysis": "", "reuse_scenario": ""}}],
  "climax_point": {{"has_climax_point": false, "type": "高潮/张力点类型(如：情感爆发/悬念释放/冲突升级/真相揭露/逆转)", "quote": "", "buildup_method": "构建方式(如:先抑后扬/信息差制造/能力展示/反转打脸/情感铺垫，50字内)"}}, 
  "style_feature": {{
    "tone": "文风调性",
    "sentence_rhythm": "句式节奏(如:短句密集/长短交替/长句为主/对话驱动)",
    "vocabulary_level": "词汇难度(如:通俗口语/文雅书面/专业术语多/古风文言)"
  }}
}} (无高潮点/技法请留空，禁止使用反引号)"""

    raw_resp = ollama_chat(prompt, 0.2, "B")
    res = safe_parse_json(raw_resp)
    if not res:
        if raw_resp.count("{") > raw_resp.count("}"):
            res = safe_parse_json(raw_resp + "}")
        if not res:
            raise Exception("JSON解析彻底失败")

    res.setdefault("narrative_skills", [])
    res.setdefault("scene_type", "未知")
    res.setdefault("climax_point", {"has_climax_point": False, "quote": "", "buildup_method": ""})
    res.setdefault("style_feature", {"tone": "无", "sentence_rhythm": "", "vocabulary_level": ""})
    res["raw_text"] = text

    if res["climax_point"].get("has_climax_point") and res["climax_point"].get("quote"):
        pos = find_quote_position_fast(text, res["climax_point"]["quote"])
        res["climax_point"]["char_pos"] = pos
        if pos == -1:
            res["_unmatched_log"] = {
                "chapter": chap["id"],
                "quote": res["climax_point"]["quote"],
            }

    res.update({"chapter_id": chap["id"], "book_name": book_name, "category": category})
    return res


class StageB(BaseStage):
    """Stage B: 写作技法与高潮点提取"""

    def __init__(self, book_name: str, category: str):
        super().__init__("B", book_name, category)

    def run(self, chapters: List[Dict], **kwargs) -> List[Dict]:
        """执行 Stage B"""
        print("=== 阶段二：多线程提取技法与高潮点 ===")

        # 恢复断点
        cache = self.load_cache()
        success_list = cache.get("data", []) if cache else []
        completed_ids = {x["chapter_id"] for x in success_list}
        if completed_ids:
            print(f"✅ [阶段B] 恢复断点：已完成 {len(completed_ids)} 章")

        pending = [c for c in chapters if c["id"] not in completed_ids]
        if not pending:
            return success_list

        def worker_task(chap):
            return process_single_chapter_b(chap, self.book_name, self.category)

        new_results, fail_list = self.run_parallel(
            pending, worker_task, STAGE_B_WORKERS, "阶段B进度"
        )
        success_list.extend(new_results)

        # 处理未匹配的引文日志
        from config.settings import UNMATCHED_LOG
        log_buffer = [r.get("_unmatched_log") for r in new_results if r.get("_unmatched_log")]
        if log_buffer:
            import json
            with open(UNMATCHED_LOG, "a", encoding="utf-8") as f:
                for item in log_buffer:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        return success_list

    def insert(self, results: List[Dict]) -> Dict[str, int]:
        """将 Stage B 结果写入数据库"""
        cursor = self.db.connect().cursor()
        stats = {"skills_db": 0, "skills_chroma": 0}

        skill_collection = self.chroma.get_collection("novel_skills")

        for res in results:
            for skill in res.get("narrative_skills", []):
                skill_id = generate_id(
                    res["book_name"], res["chapter_id"], skill.get("skill_name", "")
                )
                tags = f"{res.get('scene_type', '')}|{skill.get('skill_name', '')}"
                cursor.execute(
                    "INSERT OR REPLACE INTO skills VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        skill_id,
                        res["book_name"],
                        res["chapter_id"],
                        res["category"],
                        res.get("scene_type", ""),
                        skill.get("skill_name", ""),
                        skill.get("analysis", ""),
                        skill.get("original_example", ""),
                        tags,
                    ),
                )
                stats["skills_db"] += 1

        # ChromaDB 批量写入
        if results:
            s_ids, s_docs, s_metas = [], [], []
            for res in results:
                for skill in res.get("narrative_skills", []):
                    sid = generate_id(res["book_name"], res["chapter_id"], skill.get("skill_name", ""))
                    s_ids.append(sid)
                    s_docs.append(
                        f"技法:{skill.get('skill_name', '')}\n分析:{skill.get('analysis', '')}\n"
                        f"原文:{skill.get('original_example', '')}\n复用场景:{skill.get('reuse_scenario', '')}"
                    )
                    s_metas.append({
                        "book_name": res["book_name"],
                        "category": res["category"],
                        "scene_type": res.get("scene_type", ""),
                        "skill_name": skill.get("skill_name", ""),
                    })
            if s_ids:
                self.chroma.upsert_batch("novel_skills", s_ids, s_docs, s_metas)
                stats["skills_chroma"] = len(s_ids)

        self.db.commit()
        logger.info(f"   ✅ [阶段B战报] skills DB: {stats['skills_db']} 条 | ChromaDB: {stats['skills_chroma']} 条")
        return stats
