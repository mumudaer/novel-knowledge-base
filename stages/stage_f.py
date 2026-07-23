"""
Stage F: 对话/描写/动作专项样本库
使用 qwen14b:latest 模型，按场景类型提取高质量的原文样本
"""

import json
import os
import re
import math
import logging
import unicodedata
import unicodedata
import os
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from stages.base import BaseStage
from core.ollama_client import ollama_chat, safe_parse_json
from core.utils import generate_id
from core.stage_result import StageResult
from core.chroma_utils import bulk_upsert_to_chroma
from config.settings import (
    STAGE_F_WORKERS,
    STAGE_SAMPLE_BASE,
    STAGE_SAMPLE_MULTIPLIER,
    STAGE_SAMPLE_DENOMINATOR,
)

logger = logging.getLogger(__name__)

_DIALOGUE_RE = re.compile(
    r'[""\u201c\u201d\u300c\u300e\u0022](.*?)[""\u201d\u201c\u300d\u300f\u0022]',
    re.DOTALL,
)


class StageF(BaseStage):
    """Stage F: 对话/描写/动作专项样本库"""

    def __init__(self, book_name: str, category: str):
        super().__init__("F", book_name, category)

    @staticmethod
    def _compute_chunk_score(text: str) -> float:
        """计算 chunk 综合质量预评分（0-1）"""
        if not text:
            return 0.0
        total = len(text)
        dialogue_chars = sum(len(m) for m in _DIALOGUE_RE.findall(text))
        dialogue_density = min(1.0, dialogue_chars / max(total, 1) * 3)
        emotion_count = sum(1 for c in text if c in ("！", "？", "…"))
        emotion_density = min(1.0, emotion_count / max(total, 1) * 200)
        unique_ratio = len(set(text)) / max(total, 1)
        vocab_richness = min(1.0, unique_ratio * 8)
        return dialogue_density * 0.4 + emotion_density * 0.3 + vocab_richness * 0.3

    def _select_sample_chapters(self, chapters):
        """区间择优采样：均分区间，每区间取评分最高 chunk"""
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
        if total <= sample_count:
            return chapters
        interval = total / sample_count
        selected = []
        for i in range(sample_count):
            start = int(i * interval)
            end = int((i + 1) * interval) if i < sample_count - 1 else total
            best_idx, best_score = start, -1
            for j in range(start, min(end, total)):
                score = self._compute_chunk_score(chapters[j].get("text", ""))
                if score > best_score:
                    best_score, best_idx = score, j
            selected.append(chapters[best_idx])
        return selected

    def _extract_basic_samples(
        self, text: str, chap_id: str, stage_result=None
    ) -> Dict[str, List[Dict]]:
        """批次1：提取基础样本（对话+描写+转场）"""
        result = {
            "dialogue_samples": [],
            "description_samples": [],
            "transition_samples": [],
        }

        prompt = f"""你是顶级的文学编辑与写作教练。请从以下章节文本中，提取高质量的【对话样本】、【描写样本】和【转场样本】，作为写作参考的 Few-Shot 典例。

【书名】{self.book_name} 【分类】{self.category}
【章节】{chap_id}
【正文】
{text}

请输出纯 JSON 格式：
{{
  "dialogue_samples": [
    {{
      "scene_type": "对话场景类型(争吵/告白/谈判/日常闲聊/师徒教导/威胁恐吓/幽默调侃/哲学讨论)",
      "original_text": "从原文中原封不动摘录的完整对话段落（200-400字，包含说话人标识和对话内容）",
      "emotional_tension": "情绪张力分析(如:表面平静暗流涌动/激烈对抗/温馨感人，30字内)",
      "subtext": "潜台词分析(角色真正想表达但未明说的内容，50字内)",
      "plot_function": "对话推动剧情的作用(如:揭露秘密/建立联盟/引发冲突/传递信息，30字内)",
      "writing_quality": 8
    }}
  ],
  "description_samples": [
    {{
      "description_type": "描写类型(打斗动作/环境氛围/心理活动/外貌特征/细节特写)",
      "original_text": "从原文中原封不动摘录的完整描写段落（200-400字）",
      "technique_analysis": "技法分析(如:动词精准/五感并用/意识流/白描/工笔，50字内)",
      "sensory_details": "感官细节(视觉/听觉/嗅觉/触觉/味觉的运用，50字内)",
      "writing_quality": 8
    }}
  ],
  "transition_samples": [
    {{
      "transition_type": "转场类型(场景切换/时间跳跃/视角切换/蒙太奇)",
      "original_text": "从原文中原封不动摘录的转场段落（100-300字，包含转场前后的衔接）",
      "technique_analysis": "转场技法分析(如:空行分隔/时间标记/视角切换/意象过渡，50字内)",
      "writing_quality": 8
    }}
  ]
}}
(⚠️核心要求：
1. **所有原文摘录必须原封不动复制，禁止任何改写、缩写、扩写！** 这是最重要的要求！
2. 每个摘录必须附带 writing_quality（写作质量1-10）
   writing_quality标准：仅用于同书内相对排序。7分=本章中优秀，5分=普通，3分=流水账。不同书的分数不可比较
   ⚠️ writing_quality必须如实评估，禁止全部给8-10分。评分仅供同书内排序，不跨书比较！
3. 每章最多提取 2-3 个最典型的对话样本、2-3 个描写样本、1-2 个转场样本
4. 优先选择包含完整情境、情绪转变、技法突出的段落
5. 转场样本要特别关注章节之间、场景之间的过渡手法
6. 如果没有合适的样本，对应数组留空
7. 禁止使用反引号，必须输出合法JSON)"""

        try:
            resp = ollama_chat(prompt, 0.2, "F")
            data = safe_parse_json(resp)
            if data:
                for ds in data.get("dialogue_samples", []):
                    if isinstance(ds, dict) and ds.get("original_text"):
                        result["dialogue_samples"].append(
                            {
                                "book_name": self.book_name,
                                "chapter_id": chap_id,
                                "scene_type": ds.get("scene_type", "未知"),
                                "original_text": ds.get("original_text"),
                                "emotional_tension": ds.get("emotional_tension", ""),
                                "subtext": ds.get("subtext", ""),
                                "plot_function": ds.get("plot_function", ""),
                                "writing_quality": ds.get("writing_quality", 5),
                            }
                        )

                for desc in data.get("description_samples", []):
                    if isinstance(desc, dict) and desc.get("original_text"):
                        result["description_samples"].append(
                            {
                                "book_name": self.book_name,
                                "chapter_id": chap_id,
                                "description_type": desc.get(
                                    "description_type", "未知"
                                ),
                                "original_text": desc.get("original_text"),
                                "technique_analysis": desc.get(
                                    "technique_analysis", ""
                                ),
                                "sensory_details": desc.get("sensory_details", ""),
                                "writing_quality": desc.get("writing_quality", 5),
                            }
                        )

                for trans in data.get("transition_samples", []):
                    if isinstance(trans, dict) and trans.get("original_text"):
                        result["transition_samples"].append(
                            {
                                "book_name": self.book_name,
                                "chapter_id": chap_id,
                                "transition_type": trans.get("transition_type", "未知"),
                                "original_text": trans.get("original_text"),
                                "technique_analysis": trans.get(
                                    "technique_analysis", ""
                                ),
                                "writing_quality": trans.get("writing_quality", 5),
                            }
                        )
        except Exception as e:
            logger.warning(f"⚠️ [阶段F-基础样本] 解析章节 {chap_id} 失败: {e}")
            if stage_result:
                stage_result.add_failure(chap_id, str(e), "F-basic")

        return result

    def _extract_advanced_samples(
        self, text: str, chap_id: str, stage_result=None
    ) -> Dict[str, List[Dict]]:
        """批次2：提取进阶分析（叙事距离+Show/Tell+动作场景+高潮段落+金句）"""
        result = {
            "narrative_distance": [],
            "show_tell_patterns": [],
            "action_scene_samples": [],
            "climax_excerpts": [],
            "memorable_quotes": [],
        }

        prompt = f"""你是顶级的文学编辑与写作教练。请从以下章节文本中，提取高质量的【叙事距离控制】、【Show vs Tell 策略】、【动作场景范文】、【高潮段落】和【金句名句】。

【书名】{self.book_name} 【分类】{self.category}
【章节】{chap_id}
【正文】
{text}

请输出纯 JSON 格式：
{{
  "narrative_distance": [
    {{
      "distance_type": "叙事距离类型(贴近内心/中等距离/全知鸟瞰)",
      "trigger_reason": "触发距离变化的原因(如:情感高潮/信息揭露/场景转换，50字内)",
      "original_example": "原文示例(摘录体现该距离的关键段落，100-200字)",
      "writing_quality": 8
    }}
  ],
  "show_tell_patterns": [
    {{
      "pattern_type": "模式类型(Show为主/Tell为主/混合)",
            "switching_triggers": "切换时机(如:情感场景用Show/背景介绍用Tell，50字内)",
      "original_example": "原文示例(摘录体现该模式的关键段落，100-200字)",
      "writing_quality": 8
    }}
  ],
  "action_scene_samples": [
    {{
      "action_type": "动作场景类型(打斗战斗/追逐逃亡/竞技比赛/武打招式/魔法对决)",
      "original_text": "从原文中原封不动摘录的完整动作场景段落（200-400字）",
      "technique_analysis": "技法分析(如:短句快节奏/动词精准/感官并用/视角切换，50字内)",
      "pacing_analysis": "节奏控制分析(如:快慢交替/逐步加速/爆发式高潮，50字内)",
      "sensory_details": "感官细节(视觉/听觉/触觉的运用，50字内)",
      "writing_quality": 8
    }}
  ],
  "climax_excerpts": [
    {{
      "excerpt_type": "高潮段落类型(决战/揭秘/情感爆发/逆转/生死抉择)",
      "original_text": "从原文中原封不动摘录的高潮段落（200-400字）",
      "technique_analysis": "技法分析(如:短句爆发/感官轰炸/情感渲染/悬念释放，50字内)",
      "emotional_impact": "情感冲击力分析(如:震撼/感动/紧张/释然，50字内)",
      "writing_quality": 8
    }}
  ],
  "memorable_quotes": [
    {{
      "quote_text": "金句/名句原文（50-150字）",
      "context": "上下文背景(如:主角在绝境中的感悟/角色间的哲学对话，50字内)",
      "technique_analysis": "技法分析(如:对比/排比/隐喻/哲理/金句结构，50字内)",
      "quote_type": "金句类型(哲理句/经典台词/情感金句/励志金句/讽刺金句)",
      "writing_quality": 8
    }}
  ]
}}
(⚠️核心要求：
1. **所有原文摘录必须原封不动复制，禁止任何改写、缩写、扩写！** 这是最重要的要求！
2. 每个摘录必须附带 writing_quality（写作质量1-10）
   writing_quality标准：仅用于同书内相对排序。7分=本章中优秀，5分=普通，3分=流水账。不同书的分数不可比较
   ⚠️ writing_quality必须如实评估，禁止全部给8-10分。评分仅供同书内排序，不跨书比较！
3. 必须分析叙事距离的变化（何时贴近内心、何时拉远鸟瞰）！
4. 必须分析 Show vs Tell 的比例和切换时机！
5. 如果本章有动作/战斗/追逐/竞技场景，必须提取高质量的动作场景范文！没有则返回空数组！
6. 如果本章包含高潮段落或名场面（决战/揭秘/情感爆发/逆转），必须提取原文并分析！没有则返回空数组！
7. 必须提取本章中最精彩的1-2句话（金句/名句/哲理句/经典台词）！没有则返回空数组！
8. 禁止使用反引号，必须输出合法JSON)"""

        try:
            resp = ollama_chat(prompt, 0.2, "F")
            data = safe_parse_json(resp)
            if data:
                for nd in data.get("narrative_distance", []):
                    if (
                        isinstance(nd, dict)
                        and nd.get("distance_type")
                        and nd.get("original_example")
                    ):
                        result["narrative_distance"].append(
                            {
                                "book_name": self.book_name,
                                "chapter_id": chap_id,
                                "distance_type": nd.get("distance_type"),
                                "trigger_reason": nd.get("trigger_reason", ""),
                                "original_example": nd.get("original_example"),
                                "writing_quality": nd.get("writing_quality", 5),
                            }
                        )

                for st in data.get("show_tell_patterns", []):
                    if (
                        isinstance(st, dict)
                        and st.get("pattern_type")
                        and st.get("original_example")
                    ):
                        result["show_tell_patterns"].append(
                            {
                                "book_name": self.book_name,
                                "chapter_id": chap_id,
                                "pattern_type": st.get("pattern_type"),
                                "switching_triggers": st.get("switching_triggers", ""),
                                "original_example": st.get("original_example"),
                                "writing_quality": st.get("writing_quality", 5),
                            }
                        )

                for action in data.get("action_scene_samples", []):
                    if isinstance(action, dict) and action.get("original_text"):
                        result["action_scene_samples"].append(
                            {
                                "book_name": self.book_name,
                                "chapter_id": chap_id,
                                "action_type": action.get("action_type", "未知"),
                                "original_text": action.get("original_text"),
                                "technique_analysis": action.get(
                                    "technique_analysis", ""
                                ),
                                "pacing_analysis": action.get("pacing_analysis", ""),
                                "sensory_details": action.get("sensory_details", ""),
                            }
                        )

                for climax in data.get("climax_excerpts", []):
                    if isinstance(climax, dict) and climax.get("original_text"):
                        result["climax_excerpts"].append(
                            {
                                "book_name": self.book_name,
                                "chapter_id": chap_id,
                                "excerpt_type": climax.get("excerpt_type", "未知"),
                                "original_text": climax.get("original_text"),
                                "technique_analysis": climax.get(
                                    "technique_analysis", ""
                                ),
                                "emotional_impact": climax.get("emotional_impact", ""),
                            }
                        )

                for quote in data.get("memorable_quotes", []):
                    if isinstance(quote, dict) and quote.get("quote_text"):
                        result["memorable_quotes"].append(
                            {
                                "book_name": self.book_name,
                                "chapter_id": chap_id,
                                "quote_text": quote.get("quote_text"),
                                "context": quote.get("context", ""),
                                "technique_analysis": quote.get(
                                    "technique_analysis", ""
                                ),
                                "quote_type": quote.get("quote_type", ""),
                            }
                        )
                        # 验证：金句必须在原文中存在
                        quote_text_val = quote.get("quote_text", "")
                        # 简化的存在性检测（NFKC + 去空格，跳过过短片段）
                        tq = unicodedata.normalize("NFKC", quote_text_val.strip().replace(" ", "").replace("\n", ""))
                        tt = unicodedata.normalize("NFKC", text.strip().replace(" ", "").replace("\n", ""))
                        if len(tq) >= 8 and tq not in tt:
                            result.setdefault("_unverified_quotes", []).append(chap_id)
        except Exception as e:
            logger.warning(f"⚠️ [阶段F-进阶样本] 解析章节 {chap_id} 失败: {e}")
            if stage_result:
                stage_result.add_failure(chap_id, str(e), "F-advanced")

        return result

    def _process_single_chapter(self, chap: Dict) -> Dict[str, Any]:
        """
        处理单章：提取基础样本 + 进阶样本 + 开头结尾（线程安全，无共享状态写入）
        """
        text = chap["text"]
        chap_id = chap.get("id", "未知章节")

        # 开头/结尾截取（不依赖 LLM）
        opening_text = text[:200] if len(text) >= 200 else text
        ending_text = text[-200:] if len(text) >= 200 else text

        chapter_result = {
            "chapter_opening_ending_samples": [
                {
                    "book_name": self.book_name,
                    "chapter_id": chap_id,
                    "sample_position": "opening",
                    "original_text": opening_text,
                    "technique_analysis": "",
                    "hook_type": "",
                },
                {
                    "book_name": self.book_name,
                    "chapter_id": chap_id,
                    "sample_position": "ending",
                    "original_text": ending_text,
                    "technique_analysis": "",
                    "hook_type": "",
                },
            ],
            "dialogue_samples": [],
            "description_samples": [],
            "transition_samples": [],
            "narrative_distance": [],
            "show_tell_patterns": [],
            "action_scene_samples": [],
            "climax_excerpts": [],
            "memorable_quotes": [],
        }

        # 批次1：基础样本（对话+描写+转场）
        basic_data = self._extract_basic_samples(text, chap_id)
        chapter_result["dialogue_samples"] = basic_data["dialogue_samples"]
        chapter_result["description_samples"] = basic_data["description_samples"]
        chapter_result["transition_samples"] = basic_data["transition_samples"]

        # 批次2：进阶分析（叙事距离+Show/Tell+动作场景+高潮段落+金句）
        advanced_data = self._extract_advanced_samples(text, chap_id)
        chapter_result["narrative_distance"] = advanced_data["narrative_distance"]
        chapter_result["show_tell_patterns"] = advanced_data["show_tell_patterns"]
        chapter_result["action_scene_samples"] = advanced_data["action_scene_samples"]
        chapter_result["climax_excerpts"] = advanced_data["climax_excerpts"]
        chapter_result["memorable_quotes"] = advanced_data["memorable_quotes"]

        return chapter_result

    def run(self, chapters: List[Dict], **kwargs) -> Dict[str, List[Dict]]:
        """
        执行 Stage F（并行处理章节，利用 STAGE_F_WORKERS 并发，支持断点续跑）

        Args:
            chapters: 章节列表

        Returns:
            包含 dialogue_samples, description_samples, transition_samples, style_summaries, narrative_distance, show_tell_patterns 的字典
        """
        logger.info(f"=== 阶段六：对话/描写/动作专项样本库提取 ({self.book_name}) ===")
        stage_result = StageResult()

        # 数据键列表（用于合并结果）
        DATA_KEYS = [
            "dialogue_samples",
            "description_samples",
            "transition_samples",
            "narrative_distance",
            "show_tell_patterns",
            "action_scene_samples",
            "climax_excerpts",
            "memorable_quotes",
            "chapter_opening_ending_samples",
        ]

        result = {key: [] for key in DATA_KEYS}
        result["style_summaries"] = []
        result["_chapter_summaries"] = {
            chap.get("id", ""): chap.get("summary", "")
            for chap in chapters
            if chap.get("id")
        }

        # 断点恢复：加载已完成的章节结果
        cache = self.load_cache()
        completed_items = cache.get("data", []) if cache else []
        completed_ids = {
            item.get("_chapter_id", "")
            for item in completed_items
            if item.get("_chapter_id")
        }

        # 合并缓存数据到主结果
        for item in completed_items:
            for key in DATA_KEYS:
                result[key].extend(item.get(key, []))

        if completed_ids:
            logger.info(
                f"✅ [阶段F] 恢复断点：已完成 {len(completed_ids)}/{len(chapters)} 章"
            )

        # 区间择优采样（--full 模式下全量处理）
        if os.environ.get("NOVEL_KB_FULL_SAMPLE"):
            sampled = chapters
            logger.info(f"[阶段F] --full 全量模式: {len(chapters)} 章")
        else:
            sampled = self._select_sample_chapters(chapters)
            logger.info(
                f"[阶段F] 区间择优采样: {len(sampled)}/{len(chapters)} 章 (采样率 {len(sampled)*100//max(len(chapters),1)}%)"
            )
        pending = [c for c in sampled if c.get("id") not in completed_ids]
        if not pending:
            logger.info(f"[阶段F] 所有章节已处理完毕")
        else:
            # 并行处理未完成章节
            workers = min(STAGE_F_WORKERS, len(pending))
            if os.environ.get("STAGE_F_INTERNAL_SINGLE"):
                workers = 1
            logger.info(
                f"[阶段F] 使用 {workers} 个并发 worker 处理剩余 {len(pending)} 章"
            )

            processed_count = 0
            checkpoint_interval = 10  # 每 10 章保存一次断点

            with ThreadPoolExecutor(max_workers=max(workers, 1)) as executor:
                futures = {
                    executor.submit(self._process_single_chapter, chap): chap.get(
                        "id", "unknown"
                    )
                    for chap in pending
                }

                try:
                    for future in tqdm(
                        as_completed(futures), total=len(futures), desc="提取样本库"
                    ):
                        chap_id = futures[future]
                        try:
                            chapter_result = future.result()
                            chapter_result["_chapter_id"] = chap_id
                            completed_items.append(chapter_result)
                            # 合并到主结果
                            for key in DATA_KEYS:
                                result[key].extend(chapter_result.get(key, []))
                            processed_count += 1
                            # 定期保存断点
                            if processed_count % checkpoint_interval == 0:
                                self.save_cache({"data": completed_items})
                        except Exception as e:
                            logger.warning(f"⚠️ [阶段F] 章节 {chap_id} 处理失败: {e}")
                            stage_result.add_failure(chap_id, str(e), "F")
                except KeyboardInterrupt:
                    # 中断时立即保存进度
                    logger.info(
                        f"[阶段F] 用户中断，保存进度 ({len(completed_items)} 章已完成)..."
                    )
                    self.save_cache({"data": completed_items})
                    raise

            # 最终保存断点
            self.save_cache({"data": completed_items})

        # 生成风格总结（基于已提取的样本）
        logger.info(f"📊 [阶段F] 生成风格总结...")
        result["style_summaries"] = self._generate_style_summaries(result)

        logger.info(
            f"✅ [阶段F战报] 对话样本: {len(result['dialogue_samples'])} 条, "
            f"描写样本: {len(result['description_samples'])} 条, "
            f"转场样本: {len(result['transition_samples'])} 条, "
            f"风格总结: {len(result['style_summaries'])} 条, "
            f"叙事距离: {len(result['narrative_distance'])} 条, "
            f"Show/Tell: {len(result['show_tell_patterns'])} 条, "
            f"动作场景: {len(result['action_scene_samples'])} 条, "
            f"高潮段落: {len(result['climax_excerpts'])} 条, "
            f"开头结尾: {len(result['chapter_opening_ending_samples'])} 条, "
            f"金句名句: {len(result['memorable_quotes'])} 条"
        )
        stage_result.data = result
        summary = stage_result.get_summary()
        if summary["failure_count"] > 0:
            logger.warning(f"⚠️ [阶段F] 有 {summary['failure_count']} 个章节处理失败")
        return result

    def _generate_style_summaries(self, result: Dict) -> List[Dict]:
        """
        基于已提取的样本，生成风格总结
        对每种对话场景和描写类型，总结该书的风格特点
        """
        style_summaries = []

        # 对话风格总结
        dialogue_by_scene = {}
        for ds in result.get("dialogue_samples", []):
            scene = ds.get("scene_type", "未知")
            if scene not in dialogue_by_scene:
                dialogue_by_scene[scene] = []
            dialogue_by_scene[scene].append(ds)

        for scene, samples in dialogue_by_scene.items():
            if len(samples) < 2:
                continue

            # 简单统计特征
            avg_length = sum(len(s.get("original_text", "")) for s in samples) / len(
                samples
            )
            tension_types = [
                s.get("emotional_tension", "")
                for s in samples
                if s.get("emotional_tension")
            ]

            style_desc = f"本书的{scene}对话特点：平均长度{int(avg_length)}字，"
            if tension_types:
                style_desc += f"情绪张力类型包括：{'、'.join(tension_types[:3])}"

            style_summaries.append(
                {
                    "book_name": self.book_name,
                    "category": self.category,
                    "summary_type": "dialogue",
                    "scene_or_desc_type": scene,
                    "style_description": style_desc,
                    "key_features": f"样本数:{len(samples)}",
                }
            )

        # 描写风格总结
        desc_by_type = {}
        for desc in result.get("description_samples", []):
            desc_type = desc.get("description_type", "未知")
            if desc_type not in desc_by_type:
                desc_by_type[desc_type] = []
            desc_by_type[desc_type].append(desc)

        for desc_type, samples in desc_by_type.items():
            if len(samples) < 2:
                continue

            avg_length = sum(len(s.get("original_text", "")) for s in samples) / len(
                samples
            )
            techniques = [
                s.get("technique_analysis", "")
                for s in samples
                if s.get("technique_analysis")
            ]

            style_desc = f"本书的{desc_type}描写特点：平均长度{int(avg_length)}字，"
            if techniques:
                style_desc += f"常用技法：{'、'.join(techniques[:3])}"

            style_summaries.append(
                {
                    "book_name": self.book_name,
                    "category": self.category,
                    "summary_type": "description",
                    "scene_or_desc_type": desc_type,
                    "style_description": style_desc,
                    "key_features": f"样本数:{len(samples)}",
                }
            )

        # 转场风格总结
        transitions = result.get("transition_samples", [])
        if len(transitions) >= 2:
            trans_types = [
                t.get("transition_type", "")
                for t in transitions
                if t.get("transition_type")
            ]
            techniques = [
                t.get("technique_analysis", "")
                for t in transitions
                if t.get("technique_analysis")
            ]

            style_desc = (
                f"本书的转场手法：常用类型包括{'、'.join(set(trans_types[:5]))}，"
            )
            if techniques:
                style_desc += f"技法特点：{'、'.join(techniques[:3])}"

            style_summaries.append(
                {
                    "book_name": self.book_name,
                    "category": self.category,
                    "summary_type": "transition",
                    "scene_or_desc_type": "转场",
                    "style_description": style_desc,
                    "key_features": f"样本数:{len(transitions)}",
                }
            )

        return style_summaries

    def insert(self, results: Dict[str, List[Dict]]) -> Dict[str, int]:
        """将 Stage F 结果写入数据库"""
        cursor = self.db.connect().cursor()
        stats = {
            "dialogue_samples": 0,
            "description_samples": 0,
            "transition_samples": 0,
            "style_summaries": 0,
            "narrative_distance": 0,
            "show_tell_patterns": 0,
            "action_scene_samples": 0,
            "climax_excerpts": 0,
            "chapter_opening_ending_samples": 0,
            "memorable_quotes": 0,
        }

        # 章节摘要索引：用于为 ChromaDB 样本附带剧情上下文
        chapter_summaries = results.get("_chapter_summaries", {})

        def get_chapter_context(chapter_id: str) -> str:
            """获取章节的剧情摘要，作为样本的上下文"""
            summary = chapter_summaries.get(chapter_id, "")
            if summary and len(summary) > 150:
                summary = summary[:150] + "..."
            return summary

        # 质量筛选：按 writing_quality 排序，每种类型只保留 Top-N
        def _filter_quality(samples, key_field, top_n):
            if not samples or len(samples) <= top_n:
                return samples
            return sorted(
                samples, key=lambda x: x.get("writing_quality", 5), reverse=True
            )[:top_n]

        # 按场景/类型分组过滤
        dialogue_by_type = {}
        for ds in results.get("dialogue_samples", []):
            t = ds.get("scene_type", "未知")
            dialogue_by_type.setdefault(t, []).append(ds)
        results["dialogue_samples"] = [
            s
            for samples in dialogue_by_type.values()
            for s in _filter_quality(samples, "scene_type", 3)
        ]
        desc_by_type = {}
        for ds in results.get("description_samples", []):
            t = ds.get("description_type", "未知")
            desc_by_type.setdefault(t, []).append(ds)
        results["description_samples"] = [
            s
            for samples in desc_by_type.values()
            for s in _filter_quality(samples, "description_type", 3)
        ]
        results["transition_samples"] = _filter_quality(
            results.get("transition_samples", []), "transition_type", 5
        )
        results["narrative_distance"] = _filter_quality(
            results.get("narrative_distance", []), "distance_type", 5
        )
        results["show_tell_patterns"] = _filter_quality(
            results.get("show_tell_patterns", []), "pattern_type", 5
        )
        results["action_scene_samples"] = _filter_quality(
            results.get("action_scene_samples", []), "action_type", 5
        )
        results["climax_excerpts"] = _filter_quality(
            results.get("climax_excerpts", []), "excerpt_type", 5
        )
        results["memorable_quotes"] = _filter_quality(
            results.get("memorable_quotes", []), "quote_type", 10
        )

        # 对话样本入库
        for ds in results.get("dialogue_samples", []):
            ds_id = generate_id(
                ds["book_name"],
                ds["chapter_id"],
                ds["scene_type"],
            )
            cursor.execute(
                "INSERT OR REPLACE INTO dialogue_samples VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    ds_id,
                    ds["book_name"],
                    ds["chapter_id"],
                    ds["scene_type"],
                    ds["original_text"],
                    ds.get("emotional_tension", ""),
                    ds.get("subtext", ""),
                    ds.get("plot_function", ""),
                    ds.get("writing_quality", 5),
                ),
            )
            stats["dialogue_samples"] += 1

        # ChromaDB: 对话样本（浅拷贝，不污染原始数据，附带章节上下文）
        chroma_dialogue_items = []
        for ds in results.get("dialogue_samples", []):
            chroma_item = {**ds}
            ctx = get_chapter_context(ds.get("chapter_id", ""))
            chroma_item["_chroma_text"] = (
                f"场景:{ds['scene_type']}\n对话:{ds['original_text']}\n"
                f"情绪张力:{ds.get('emotional_tension', '')}\n"
                f"潜台词:{ds.get('subtext', '')}\n"
                f"剧情作用:{ds.get('plot_function', '')}"
                + (f"\n剧情上下文:{ctx}" if ctx else "")
            )
            chroma_dialogue_items.append(chroma_item)
        bulk_upsert_to_chroma(
            "dialogue_samples_kb",
            chroma_dialogue_items,
            id_fields=["book_name", "chapter_id", "scene_type"],  # 确定性ID,
            text_field="_chroma_text",
            metadata_fields=["book_name", "chapter_id", "scene_type"],
        )

        # 描写样本入库
        for desc in results.get("description_samples", []):
            desc_id = generate_id(
                desc["book_name"],
                desc["chapter_id"],
                desc["description_type"],
            )
            cursor.execute(
                "INSERT OR REPLACE INTO description_samples VALUES (?,?,?,?,?,?,?,?)",
                (
                    desc_id,
                    desc["book_name"],
                    desc["chapter_id"],
                    desc["description_type"],
                    desc["original_text"],
                    desc.get("technique_analysis", ""),
                    desc.get("sensory_details", ""),
                    desc.get("writing_quality", 5),
                ),
            )
            stats["description_samples"] += 1

        # ChromaDB: 描写样本（浅拷贝，不污染原始数据，附带章节上下文）
        chroma_desc_items = []
        for desc in results.get("description_samples", []):
            chroma_item = {**desc}
            ctx = get_chapter_context(desc.get("chapter_id", ""))
            chroma_item["_chroma_text"] = (
                f"类型:{desc['description_type']}\n描写:{desc['original_text']}\n"
                f"技法:{desc.get('technique_analysis', '')}\n"
                f"感官:{desc.get('sensory_details', '')}"
                + (f"\n剧情上下文:{ctx}" if ctx else "")
            )
            chroma_desc_items.append(chroma_item)
        bulk_upsert_to_chroma(
            "description_samples_kb",
            chroma_desc_items,
            id_fields=["book_name", "chapter_id", "description_type"],
            text_field="_chroma_text",
            metadata_fields=["book_name", "chapter_id", "description_type"],
        )

        # 转场样本入库
        for trans in results.get("transition_samples", []):
            trans_id = generate_id(
                trans["book_name"],
                trans["chapter_id"],
                trans["transition_type"],
            )
            cursor.execute(
                "INSERT OR REPLACE INTO transition_samples VALUES (?,?,?,?,?,?,?)",
                (
                    trans_id,
                    trans["book_name"],
                    trans["chapter_id"],
                    trans["transition_type"],
                    trans["original_text"],
                    trans.get("technique_analysis", ""),
                    trans.get("writing_quality", 5),
                ),
            )
            stats["transition_samples"] += 1

        # ChromaDB: 转场样本（浅拷贝，不污染原始数据，附带章节上下文）
        chroma_trans_items = []
        for trans in results.get("transition_samples", []):
            chroma_item = {**trans}
            ctx = get_chapter_context(trans.get("chapter_id", ""))
            chroma_item["_chroma_text"] = (
                f"转场类型:{trans['transition_type']}\n原文:{trans['original_text']}\n"
                f"技法:{trans.get('technique_analysis', '')}"
                + (f"\n剧情上下文:{ctx}" if ctx else "")
            )
            chroma_trans_items.append(chroma_item)
        bulk_upsert_to_chroma(
            "transition_samples_kb",
            chroma_trans_items,
            id_fields=["book_name", "chapter_id", "transition_type"],
            text_field="_chroma_text",
            metadata_fields=["book_name", "chapter_id", "transition_type"],
        )

        # 风格总结入库
        for ss in results.get("style_summaries", []):
            ss_id = generate_id(
                ss["book_name"], ss["summary_type"], ss["scene_or_desc_type"]
            )
            cursor.execute(
                "INSERT OR REPLACE INTO style_summaries VALUES (?,?,?,?,?,?,?)",
                (
                    ss_id,
                    ss["book_name"],
                    ss["category"],
                    ss["summary_type"],
                    ss["scene_or_desc_type"],
                    ss["style_description"],
                    ss.get("key_features", ""),
                ),
            )
            stats["style_summaries"] += 1

        # 叙事距离控制入库
        for nd in results.get("narrative_distance", []):
            nd_id = generate_id(nd["book_name"], nd["chapter_id"], nd["distance_type"])
            cursor.execute(
                "INSERT OR REPLACE INTO narrative_distance VALUES (?,?,?,?,?,?,?)",
                (
                    nd_id,
                    nd["book_name"],
                    nd["chapter_id"],
                    nd["distance_type"],
                    nd.get("trigger_reason", ""),
                    nd.get("original_example", ""),
                    nd.get("writing_quality", 5),
                ),
            )
            stats["narrative_distance"] += 1

        # Show vs Tell 策略入库
        for st in results.get("show_tell_patterns", []):
            st_id = generate_id(st["book_name"], st["chapter_id"], st["pattern_type"])
            cursor.execute(
                "INSERT OR REPLACE INTO show_tell_patterns VALUES (?,?,?,?,?,?,?)",
                (
                    st_id,
                    st["book_name"],
                    st["chapter_id"],
                    st["pattern_type"],
                    st.get("switching_triggers", ""),
                    st.get("original_example", ""),
                    st.get("writing_quality", 5),
                ),
            )
            stats["show_tell_patterns"] += 1

        # 动作/战斗场景范文入库
        for action in results.get("action_scene_samples", []):
            action_id = generate_id(
                action["book_name"],
                action["chapter_id"],
                action["action_type"],
            )
            cursor.execute(
                "INSERT OR REPLACE INTO action_scene_samples VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    action_id,
                    action["book_name"],
                    action["chapter_id"],
                    action["action_type"],
                    action["original_text"],
                    action.get("technique_analysis", ""),
                    action.get("pacing_analysis", ""),
                    action.get("sensory_details", ""),
                    action.get("writing_quality", 5),
                ),
            )
            stats["action_scene_samples"] += 1

        # ChromaDB: 动作场景样本（附带章节上下文）
        a_ids, a_docs, a_metas = [], [], []
        for action in results.get("action_scene_samples", []):
            aid = generate_id(
                action["book_name"],
                action["chapter_id"],
                action["action_type"],
            )
            a_ids.append(aid)
            ctx = get_chapter_context(action.get("chapter_id", ""))
            a_docs.append(
                f"动作类型:{action['action_type']}\n原文:{action['original_text']}\n"
                f"技法:{action.get('technique_analysis', '')}\n"
                f"节奏:{action.get('pacing_analysis', '')}"
                + (f"\n剧情上下文:{ctx}" if ctx else "")
            )
            a_metas.append(
                {
                    "book_name": action["book_name"],
                    "chapter_id": action["chapter_id"],
                    "action_type": action["action_type"],
                }
            )
        if a_ids:
            self.chroma.upsert_batch("action_scene_samples_kb", a_ids, a_docs, a_metas)

        # 高潮段落/名场面原文入库
        for climax in results.get("climax_excerpts", []):
            climax_id = generate_id(
                climax["book_name"],
                climax["chapter_id"],
                climax["excerpt_type"],
            )
            cursor.execute(
                "INSERT OR REPLACE INTO climax_excerpts VALUES (?,?,?,?,?,?,?,?)",
                (
                    climax_id,
                    climax["book_name"],
                    climax["chapter_id"],
                    climax["excerpt_type"],
                    climax["original_text"],
                    climax.get("technique_analysis", ""),
                    climax.get("emotional_impact", ""),
                    climax.get("writing_quality", 5),
                ),
            )
            stats["climax_excerpts"] += 1

        # ChromaDB: 高潮段落（附带章节上下文）
        c_ids, c_docs, c_metas = [], [], []
        for climax in results.get("climax_excerpts", []):
            cid = generate_id(
                climax["book_name"],
                climax["chapter_id"],
                climax["excerpt_type"],
            )
            c_ids.append(cid)
            ctx = get_chapter_context(climax.get("chapter_id", ""))
            c_docs.append(
                f"高潮类型:{climax['excerpt_type']}\n原文:{climax['original_text']}\n"
                f"技法:{climax.get('technique_analysis', '')}\n"
                f"情感冲击:{climax.get('emotional_impact', '')}"
                + (f"\n剧情上下文:{ctx}" if ctx else "")
            )
            c_metas.append(
                {
                    "book_name": climax["book_name"],
                    "chapter_id": climax["chapter_id"],
                    "excerpt_type": climax["excerpt_type"],
                }
            )
        if c_ids:
            self.chroma.upsert_batch("climax_excerpts_kb", c_ids, c_docs, c_metas)

        # 章节开头/结尾范文入库
        for oe in results.get("chapter_opening_ending_samples", []):
            oe_id = generate_id(
                oe["book_name"],
                oe["chapter_id"],
                oe["sample_position"],
            )
            cursor.execute(
                "INSERT OR REPLACE INTO chapter_opening_ending_samples VALUES (?,?,?,?,?,?,?)",
                (
                    oe_id,
                    oe["book_name"],
                    oe["chapter_id"],
                    oe["sample_position"],
                    oe["original_text"],
                    oe.get("technique_analysis", ""),
                    oe.get("hook_type", ""),
                ),
            )
            stats["chapter_opening_ending_samples"] += 1

        # 金句/名句入库
        for quote in results.get("memorable_quotes", []):
            quote_id = generate_id(
                quote["book_name"],
                quote["chapter_id"],
                quote["quote_type"],
            )
            cursor.execute(
                "INSERT OR REPLACE INTO memorable_quotes VALUES (?,?,?,?,?,?,?,?)",
                (
                    quote_id,
                    quote["book_name"],
                    quote["chapter_id"],
                    quote["quote_text"],
                    quote.get("context", ""),
                    quote.get("technique_analysis", ""),
                    quote.get("quote_type", ""),
                    quote.get("writing_quality", 5),
                ),
            )
            stats["memorable_quotes"] += 1

        # ChromaDB: 金句/名句
        q_ids, q_docs, q_metas = [], [], []
        for quote in results.get("memorable_quotes", []):
            qid = generate_id(
                quote["book_name"],
                quote["chapter_id"],
                quote["quote_type"],
            )
            q_ids.append(qid)
            ctx = get_chapter_context(quote.get("chapter_id", ""))
            q_docs.append(
                f"金句:{quote['quote_text']}\n"
                f"上下文:{quote.get('context', '')}\n"
                f"技法:{quote.get('technique_analysis', '')}\n"
                f"类型:{quote.get('quote_type', '')}"
                + (f"\n剧情上下文:{ctx}" if ctx else "")
            )
            q_metas.append(
                {
                    "book_name": quote["book_name"],
                    "chapter_id": quote["chapter_id"],
                    "quote_type": quote.get("quote_type", ""),
                }
            )
        if q_ids:
            self.chroma.upsert_batch("memorable_quotes_kb", q_ids, q_docs, q_metas)

        # 金句验证汇总
        unverified = results.get("_unverified_quotes", [])
        total_quotes = len(results.get("memorable_quotes", []))
        if unverified:
            logger.warning(
                f"📊 [阶段F] 金句抽查: {len(unverified)}/{total_quotes} 条未在原文匹配 "
                f"({len(unverified)*100//max(total_quotes,1)}%), 章节: {unverified[:5]}..."
            )

        self.db.commit()
        logger.info(
            f"   ✅ [阶段F战报] 对话样本: {stats['dialogue_samples']} | "
            f"描写样本: {stats['description_samples']} | "
            f"转场样本: {stats['transition_samples']} | "
            f"风格总结: {stats['style_summaries']} | "
            f"叙事距离: {stats['narrative_distance']} | "
            f"Show/Tell: {stats['show_tell_patterns']} | "
            f"动作场景: {stats['action_scene_samples']} | "
            f"高潮段落: {stats['climax_excerpts']} | "
            f"开头结尾: {stats['chapter_opening_ending_samples']} | "
            f"金句名句: {stats['memorable_quotes']}"
        )
        return stats
