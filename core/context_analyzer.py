"""
上下文分析模块
用 LLM 分析创作上下文，识别当前场景类型
输出结构化的"场景标签"，并映射到知识库查询策略
"""
import logging
from typing import Dict, Any, List, Optional
from core.ollama_client import ollama_chat, safe_parse_json

logger = logging.getLogger(__name__)


class ContextAnalyzer:
    """上下文分析器"""

    def __init__(self):
        pass

    def analyze_context(
        self,
        context_text: str,
        creation_stage: str,
        genre: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        分析创作上下文，识别当前场景

        Args:
            context_text: 当前正在创作的内容片段
            creation_stage: 创作阶段（世界观设计/人物设计/大纲设计/正文写作/修改打磨）
            genre: 作品类型（可选）

        Returns:
            场景分析结果，包含场景标签和查询策略
        """
        logger.info(f"=== 上下文分析 (阶段: {creation_stage}) ===")

        # 1. 调用 LLM 识别场景类型
        prompt = self._build_scene_detection_prompt(context_text, creation_stage, genre)

        try:
            resp = ollama_chat(prompt, 0.2, "CTX")
            data = safe_parse_json(resp)
            if not data:
                logger.warning("⚠️ [上下文分析] LLM 返回解析失败，使用默认策略")
                return self._get_default_result(creation_stage)

            # 2. 根据场景标签映射到查询策略
            query_strategy = self._map_to_query_strategy(data, creation_stage)

            result = {
                "detected_scene": data.get("scene_type", "未知"),
                "detected_sub_type": data.get("sub_type", ""),
                "detected_mood": data.get("mood", ""),
                "key_elements": data.get("key_elements", []),
                "query_strategy": query_strategy,
            }

            logger.info(
                f"✅ [上下文分析] 识别场景: {result['detected_scene']} | "
                f"子类型: {result['detected_sub_type']} | "
                f"情绪: {result['detected_mood']}"
            )
            return result

        except Exception as exc:
            logger.error(f"❌ [上下文分析] 分析失败: {exc}")
            return self._get_default_result(creation_stage)

    def _build_scene_detection_prompt(
        self,
        context_text: str,
        creation_stage: str,
        genre: Optional[str],
    ) -> str:
        """构建场景识别 Prompt"""
        genre_text = f"\n【作品类型】{genre}" if genre else ""

        return f"""你是专业的创作场景识别专家。请分析以下创作内容，识别当前的创作场景类型、子类型、情绪氛围和关键元素。

【创作阶段】{creation_stage}
【创作内容】
{context_text[:500]}{genre_text}

请输出纯 JSON 格式：
{{
  "scene_type": "场景大类(世界观设计/人物设计/大纲设计/正文写作/修改打磨)",
  "sub_type": "场景子类型(如:力量体系设计/角色对话/打斗场景/感情线对话/环境描写/高潮铺垫等)",
  "mood": "情绪氛围(如:紧张/轻松/悲伤/热血/悬疑/浪漫等)",
  "key_elements": ["关键元素1", "关键元素2", "关键元素3"],
  "confidence": 8
}}
(⚠️核心要求：
1. scene_type 必须是创作阶段之一！
2. sub_type 必须具体，能指导知识库查询！
3. mood 必须能从文本中推断出来！
4. key_elements 必须是文本中出现的关键概念或元素！
5. confidence 是识别置信度 1-10！
6. 禁止使用反引号，必须输出合法JSON)"""

    def _map_to_query_strategy(
        self,
        scene_data: Dict[str, Any],
        creation_stage: str,
    ) -> Dict[str, Any]:
        """根据场景标签映射到知识库查询策略"""
        scene_type = scene_data.get("scene_type", "")
        sub_type = scene_data.get("sub_type", "")
        mood = scene_data.get("mood", "")
        key_elements = scene_data.get("key_elements", [])

        strategy = {
            "excerpts": [],
            "techniques": [],
            "structure_tips": [],
            "common_mistakes": [],
        }

        # 根据创作阶段和子类型确定查询策略
        if creation_stage == "世界观设计" or sub_type in ["力量体系设计", "社会结构设计", "地理设计"]:
            strategy["excerpts"].append({"table": "world_settings", "query": " ".join(key_elements)})
            strategy["techniques"].append({"table": "genre_specific_techniques", "category": "设定设计"})
            strategy["structure_tips"].append({"table": "cross_book_comparisons", "dimension": "世界观设计"})

        elif creation_stage == "人物设计" or sub_type in ["角色对话", "人物塑造", "关系设计"]:
            strategy["excerpts"].append({"table": "dialogue_samples", "query": " ".join(key_elements)})
            strategy["techniques"].append({"table": "character_speech_style", "query": " ".join(key_elements)})
            strategy["structure_tips"].append({"table": "cross_book_comparisons", "dimension": "人物塑造"})
            strategy["common_mistakes"].append({"table": "common_mistakes", "dimension": "人物"})

        elif creation_stage == "大纲设计" or sub_type in ["大纲设计", "结构设计", "情节编排"]:
            strategy["structure_tips"].append({"table": "book_structure", "query": " ".join(key_elements)})
            strategy["structure_tips"].append({"table": "macro_outlines", "query": " ".join(key_elements)})
            strategy["techniques"].append({"table": "climax_buildup_chains", "query": " ".join(key_elements)})
            strategy["common_mistakes"].append({"table": "common_mistakes", "dimension": "情节"})

        elif creation_stage == "正文写作" or creation_stage == "修改打磨":
            # 根据子类型细分
            if sub_type in ["打斗场景", "动作场景", "战斗描写"]:
                strategy["excerpts"].append({"table": "action_scene_samples", "query": " ".join(key_elements)})
                strategy["excerpts"].append({"table": "climax_excerpts", "type": "打斗"})
                strategy["techniques"].append({"table": "technique_combinations", "scene_type": "打斗"})
                strategy["common_mistakes"].append({"table": "common_mistakes", "dimension": "节奏"})

            elif sub_type in ["感情线对话", "暧昧对话", "情感对话"]:
                strategy["excerpts"].append({"table": "dialogue_samples", "scene_type": "感情"})
                strategy["excerpts"].append({"table": "romance_lines", "query": " ".join(key_elements)})
                strategy["techniques"].append({"table": "technique_combinations", "scene_type": "对话"})
                strategy["common_mistakes"].append({"table": "common_mistakes", "dimension": "对话"})

            elif sub_type in ["环境描写", "氛围描写", "场景描写"]:
                strategy["excerpts"].append({"table": "description_samples", "query": " ".join(key_elements)})
                strategy["techniques"].append({"table": "sensory_mappings", "query": mood})
                strategy["common_mistakes"].append({"table": "common_mistakes", "dimension": "描写"})

            elif sub_type in ["高潮铺垫", "高潮场景", "决战", "揭秘"]:
                strategy["excerpts"].append({"table": "climax_excerpts", "query": " ".join(key_elements)})
                strategy["techniques"].append({"table": "climax_buildup_chains", "query": " ".join(key_elements)})
                strategy["techniques"].append({"table": "technique_combinations", "scene_type": "高潮"})
                strategy["common_mistakes"].append({"table": "common_mistakes", "dimension": "节奏"})

            elif sub_type in ["悬疑场景", "推理场景", "线索揭露"]:
                strategy["excerpts"].append({"table": "mystery_clues", "query": " ".join(key_elements)})
                strategy["techniques"].append({"table": "information_management", "query": " ".join(key_elements)})
                strategy["common_mistakes"].append({"table": "common_mistakes", "dimension": "情节"})

            else:
                # 通用正文写作策略
                strategy["excerpts"].append({"table": "dialogue_samples", "query": " ".join(key_elements)})
                strategy["excerpts"].append({"table": "description_samples", "query": " ".join(key_elements)})
                strategy["techniques"].append({"table": "skills", "query": " ".join(key_elements)})
                strategy["common_mistakes"].append({"table": "common_mistakes", "dimension": "对话"})
                strategy["common_mistakes"].append({"table": "common_mistakes", "dimension": "描写"})

        return strategy

    def _get_default_result(self, creation_stage: str) -> Dict[str, Any]:
        """返回默认结果"""
        return {
            "detected_scene": creation_stage,
            "detected_sub_type": "",
            "detected_mood": "",
            "key_elements": [],
            "query_strategy": {
                "excerpts": [],
                "techniques": [],
                "structure_tips": [],
                "common_mistakes": [],
            },
        }


# 全局上下文分析器实例
_global_context_analyzer: Optional[ContextAnalyzer] = None


def get_context_analyzer() -> ContextAnalyzer:
    """获取全局上下文分析器实例"""
    global _global_context_analyzer
    if _global_context_analyzer is None:
        _global_context_analyzer = ContextAnalyzer()
    return _global_context_analyzer
