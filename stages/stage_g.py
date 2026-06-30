"""
Stage G: 人物深度特征提取
使用 qwen14b:latest 模型，提取人物语言风格、行为标志、关系动态演变
"""

import json
import logging
from typing import List, Dict, Any
from tqdm import tqdm
from stages.base import BaseStage
from core.ollama_client import ollama_chat, safe_parse_json
from core.utils import generate_id
from core.graph import get_graph_manager

logger = logging.getLogger(__name__)


class StageG(BaseStage):
    """Stage G: 人物深度特征提取"""

    def __init__(self, book_name: str, category: str):
        super().__init__("G", book_name, category)

    def run(self, chapters: List[Dict], **kwargs) -> Dict[str, List[Dict]]:
        """
        执行 Stage G

        Args:
            chapters: 章节列表

        Returns:
            包含 character_speech_style, character_behavior_marks, character_relationship_dynamics 的字典
        """
        logger.info(f"=== 阶段七：人物深度特征提取 ({self.book_name}) ===")

        result = {
            "character_speech_style": [],
            "character_behavior_marks": [],
            "character_relationship_dynamics": [],
        }

        # 收集所有出现过的人物名及其出现频率
        character_frequency = {}
        for chap in chapters:
            char_state = chap.get("character_state", {})
            for char_name in char_state.keys():
                if char_name not in ["_raw", "旁白"]:
                    character_frequency[char_name] = (
                        character_frequency.get(char_name, 0) + 1
                    )

        logger.info(
            f"📊 [阶段G] 发现 {len(character_frequency)} 个人物，将提取深度特征"
        )

        # 按出现频率排序，优先提取高频人物
        sorted_characters = sorted(
            character_frequency.items(), key=lambda x: x[1], reverse=True
        )
        target_characters = [char_name for char_name, freq in sorted_characters[:20]]

        for char_name in tqdm(target_characters, desc="提取人物深度特征"):
            # 收集该人物出现的章节
            char_chapters = []
            for chap in chapters:
                char_state = chap.get("character_state", {})
                if char_name in char_state:
                    char_chapters.append(chap)

            if len(char_chapters) < 3:
                continue  # 出现次数太少，跳过

            # 取前 5 章作为样本
            sample_chapters = char_chapters[:5]
            sample_text = "\n\n".join(
                [
                    f"【{c.get('id', '')}】\n{c.get('text', '')[:1000]}"
                    for c in sample_chapters
                ]
            )

            prompt = f"""你是顶级的人物塑造大师与文学分析师。请根据以下章节片段，深度分析人物【{char_name}】的语言风格、行为标志和关系动态。

【书名】{self.book_name} 【分类】{self.category}
【人物】{char_name}
【章节片段】
{sample_text}

请输出纯 JSON 格式：
{{
  "speech_style": {{
    "catchphrases": ["口头禅1", "口头禅2", "口头禅3"],
    "vocabulary_preference": "词汇偏好(如:偏好文言词汇/网络用语/专业术语/粗俗俚语，30字内)",
    "sentence_pattern": "句式偏好(如:偏爱短句/长句/反问句/感叹句/省略句，30字内)",
    "tone_contexts": {{
      "对上级": "语气特点(如:恭敬谦卑/不卑不亢/阳奉阴违)",
      "对平辈": "语气特点(如:随意调侃/真诚友善/冷漠疏离)",
      "对下级": "语气特点(如:威严冷峻/温和亲切/傲慢轻视)",
      "对敌人": "语气特点(如:冷酷威胁/嘲讽挑衅/沉默寡言)"
    }},
    "dialogue_samples": ["从原文摘录的对话样本1（50字内）", "对话样本2"]
  }},
  "behavior_marks": {{
    "habitual_actions": ["习惯性动作1(如:思考时摸下巴)", "习惯性动作2"],
    "micro_expressions": ["微表情1(如:眼神闪烁)", "微表情2"],
    "defense_mechanisms": "心理防御机制(如:用幽默掩饰不安/用愤怒掩盖恐惧/用冷漠保护脆弱，50字内)",
    "behavior_samples": ["从原文摘录的行为描写样本1（50字内）", "行为样本2"]
  }},
  "relationship_dynamics": [
    {{
      "other_character": "另一个角色名",
      "relationship_timeline": [
        {{
          "chapter_range": "章节范围(如:1-50)",
          "relationship_state": "关系状态(如:敌对/合作/暧昧/师徒/恋人)",
          "key_events": ["关键事件1", "关键事件2"],
          "emotional_tone": "情感基调(如:紧张戒备/信任默契/甜蜜温馨)"
        }}
      ]
    }}
  ]
}}
(⚠️核心要求：
1. 必须从原文中摘录真实的对话样本和行为描写样本！
2. 必须分析人物在不同关系中的语气差异！
3. 必须识别人物的心理防御机制！
4. 如果某个人物信息不足，对应字段留空
5. 禁止使用反引号，必须输出合法JSON)"""

            try:
                resp = ollama_chat(prompt, 0.2, "G")
                data = safe_parse_json(resp)
                if not data:
                    continue

                # 解析语言风格
                speech = data.get("speech_style", {})
                if speech:
                    result["character_speech_style"].append(
                        {
                            "book_name": self.book_name,
                            "character_name": char_name,
                            "catchphrases": speech.get("catchphrases", []),
                            "vocabulary_preference": speech.get(
                                "vocabulary_preference", ""
                            ),
                            "sentence_pattern": speech.get("sentence_pattern", ""),
                            "tone_contexts": speech.get("tone_contexts", {}),
                            "dialogue_samples": speech.get("dialogue_samples", []),
                        }
                    )

                # 解析行为标志
                behavior = data.get("behavior_marks", {})
                if behavior:
                    result["character_behavior_marks"].append(
                        {
                            "book_name": self.book_name,
                            "character_name": char_name,
                            "habitual_actions": behavior.get("habitual_actions", []),
                            "micro_expressions": behavior.get("micro_expressions", []),
                            "defense_mechanisms": behavior.get(
                                "defense_mechanisms", ""
                            ),
                            "behavior_samples": behavior.get("behavior_samples", []),
                        }
                    )

                # 解析关系动态
                relationships = data.get("relationship_dynamics", [])
                for rel in relationships:
                    if isinstance(rel, dict) and rel.get("other_character"):
                        result["character_relationship_dynamics"].append(
                            {
                                "book_name": self.book_name,
                                "character_a": char_name,
                                "character_b": rel["other_character"],
                                "timeline": rel.get("relationship_timeline", []),
                            }
                        )

            except Exception as e:
                logger.warning(f"⚠️ [阶段G] 解析人物 {char_name} 失败: {e}")

        logger.info(
            f"✅ [阶段G战报] 语言风格: {len(result['character_speech_style'])} 人, "
            f"行为标志: {len(result['character_behavior_marks'])} 人, "
            f"关系动态: {len(result['character_relationship_dynamics'])} 对"
        )

        # 更新知识图谱：人物关系边
        self._update_graph(result)

        return result

    def _update_graph(self, result: Dict[str, List[Dict]]):
        """将人物关系同步到知识图谱"""
        try:
            graph = get_graph_manager()

            for rd in result.get("character_relationship_dynamics", []):
                char_a = rd.get("character_a", "")
                char_b = rd.get("character_b", "")
                if not char_a or not char_b:
                    continue

                node_a = f"char:{self.book_name}:{char_a}"
                node_b = f"char:{self.book_name}:{char_b}"

                # 从 timeline 中提取最新关系状态
                timeline = rd.get("timeline", [])
                latest_state = ""
                if timeline and isinstance(timeline, list):
                    last_entry = timeline[-1] if timeline else {}
                    latest_state = last_entry.get("relationship_state", "")

                graph.add_edge(
                    node_a,
                    node_b,
                    relation_type=latest_state or "unknown",
                    book_name=self.book_name,
                )

            graph.save()
            logger.info(
                f"📊 [阶段G] 知识图谱已更新 {len(result.get('character_relationship_dynamics', []))} 条人物关系边"
            )
        except Exception as e:
            logger.warning(f"⚠️ [阶段G] 知识图谱更新失败: {e}")

    def insert(self, results: Dict[str, List[Dict]]) -> Dict[str, int]:
        """将 Stage G 结果写入数据库"""
        cursor = self.db.connect().cursor()
        stats = {"speech_style": 0, "behavior_marks": 0, "relationship_dynamics": 0}

        # 语言风格入库
        for ss in results.get("character_speech_style", []):
            ss_id = generate_id(ss["book_name"], ss["character_name"], "speech")
            cursor.execute(
                "INSERT OR REPLACE INTO character_speech_style VALUES (?,?,?,?,?,?,?,?)",
                (
                    ss_id,
                    ss["book_name"],
                    ss["character_name"],
                    "|".join(ss.get("catchphrases", [])),
                    ss.get("vocabulary_preference", ""),
                    ss.get("sentence_pattern", ""),
                    json.dumps(ss.get("tone_contexts", {}), ensure_ascii=False),
                    json.dumps(ss.get("dialogue_samples", []), ensure_ascii=False),
                ),
            )
            stats["speech_style"] += 1

        # ChromaDB: 语言风格
        ss_ids, ss_docs, ss_metas = [], [], []
        for ss in results.get("character_speech_style", []):
            ssid = generate_id(ss["book_name"], ss["character_name"], "speech")
            ss_ids.append(ssid)
            ss_docs.append(
                f"人物:{ss['character_name']}\n"
                f"口头禅:{'|'.join(ss.get('catchphrases', []))}\n"
                f"词汇偏好:{ss.get('vocabulary_preference', '')}\n"
                f"句式偏好:{ss.get('sentence_pattern', '')}\n"
                f"对话样本:{'|'.join(ss.get('dialogue_samples', []))}"
            )
            ss_metas.append(
                {
                    "book_name": ss["book_name"],
                    "character_name": ss["character_name"],
                }
            )
        if ss_ids:
            self.chroma.upsert_batch(
                "character_speech_style_kb", ss_ids, ss_docs, ss_metas
            )

        # 行为标志入库
        for bm in results.get("character_behavior_marks", []):
            bm_id = generate_id(bm["book_name"], bm["character_name"], "behavior")
            cursor.execute(
                "INSERT OR REPLACE INTO character_behavior_marks VALUES (?,?,?,?,?,?,?)",
                (
                    bm_id,
                    bm["book_name"],
                    bm["character_name"],
                    "|".join(bm.get("habitual_actions", [])),
                    "|".join(bm.get("micro_expressions", [])),
                    bm.get("defense_mechanisms", ""),
                    json.dumps(bm.get("behavior_samples", []), ensure_ascii=False),
                ),
            )
            stats["behavior_marks"] += 1

        # 关系动态入库
        for rd in results.get("character_relationship_dynamics", []):
            rd_id = generate_id(rd["book_name"], rd["character_a"], rd["character_b"])
            cursor.execute(
                "INSERT OR REPLACE INTO character_relationship_dynamics VALUES (?,?,?,?,?)",
                (
                    rd_id,
                    rd["book_name"],
                    rd["character_a"],
                    rd["character_b"],
                    json.dumps(rd.get("timeline", []), ensure_ascii=False),
                ),
            )
            stats["relationship_dynamics"] += 1

        self.db.commit()
        logger.info(
            f"   ✅ [阶段G战报] 语言风格: {stats['speech_style']} | "
            f"行为标志: {stats['behavior_marks']} | 关系动态: {stats['relationship_dynamics']}"
        )
        return stats
