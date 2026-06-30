"""
Stage N: 技法组合模板提取模块
从已有的高潮构建链 + 范文中提取技法组合模式
返回"一组技法如何组合使用"的模板，而非单个技法
"""
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict
from stages.base import BaseStage
from core.ollama_client import ollama_chat, safe_parse_json
from core.utils import generate_id

logger = logging.getLogger(__name__)


class StageN(BaseStage):
    """Stage N: 技法组合模板提取"""

    def __init__(self, book_name: str = "technique_combos", category: str = "fiction"):
        super().__init__("N", book_name, category)

    def run(
        self,
        scene_types: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        执行技法组合模板提取

        Args:
            scene_types: 场景类型列表（如["打斗", "对话", "描写", "高潮", "转折", "揭秘"]）
                        默认提取所有类型

        Returns:
            技法组合模板结果字典
        """
        logger.info(f"=== 阶段N：技法组合模板提取 ===")

        if scene_types is None:
            scene_types = ["打斗", "对话", "描写", "高潮", "转折", "揭秘"]

        # 1. 按场景类型收集相关数据
        scene_data = self._collect_scene_data(scene_types)
        
        if not scene_data:
            logger.warning("⚠️ [阶段N] 未收集到任何场景数据")
            return {"combinations": []}

        # 2. 对每种场景类型调用 LLM 分析技法组合
        all_combinations = []
        for scene_type, data in scene_data.items():
            if not data:
                continue
            
            logger.info(f"   📊 分析场景类型: {scene_type} (数据量: {len(data)})")
            
            prompt = self._build_combo_analysis_prompt(scene_type, data)
            
            try:
                resp = ollama_chat(prompt, 0.3, "N")
                result_data = safe_parse_json(resp)
                if result_data and result_data.get("combinations"):
                    for combo in result_data["combinations"]:
                        combo["scene_type"] = scene_type
                    all_combinations.extend(result_data["combinations"])
            except Exception as exc:
                logger.error(f"❌ [阶段N] 场景 {scene_type} 分析失败: {exc}")
                continue

        result = {
            "combinations": all_combinations,
            "scene_types_analyzed": list(scene_data.keys()),
        }
        
        logger.info(
            f"✅ [阶段N战报] 分析场景类型: {len(scene_data)} | "
            f"技法组合数: {len(all_combinations)}"
        )
        return result

    def insert(self, results: Dict[str, Any]) -> Dict[str, int]:
        """将技法组合模板写入数据库"""
        cursor = self.db.connect().cursor()
        stats = {"technique_combinations": 0}

        for combo in results.get("combinations", []):
            if not combo.get("combo_name"):
                continue

            combo_id = generate_id(
                combo.get("scene_type", ""),
                combo["combo_name"],
                combo.get("benchmark_book", ""),
            )

            cursor.execute(
                "INSERT OR REPLACE INTO technique_combinations VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    combo_id,
                    combo.get("scene_type", ""),
                    combo["combo_name"],
                    json.dumps(combo.get("technique_sequence", []), ensure_ascii=False),
                    json.dumps(combo.get("technique_roles", []), ensure_ascii=False),
                    combo.get("applicable_scenarios", ""),
                    combo.get("variations", ""),
                    combo.get("benchmark_book", ""),
                    combo.get("original_example", ""),
                    datetime.now().isoformat(),
                ),
            )
            stats["technique_combinations"] += 1

        self.db.commit()
        logger.info(f"   ✅ [阶段N] {stats['technique_combinations']} 条技法组合已写入")
        return stats

    def _collect_scene_data(self, scene_types: List[str]) -> Dict[str, List[Dict]]:
        """按场景类型收集相关数据"""
        cursor = self.db.connect().cursor()
        scene_data = defaultdict(list)
        
        for scene_type in scene_types:
            data_items = []
            
            # 收集高潮构建链数据
            if scene_type in ["打斗", "高潮", "转折", "揭秘"]:
                cursor.execute(
                    "SELECT book_name, climax_name, buildup_steps_json, tension_escalation FROM climax_buildup_chains WHERE climax_name LIKE ? LIMIT 10",
                    (f"%{scene_type}%",),
                )
                for row in cursor.fetchall():
                    steps = json.loads(row[2]) if row[2] else []
                    data_items.append({
                        "type": "buildup_chain",
                        "book_name": row[0],
                        "name": row[1],
                        "steps": steps,
                        "escalation": row[3],
                    })
            
            # 收集高潮段落原文
            cursor.execute(
                "SELECT book_name, excerpt_type, original_text, technique_analysis FROM climax_excerpts WHERE excerpt_type LIKE ? LIMIT 10",
                (f"%{scene_type}%",),
            )
            for row in cursor.fetchall():
                data_items.append({
                    "type": "excerpt",
                    "book_name": row[0],
                    "excerpt_type": row[1],
                    "text": row[2][:300] if row[2] else "",
                    "analysis": row[3],
                })
            
            # 收集对话样本
            if scene_type == "对话":
                cursor.execute(
                    "SELECT book_name, scene_type, original_text, subtext, plot_function FROM dialogue_samples LIMIT 10"
                )
                for row in cursor.fetchall():
                    data_items.append({
                        "type": "dialogue",
                        "book_name": row[0],
                        "scene_type": row[1],
                        "text": row[2][:300] if row[2] else "",
                        "subtext": row[3],
                        "function": row[4],
                    })
            
            # 收集描写样本
            if scene_type == "描写":
                cursor.execute(
                    "SELECT book_name, description_type, original_text, technique_analysis, sensory_details FROM description_samples LIMIT 10"
                )
                for row in cursor.fetchall():
                    data_items.append({
                        "type": "description",
                        "book_name": row[0],
                        "desc_type": row[1],
                        "text": row[2][:300] if row[2] else "",
                        "analysis": row[3],
                        "sensory": row[4],
                    })
            
            # 收集动作场景样本
            if scene_type == "打斗":
                cursor.execute(
                    "SELECT book_name, action_type, original_text, technique_analysis, pacing_analysis FROM action_scene_samples LIMIT 10"
                )
                for row in cursor.fetchall():
                    data_items.append({
                        "type": "action",
                        "book_name": row[0],
                        "action_type": row[1],
                        "text": row[2][:300] if row[2] else "",
                        "analysis": row[3],
                        "pacing": row[4],
                    })
            
            # 收集叙事技法
            cursor.execute(
                "SELECT book_name, skill_name, analysis, original_example FROM skills WHERE scene_type LIKE ? LIMIT 10",
                (f"%{scene_type}%",),
            )
            for row in cursor.fetchall():
                data_items.append({
                    "type": "skill",
                    "book_name": row[0],
                    "skill_name": row[1],
                    "analysis": row[2],
                    "example": row[3][:200] if row[3] else "",
                })
            
            if data_items:
                scene_data[scene_type] = data_items
        
        return dict(scene_data)

    def _build_combo_analysis_prompt(
        self,
        scene_type: str,
        data_items: List[Dict],
    ) -> str:
        """构建技法组合分析 Prompt"""
        data_text_parts = []
        
        for item in data_items:
            if item["type"] == "buildup_chain":
                data_text_parts.append(
                    f"[高潮构建链] 《{item['book_name']}》{item['name']}\n"
                    f"铺垫步骤: {item['steps']}\n"
                    f"张力升级: {item['escalation']}"
                )
            elif item["type"] == "excerpt":
                data_text_parts.append(
                    f"[高潮段落] 《{item['book_name']}》{item['excerpt_type']}\n"
                    f"原文: {item['text'][:150]}...\n"
                    f"技法分析: {item['analysis']}"
                )
            elif item["type"] == "dialogue":
                data_text_parts.append(
                    f"[对话样本] 《{item['book_name']}》{item['scene_type']}\n"
                    f"原文: {item['text'][:150]}...\n"
                    f"潜台词: {item['subtext']}\n"
                    f"剧情作用: {item['function']}"
                )
            elif item["type"] == "description":
                data_text_parts.append(
                    f"[描写样本] 《{item['book_name']}》{item['desc_type']}\n"
                    f"原文: {item['text'][:150]}...\n"
                    f"技法分析: {item['analysis']}\n"
                    f"感官细节: {item['sensory']}"
                )
            elif item["type"] == "action":
                data_text_parts.append(
                    f"[动作场景] 《{item['book_name']}》{item['action_type']}\n"
                    f"原文: {item['text'][:150]}...\n"
                    f"技法分析: {item['analysis']}\n"
                    f"节奏控制: {item['pacing']}"
                )
            elif item["type"] == "skill":
                data_text_parts.append(
                    f"[叙事技法] 《{item['book_name']}》{item['skill_name']}\n"
                    f"分析: {item['analysis']}\n"
                    f"示例: {item['example'][:100]}"
                )

        data_text = "\n\n".join(data_text_parts[:20])  # 限制数据量避免过长

        return f"""你是资深的写作技法分析师。请分析以下标杆作品在"{scene_type}"场景中使用的技法组合模式，提炼出可复用的"技法组合模板"。

【场景类型】{scene_type}

【标杆作品数据】
{data_text}

请输出纯 JSON 格式：
{{
  "combinations": [
    {{
      "combo_name": "组合模板名称(如:打脸场景四步法/情感对话潜台词链)",
      "technique_sequence": [
        "技法1: 铺垫轻视(Show，通过旁观者的轻蔑言行)",
        "技法2: 主角沉默(叙事距离拉远，不写内心)",
        "技法3: 实力展示(短句快节奏，动作描写)",
        "技法4: 旁观者震惊(对话+反应描写)"
      ],
      "technique_roles": [
        "技法1作用: 制造反差，积累读者期待",
        "技法2作用: 保持神秘感，增强张力",
        "技法3作用: 释放积累的张力，给读者爽感",
        "技法4作用: 通过旁观者反应放大效果"
      ],
      "applicable_scenarios": "适用场景(如:主角被轻视后展现实力的场景，适合网文/轻小说)",
      "variations": "变体建议(如:可以将'旁观者震惊'改为'对手恐惧'，适用于不同情境)",
      "benchmark_book": "主要参考书名",
      "original_example": "原文示例片段(100字内)"
    }}
  ]
}}
(⚠️核心要求：
1. 技法组合必须有明确的顺序，不能是并列的！
2. 每个技法在组合中的作用必须说明！
3. 适用场景必须具体，不能太宽泛！
4. 变体建议必须可操作！
5. 原文示例必须来自提供的数据！
6. 禁止使用反引号，必须输出合法JSON)"""
