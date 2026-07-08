"""
Stage D: 世界观与人物深度自动提取（重做版）
使用 qwen14b:latest 模型，从正文自动提取世界观（7维度）和人物（11维度）
不再依赖外挂设定集，采用智能采样策略提高效率
"""

import json
import math
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from stages.base import BaseStage
from core.ollama_client import ollama_chat, safe_parse_json
from core.utils import generate_id
from core.stage_result import StageResult
from core.chroma_utils import bulk_upsert_to_chroma
from config.settings import STAGE_D_WORKERS

logger = logging.getLogger(__name__)


class StageD(BaseStage):
    """Stage D: 世界观与人物深度自动提取"""

    def __init__(self, book_name: str, category: str, author: str = "未知作者"):
        super().__init__("D", book_name, category)
        self.author = author

    def _select_sample_chapters(self, chapters: List[Dict]) -> List[Dict]:
        """
        采样策略：首章 + 尾章 + 均匀间隔
        - 首尾保证世界观建立和收束不丢失
        - 均匀间隔保证全书覆盖
        """
        if len(chapters) <= 10:
            return chapters

        sampled_indices = set()
        sampled_indices.add(0)                    # 首章
        sampled_indices.add(len(chapters) - 1)     # 尾章

        # 均匀间隔：目标采样数用开方公式
        import math
        target = max(10, min(len(chapters), int(10 + 5 * math.sqrt(len(chapters) / 100))))
        step = max(2, (len(chapters) - 2) // (target - 2))
        for i in range(step, len(chapters) - 1, step):
            sampled_indices.add(i)

        # 按顺序返回
        sampled_indices = sorted(list(sampled_indices))
        sampled_chapters = [chapters[i] for i in sampled_indices]

        logger.info(
            f"📊 [阶段D] 智能采样：从 {len(chapters)} 章中选取 {len(sampled_chapters)} 章"
        )
        return sampled_chapters

    def _extract_world_group(self, text: str, chap_id: str, stage_result=None) -> Dict[str, List[Dict]]:
        """提取世界观组：world_settings + world_timeline + faction_networks"""
        result = {"world_settings": [], "world_timeline": [], "faction_networks": [], "golden_finger": None}

        prompt = f"""你是顶级的文学世界观架构师。请根据本书的实际题材，从以下章节文本中提取【世界观设定（7维度+金手指体系）】、【历史编年史】和【势力关系网络】。

【书名】{self.book_name} 【作者】{self.author} 【分类】{self.category}
【章节】{chap_id}
【正文】
{text}

请输出纯 JSON 格式：
{{
  "world_settings": [
    {{
      "module": "设定模块(自适应题材，如力量体系/社会结构/地理空间/政治格局/经济体系/文化符号/规则体系)",
      "entity": "具体实体名",
      "content": "详细规则、空间分布、核心限制/代价/底层冲突(100-300字)",
      "tags": ["标签1", "标签2"],
      "daily_life": "日常生活体系(50字内)",
      "taboos": "禁忌与边界(50字内)",
      "conflict_roots": "冲突根源图谱(50字内)",
      "geography": "地理空间拓扑(50字内)",
      "economy": "经济与资源体系(50字内)",
      "culture": "语言与文化符号(50字内)",
      "causal_chain": "设定间的因果链(50字内)",
      "rules_exceptions": "规则例外与代价(50字内)"
    }}
  ],
  "golden_finger": {{
    "name": "金手指名称(如主角有特殊能力/系统/传承，提取其名称；无则留空)",
    "type": "类型(系统/传承/血脉/异能/知识差/经营/空间/契约/无)",
    "abilities": ["能力1", "能力2"],
    "upgrade_path": "升级路径简述(如何变强/解锁新能力，50字内；无则留空)",
    "limitations": ["限制1(如冷却时间/次数限制/代价)", "限制2"],
    "cost_layers": ["身体代价/精神代价/社交代价/外部代价(无则留空)"],
    "interaction_with_plot": "与剧情的交互方式(金手指如何推动/制约剧情发展，50字内；无则留空)"
  }},
  "world_timeline": [
    {{
      "era_or_year": "纪元或年份",
      "event_name": "大事件名称",
      "event_description": "事件简述(50字内)",
      "impact": "对当前世界/主角的影响(50字内)"
    }}
  ],
  "faction_networks": [
    {{
      "faction_a": "势力A名称",
      "faction_b": "势力B名称",
      "relation_type": "关系类型(同盟/敌对/从属/竞争/中立)",
      "relation_detail": "关系详情(50字内)",
      "stability": "稳定性(稳定/脆弱/动态变化)",
      "key_events": "关键事件(50字内)"
    }}
  ]
}}
(⚠️核心要求：必须根据小说实际题材自适应提取！必须提取规则例外与代价！必须提取势力关系网络！禁止反引号)"""

        try:
            resp = ollama_chat(prompt, 0.1, "D")
            data = safe_parse_json(resp)
            if data:
                for ws in data.get("world_settings", []):
                    if isinstance(ws, dict) and ws.get("content"):
                        result["world_settings"].append(
                            {
                                "book_name": self.book_name,
                                "author": self.author,
                                "category": self.category,
                                "module": ws.get("module", "未知"),
                                "entity": ws.get("entity", "未知"),
                                "content": ws.get("content"),
                                "tags": ws.get("tags", []),
                                "daily_life": ws.get("daily_life", ""),
                                "taboos": ws.get("taboos", ""),
                                "conflict_roots": ws.get("conflict_roots", ""),
                                "geography": ws.get("geography", ""),
                                "economy": ws.get("economy", ""),
                                "culture": ws.get("culture", ""),
                                "causal_chain": ws.get("causal_chain", ""),
                                "rules_exceptions": ws.get("rules_exceptions", ""),
                            }
                        )

                for wt in data.get("world_timeline", []):
                    if isinstance(wt, dict) and wt.get("event_name"):
                        result["world_timeline"].append(
                            {
                                "book_name": self.book_name,
                                "era_or_year": wt.get("era_or_year", "未知纪元"),
                                "event_name": wt.get("event_name"),
                                "event_description": wt.get("event_description", ""),
                                "impact": wt.get("impact", ""),
                            }
                        )

                for fn in data.get("faction_networks", []):
                    if (
                        isinstance(fn, dict)
                        and fn.get("faction_a")
                        and fn.get("faction_b")
                    ):
                        result["faction_networks"].append(
                            {
                                "book_name": self.book_name,
                                "faction_a": fn.get("faction_a"),
                                "faction_b": fn.get("faction_b"),
                                "relation_type": fn.get("relation_type", "未知"),
                                "relation_detail": fn.get("relation_detail", ""),
                                "stability": fn.get("stability", ""),
                                "key_events": fn.get("key_events", ""),
                            }
                        )

                # 金手指提取（只保留第一个有效结果）
                gf = data.get("golden_finger", {})
                if isinstance(gf, dict) and gf.get("name") and gf.get("type") != "无":
                    result["golden_finger"] = {
                        "book_name": self.book_name,
                        "name": gf.get("name", ""),
                        "type": gf.get("type", ""),
                        "abilities": gf.get("abilities", []),
                        "upgrade_path": gf.get("upgrade_path", ""),
                        "limitations": gf.get("limitations", []),
                        "cost_layers": gf.get("cost_layers", []),
                        "interaction_with_plot": gf.get("interaction_with_plot", ""),
                        "source_chapter": chap_id,
                    }
        except Exception as e:
            logger.warning(f"⚠️ [阶段D-世界观] 解析章节 {chap_id} 失败: {e}")
            if stage_result:
                stage_result.add_failure(chap_id, str(e), "D-world")
        
        return result

    def _extract_character_group(self, text: str, chap_id: str, stage_result=None) -> Dict[str, List[Dict]]:
        """提取人物核心档案（14字段，从单chunk可准确判断）"""
        result = {"character_profiles": []}

        prompt = f"""你是顶级的人物塑造大师。请从以下章节文本中提取【人物核心档案】。

【书名】{self.book_name} 【作者】{self.author} 【分类】{self.category}
【章节】{chap_id}
【正文】
{text}

请输出纯 JSON 格式：
{{
  "character_profiles": [
    {{
      "name": "人物名",
      "role_type": "角色定位(主角/核心配角/对立面/导师/群像代表等)",
      "appearance": "视觉记忆点(发色/疤痕/标志性穿搭/气质，50字内)",
      "quirks": "标志性口癖/微表情/下意识动作",
      "identity": "身份/职业/阵营/社会阶层",
      "motivation": "核心动机/终极目标/核心欲望",
      "internal_conflict": "内心冲突/人物弧光",
      "personality": "性格底色/优缺点/行事底线",
      "relation_to_mc": "与主角/核心视角的初始关系",
      "abilities": "能力体系(技能/天赋/战斗风格，50字内)",
      "speech_samples": "语言风格样本(口头禅、用词习惯的原文摘录，100字内)",
      "behavior_samples": "行为标志样本(习惯性动作的原文描写，100字内)",
      "climax_or_fate": "高光时刻预设/宿命结局",
      "background": "前史/背景故事/原生家庭影响"
    }}
  ]
}}
(⚠️核心要求：必须从原文摘录 speech_samples 和 behavior_samples！禁止反引号)"""

        try:
            resp = ollama_chat(prompt, 0.1, "D")
            data = safe_parse_json(resp)
            if data:
                for cp in data.get("character_profiles", []):
                    if isinstance(cp, dict) and cp.get("name"):
                        result["character_profiles"].append(
                            {
                                "book_name": self.book_name,
                                "author": self.author,
                                "category": self.category,
                                "name": cp.get("name"),
                                "role_type": cp.get("role_type", "未知"),
                                "appearance": cp.get("appearance", ""),
                                "quirks": cp.get("quirks", ""),
                                "identity": cp.get("identity", ""),
                                "motivation": cp.get("motivation", ""),
                                "internal_conflict": cp.get("internal_conflict", ""),
                                "personality": cp.get("personality", ""),
                                "relation_to_mc": cp.get("relation_to_mc", "未知"),
                                "abilities": json.dumps(cp.get("abilities", ""), ensure_ascii=False) if isinstance(cp.get("abilities"), list) else cp.get("abilities", ""),
                                "speech_samples": cp.get("speech_samples", ""),
                                "behavior_samples": cp.get("behavior_samples", ""),
                                "climax_or_fate": cp.get("climax_or_fate", ""),
                                "background": cp.get("background", ""),
                            }
                        )
        except Exception as e:
            logger.warning(f"⚠️ [阶段D-人物] 解析章节 {chap_id} 失败: {e}")
            if stage_result:
                stage_result.add_failure(chap_id, str(e), "D-character")
        
        return result

    def _aggregate_characters(self, profiles: List[Dict]) -> Dict[str, tuple]:
        """按人物名聚合核心字段，返回 {name: (core_text, chapter_refs)}"""
        agg = {}
        for p in profiles:
            name = p.get("name", "")
            if not name:
                continue
            if name not in agg:
                agg[name] = {"texts": [], "chapters": []}
            agg[name]["chapters"].append(p)
            agg[name]["texts"].append(
                f"定位:{p.get('role_type','')}|外貌:{p.get('appearance','')}|"
                f"口癖:{p.get('quirks','')}|身份:{p.get('identity','')}|"
                f"动机:{p.get('motivation','')}|内心冲突:{p.get('internal_conflict','')}|"
                f"性格:{p.get('personality','')}|与主角关系:{p.get('relation_to_mc','')}|"
                f"能力:{p.get('abilities','')}|语言:{p.get('speech_samples','')}|"
                f"行为:{p.get('behavior_samples','')}|高光:{p.get('climax_or_fate','')}|"
                f"前史:{p.get('background','')}"
            )
        return {name: (" | ".join(d["texts"]), d["chapters"]) for name, d in agg.items()}

    def _extend_character(self, name: str, core_text: str, chapters: List[Dict]):
        """基于聚合核心档案提取扩展19字段，回填到所有该人物的记录中"""
        if not core_text.strip():
            return
        prompt = f"""你是顶级的人物塑造大师。请根据以下《{self.book_name}》({self.category})中人物【{name}】的全书核心档案，深度分析扩展维度。

【核心档案】
{core_text[:5000]}

请输出纯 JSON 格式：
{{
  "fatal_flaw": "性格缺陷/悲剧根源(50字内，无则留空)",
  "symbolism": "象征意义/社会隐喻(限30字，无则留空)",
  "relations_to_others": "与其他重要配角的社会与情感羁绊(100字内，无则留空)",
  "desire_vs_need": "欲望vs需求(表面想要的vs真正需要的，50字内，无则留空)",
  "secrets": "人物的秘密(隐藏的过去、不可告人的目的，50字内，无则留空)",
  "fears": "人物的恐惧(最害怕什么、心理阴影，50字内，无则留空)",
  "social_masks": "社交面具(在不同关系中的不同表现，50字内，无则留空)",
  "growth_cost": "成长代价(获得什么、失去什么，50字内，无则留空)",
  "relationship_evolution": "人物关系动态演变(100字内，无则留空)",
  "arc_trajectory": "人物弧光轨迹(起点→转折→终点，50字内，无则留空)",
  "internal_dilemma": "内心两难困境(两个互斥的选择及其代价，50字内，无则留空)",
  "decision_pattern": "决策模式(冲动型/理性分析型/从众型/直觉型，50字内，无则留空)",
  "cognitive_bias": "认知偏差(对世界/他人的错误认知、偏见，50字内，无则留空)",
  "transformation_trigger": "转变触发器(什么事件触发了人物转变，50字内，无则留空)",
  "contrast_design": "对比设计(与同类型角色的差异设计，50字内，无则留空)",
  "archetype_label": "角色原型标签(30字内，无则留空)",
  "writing_anti_patterns": "忌讳写法/毒点(50字内，无则留空)"
}}
(⚠️所有字段无则留空，禁止编造信息。禁止反引号)"""
        resp = ollama_chat(prompt, 0.1, "D")
        data = safe_parse_json(resp)
        if not data:
            return
        for chap in chapters:
            chap["fatal_flaw"] = data.get("fatal_flaw", "")
            chap["symbolism"] = data.get("symbolism", "")
            chap["relations_to_others"] = data.get("relations_to_others", "")
            chap["desire_vs_need"] = data.get("desire_vs_need", "")
            chap["secrets"] = data.get("secrets", "")
            chap["fears"] = data.get("fears", "")
            chap["social_masks"] = data.get("social_masks", "")
            chap["growth_cost"] = data.get("growth_cost", "")
            chap["relationship_evolution"] = data.get("relationship_evolution", "")
            chap["arc_trajectory"] = data.get("arc_trajectory", "")
            chap["internal_dilemma"] = data.get("internal_dilemma", "")
            chap["decision_pattern"] = data.get("decision_pattern", "")
            chap["cognitive_bias"] = data.get("cognitive_bias", "")
            chap["transformation_trigger"] = data.get("transformation_trigger", "")
            chap["contrast_design"] = data.get("contrast_design", "")
            chap["archetype_label"] = data.get("archetype_label", "")
            chap["writing_anti_patterns"] = data.get("writing_anti_patterns", "")

    def _process_single_chapter(self, chap: Dict) -> Dict[str, List[Dict]]:
        """
        处理单章：提取世界观组 + 人物组（线程安全）
        """
        text = chap["text"]
        chap_id = chap.get("id", "未知章节")

        chapter_result = {
            "world_settings": [],
            "character_profiles": [],
            "world_timeline": [],
            "faction_networks": [],
            "golden_finger": None,
        }

        # 批次1：世界观 + 编年史 + 势力网络
        world_data = self._extract_world_group(text, chap_id)
        chapter_result["world_settings"] = world_data["world_settings"]
        chapter_result["world_timeline"] = world_data["world_timeline"]
        chapter_result["faction_networks"] = world_data["faction_networks"]
        chapter_result["golden_finger"] = world_data.get("golden_finger")

        # 批次2：人物深度档案
        char_data = self._extract_character_group(text, chap_id)
        chapter_result["character_profiles"] = char_data["character_profiles"]

        return chapter_result

    def run(self, chapters: List[Dict], **kwargs) -> Dict[str, List[Dict]]:
        """
        执行 Stage D（并行处理采样章节，利用 STAGE_D_WORKERS 并发，支持断点续跑）

        Args:
            chapters: 章节列表

        Returns:
            包含 world_settings, character_profiles, world_timeline, faction_networks 的字典
        """
        stage_result = StageResult()
        logger.info(f"=== 阶段四：世界观与人物深度自动提取 ({self.book_name}) ===")

        DATA_KEYS = ["world_settings", "character_profiles", "world_timeline", "faction_networks"]
        result = {key: [] for key in DATA_KEYS}
        result["golden_finger"] = None  # 单独处理，不是 list

        # 智能采样
        sampled_chapters = self._select_sample_chapters(chapters)

        # 断点恢复
        cache = self.load_cache()
        completed_items = cache.get("data", []) if cache else []
        completed_ids = {item.get("_chapter_id", "") for item in completed_items if item.get("_chapter_id")}

        for item in completed_items:
            for key in DATA_KEYS:
                result[key].extend(item.get(key, []))
            # golden_finger 单独处理（dict 类型，不用 extend）
            if not result["golden_finger"] and item.get("golden_finger"):
                result["golden_finger"] = item["golden_finger"]

        if completed_ids:
            logger.info(f"✅ [阶段D] 恢复断点：已完成 {len(completed_ids)}/{len(sampled_chapters)} 章")

        pending = [c for c in sampled_chapters if c.get("id") not in completed_ids]
        if not pending:
            logger.info(f"[阶段D] 所有采样章节已处理完毕")
        else:
            workers = min(STAGE_D_WORKERS, len(pending))
            logger.info(f"[阶段D] 使用 {workers} 个并发 worker 处理剩余 {len(pending)} 个采样章节")
            print(f"  [Stage D] 首次调用需加载 qwen14b 模型到显存 (约30-60秒)，请稍候...", flush=True)

            processed_count = 0
            with ThreadPoolExecutor(max_workers=max(workers, 1)) as executor:
                futures = {
                    executor.submit(self._process_single_chapter, chap): chap.get("id", "unknown")
                    for chap in pending
                }
                try:
                    for future in tqdm(as_completed(futures), total=len(futures), desc="提取世界观与人物"):
                        chap_id = futures[future]
                        try:
                            chapter_result = future.result()
                            chapter_result["_chapter_id"] = chap_id
                            completed_items.append(chapter_result)
                            for key in DATA_KEYS:
                                result[key].extend(chapter_result.get(key, []))
                            # golden_finger 单独处理（只保留第一个有效结果）
                            if not result["golden_finger"] and chapter_result.get("golden_finger"):
                                result["golden_finger"] = chapter_result["golden_finger"]
                            processed_count += 1
                            if processed_count % 5 == 0:
                                self.save_cache({"data": completed_items})
                        except Exception as e:
                            logger.warning(f"⚠️ [阶段D] 章节 {chap_id} 处理失败: {e}")
                            stage_result.add_failure(chap_id, str(e), "D")
                except KeyboardInterrupt:
                    logger.info(f"[阶段D] 用户中断，保存进度 ({len(completed_items)} 章已完成)...")
                    self.save_cache({"data": completed_items})
                    raise

            self.save_cache({"data": completed_items})

        # Phase 2: 聚合核心档案 → 提取扩展19字段
        char_profiles = result.get("character_profiles", [])
        if char_profiles:
            aggregated = self._aggregate_characters(char_profiles)
            if aggregated:
                logger.info(f"[阶段D] Phase2: 提取 {len(aggregated)} 个人物的扩展字段...")
                for name, (core_text, chapters) in aggregated.items():
                    try:
                        self._extend_character(name, core_text, chapters)
                    except Exception as e:
                        logger.warning(f"⚠️ [阶段D] 人物 {name} 扩展提取失败: {e}")

        logger.info(
            f"✅ [阶段D战报] 世界观: {len(result['world_settings'])} 条, "
            f"人物档案: {len(result['character_profiles'])} 条, "
            f"编年史: {len(result['world_timeline'])} 条, "
            f"势力网络: {len(result['faction_networks'])} 条"
        )
        stage_result.data = result
        stage_result.stats = {
            "world_settings": len(result["world_settings"]),
            "character_profiles": len(result["character_profiles"]),
            "world_timeline": len(result["world_timeline"]),
            "faction_networks": len(result["faction_networks"]),
        }
        summary = stage_result.get_summary()
        if summary["failure_count"] > 0:
            logger.warning(f"⚠️ [阶段D] 有 {summary['failure_count']} 个章节处理失败")
        return result

    def insert(self, results: Dict[str, List[Dict]]) -> Dict[str, int]:
        """将 Stage D 结果写入数据库"""
        cursor = self.db.connect().cursor()
        stats = {
            "world_settings": 0,
            "golden_finger": 0,
            "character_profiles": 0,
            "world_timeline": 0,
            "faction_networks": 0,
        }

        # 世界观入库（16个字段）
        for ws in results.get("world_settings", []):
            ws_id = generate_id(ws["book_name"], ws["module"], ws["entity"])
            cursor.execute(
                "INSERT OR REPLACE INTO world_settings VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    ws_id,
                    ws["book_name"],
                    ws["author"],
                    ws["category"],
                    ws["module"],
                    ws["entity"],
                    ws["content"],
                    "|".join(ws.get("tags", [])),
                    ws.get("daily_life", ""),
                    ws.get("taboos", ""),
                    ws.get("conflict_roots", ""),
                    ws.get("geography", ""),
                    ws.get("economy", ""),
                    ws.get("culture", ""),
                    ws.get("causal_chain", ""),
                    ws.get("rules_exceptions", ""),
                ),
            )
            stats["world_settings"] += 1

        # ChromaDB: 世界观（构建独立的 ChromaDB 数据，不污染原始数据）
        chroma_world_items = []
        for ws in results.get("world_settings", []):
            chroma_item = {**ws}  # 浅拷贝，避免修改原始字典
            chroma_item["_chroma_text"] = (
                f"模块:{ws['module']}\n实体:{ws['entity']}\n设定:{ws['content']}\n"
                f"日常生活:{ws.get('daily_life', '')}\n禁忌:{ws.get('taboos', '')}\n"
                f"冲突根源:{ws.get('conflict_roots', '')}\n地理:{ws.get('geography', '')}\n"
                f"经济:{ws.get('economy', '')}\n文化:{ws.get('culture', '')}\n"
                f"因果链:{ws.get('causal_chain', '')}"
            )
            chroma_world_items.append(chroma_item)
        bulk_upsert_to_chroma(
            "world_settings_kb",
            chroma_world_items,
            id_fields=["book_name", "module", "entity"],
            text_field="_chroma_text",
            metadata_fields=[
                "book_name",
                "author",
                "category",
                "module",
                "entity",
                "tags",
            ],
        )

        # 金手指入库
        gf = results.get("golden_finger")
        if gf and isinstance(gf, dict) and gf.get("name"):
            gf_id = generate_id(gf["book_name"], "golden_finger", gf["name"])
            cursor.execute(
                "INSERT OR REPLACE INTO golden_finger VALUES (?,?,?,?,?,?,?,?,?,?)",
                (gf_id, gf["book_name"], gf["name"], gf.get("type", ""),
                 json.dumps(gf.get("abilities", []), ensure_ascii=False),
                 gf.get("upgrade_path", ""),
                 json.dumps(gf.get("limitations", []), ensure_ascii=False),
                 json.dumps(gf.get("cost_layers", []), ensure_ascii=False),
                 gf.get("interaction_with_plot", ""),
                 gf.get("source_chapter", "")),
            )
            stats["golden_finger"] = 1

        # 人物档案入库（33个字段）
        for cp in results.get("character_profiles", []):
            cp_id = generate_id(cp["book_name"], cp["name"], "profile")
            cursor.execute(
                "INSERT OR REPLACE INTO character_profiles VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    cp_id,
                    cp["book_name"],
                    cp["author"],
                    cp["category"],
                    cp["name"],
                    cp.get("role_type", "未知"),
                    cp.get("appearance", ""),
                    cp.get("quirks", ""),
                    cp.get("identity", ""),
                    cp.get("motivation", ""),
                    cp.get("internal_conflict", ""),
                    cp.get("fatal_flaw", ""),
                    cp.get("symbolism", ""),
                    cp.get("personality", ""),
                    cp.get("relation_to_mc", "未知"),
                    cp.get("relations_to_others", ""),
                    cp.get("climax_or_fate", ""),
                    cp.get("background", ""),
                    cp.get("desire_vs_need", ""),
                    cp.get("secrets", ""),
                    cp.get("fears", ""),
                    cp.get("social_masks", ""),
                    cp.get("growth_cost", ""),
                    cp.get("speech_samples", ""),
                    cp.get("behavior_samples", ""),
                    cp.get("relationship_evolution", ""),
                    cp.get("abilities", ""),
                    cp.get("arc_trajectory", ""),
                    cp.get("internal_dilemma", ""),
                    cp.get("decision_pattern", ""),
                    cp.get("cognitive_bias", ""),
                    cp.get("transformation_trigger", ""),
                    cp.get("contrast_design", ""),
                    cp.get("archetype_label", ""),
                    cp.get("writing_anti_patterns", ""),
                ),
            )
            stats["character_profiles"] += 1

        # ChromaDB: 人物档案（构建独立的 ChromaDB 数据，不污染原始数据）
        chroma_char_items = []
        for cp in results.get("character_profiles", []):
            chroma_item = {**cp}  # 浅拷贝
            chroma_item["_chroma_text"] = (
                f"定位:{cp.get('role_type', '未知')}\n"
                f"外貌:{cp.get('appearance', '无')}\n"
                f"微表情/口癖:{cp.get('quirks', '无')}\n"
                f"身份:{cp.get('identity', '')}\n"
                f"动机:{cp.get('motivation', '')}\n"
                f"内心冲突/弧光:{cp.get('internal_conflict', '无')}\n"
                f"性格缺陷/悲剧根源:{cp.get('fatal_flaw', '无')}\n"
                f"象征意义/隐喻:{cp.get('symbolism', '无')}\n"
                f"性格:{cp.get('personality', '')}\n"
                f"与主角关系:{cp.get('relation_to_mc', '未知')}\n"
                f"与其他配角关系:{cp.get('relations_to_others', '无')}\n"
                f"高光/宿命预设:{cp.get('climax_or_fate', '无')}\n"
                f"前史:{cp.get('background', '')}\n"
                f"欲望vs需求:{cp.get('desire_vs_need', '')}\n"
                f"秘密:{cp.get('secrets', '')}\n"
                f"恐惧:{cp.get('fears', '')}\n"
                f"社交面具:{cp.get('social_masks', '')}\n"
                f"成长代价:{cp.get('growth_cost', '')}\n"
                f"语言风格样本:{cp.get('speech_samples', '')}\n"
                f"行为标志样本:{cp.get('behavior_samples', '')}\n"
                f"关系演变:{cp.get('relationship_evolution', '')}\n"
                f"能力体系:{cp.get('abilities', '')}\n"
                f"弧光轨迹:{cp.get('arc_trajectory', '')}\n"
                f"两难困境:{cp.get('internal_dilemma', '')}\n"
                f"决策模式:{cp.get('decision_pattern', '')}\n"
                f"认知偏差:{cp.get('cognitive_bias', '')}\n"
                f"转变触发器:{cp.get('transformation_trigger', '')}\n"
                f"对比设计:{cp.get('contrast_design', '')}"
            )
            chroma_char_items.append(chroma_item)
        bulk_upsert_to_chroma(
            "character_profiles_kb",
            chroma_char_items,
            id_fields=["book_name", "name", "profile"],
            text_field="_chroma_text",
            metadata_fields=["book_name", "author", "category", "name", "role_type"],
        )

        # 编年史入库
        for wt in results.get("world_timeline", []):
            wt_id = generate_id(wt["book_name"], wt["era_or_year"], wt["event_name"])
            cursor.execute(
                "INSERT OR IGNORE INTO world_timeline VALUES (?,?,?,?,?,?)",
                (
                    wt_id,
                    wt["book_name"],
                    wt["era_or_year"],
                    wt["event_name"],
                    wt["event_description"],
                    wt["impact"],
                ),
            )
            stats["world_timeline"] += 1

        # 势力关系网络入库
        for fn in results.get("faction_networks", []):
            fn_id = generate_id(fn["book_name"], fn["faction_a"], fn["faction_b"])
            cursor.execute(
                "INSERT OR REPLACE INTO faction_networks VALUES (?,?,?,?,?,?,?,?)",
                (
                    fn_id,
                    fn["book_name"],
                    fn["faction_a"],
                    fn["faction_b"],
                    fn.get("relation_type", ""),
                    fn.get("relation_detail", ""),
                    fn.get("stability", ""),
                    fn.get("key_events", ""),
                ),
            )
            stats["faction_networks"] += 1

        self.db.commit()
        logger.info(
            f"   ✅ [阶段D战报] 世界观: {stats['world_settings']} | "
            f"人物档案: {stats['character_profiles']} | 编年史: {stats['world_timeline']} | "
            f"势力网络: {stats['faction_networks']}"
        )
        return stats
