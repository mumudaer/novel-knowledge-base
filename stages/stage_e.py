"""
Stage E: 宏观大纲与卷节拍聚合 + 章节功能分类
使用 qwen2.5:7b 模型，将单章摘要聚合为卷大纲，并分析章节功能
"""
import json
import logging
from typing import List, Dict, Any
from tqdm import tqdm
from stages.base import BaseStage
from core.ollama_client import ollama_chat, safe_parse_json
from core.utils import generate_id

logger = logging.getLogger(__name__)


class StageE(BaseStage):
    """Stage E: 宏观大纲与卷节拍聚合 + 章节功能分类"""

    def __init__(self, book_name: str, category: str):
        super().__init__("E", book_name, category)

    def run(
        self,
        stage_a_res: List[Dict],
        chapters_per_volume: int = None,
        **kwargs,
    ) -> Dict[str, List[Dict]]:
        """
        执行 Stage E

        Args:
            stage_a_res: Stage A 的章节摘要结果
            chapters_per_volume: 每卷包含的章节数（None 表示自适应）

        Returns:
            包含 macro_outlines, plot_foreshadowing, entity_state_tracker, chapter_functions, setting_evolutions 的字典
        """
        # 自适应卷划分：根据总章数动态调整
        total_chapters = len(stage_a_res)
        if chapters_per_volume is None:
            if total_chapters <= 100:
                chapters_per_volume = 20  # 短篇小说：20章/卷
            elif total_chapters <= 300:
                chapters_per_volume = 50  # 中篇小说：50章/卷
            else:
                chapters_per_volume = 100  # 长篇小说：100章/卷
        
        logger.info(f"=== 阶段五：宏观大纲与卷节拍聚合 (每 {chapters_per_volume} 章为一卷) ({self.book_name}) ===")

        result = {
            "macro_outlines": [],
            "plot_foreshadowing": [],
            "entity_state_tracker": [],
            "chapter_functions": [],
            "setting_evolutions": [],
        }

        if not stage_a_res:
            logger.warning("⚠️ [阶段E] 没有 Stage A 数据，跳过宏观聚合。")
            return result

        # 断点恢复
        cache = self.load_cache()
        completed_vol_outlines = set()
        completed_vol_functions = set()
        if cache and "result" in cache:
            cached_result = cache["result"]
            result["macro_outlines"] = cached_result.get("macro_outlines", [])
            result["plot_foreshadowing"] = cached_result.get("plot_foreshadowing", [])
            result["entity_state_tracker"] = cached_result.get("entity_state_tracker", [])
            result["chapter_functions"] = cached_result.get("chapter_functions", [])
            result["setting_evolutions"] = cached_result.get("setting_evolutions", [])
            completed_vol_outlines = set(cache.get("completed_vol_outlines", []))
            completed_vol_functions = set(cache.get("completed_vol_functions", []))
            if completed_vol_outlines:
                logger.info(f"✅ [阶段E] 恢复断点：卷大纲已完成 {len(completed_vol_outlines)} 卷，章节功能已完成 {len(completed_vol_functions)} 批次")

        # 卷大纲聚合
        volumes = [
            stage_a_res[i : i + chapters_per_volume]
            for i in range(0, len(stage_a_res), chapters_per_volume)
        ]
        logger.info(f"📚 [阶段E] 共 {len(stage_a_res)} 章，将聚合为 {len(volumes)} 个宏观卷大纲。")

        for vol_idx, vol_chapters in enumerate(tqdm(volumes, desc="聚合卷大纲")):
            # 断点跳过已完成卷
            if vol_idx in completed_vol_outlines:
                continue

            start_chap = vol_idx * chapters_per_volume + 1
            end_chap = (vol_idx + 1) * chapters_per_volume

            summaries_text = "\n".join([
                f"{ch.get('id', '未知章节')}: {ch.get('summary', '无摘要')}"
                for ch in vol_chapters
            ])
            if len(summaries_text) > 6000:
                summaries_text = summaries_text[:6000] + "\n...(截断)"

            # 卷大纲 Prompt
            prompt_e = f"""你是资深文学主编。根据《{self.book_name}》({self.category})第{start_chap}-{end_chap}章摘要，提炼宏观大纲，并盘点本卷【全阵营人物状态变更】与【伏笔/意象悬念】。
【摘要】
{summaries_text}

输出纯JSON：
{{
  "volume_theme": "本卷核心主题/探讨的哲学或社会问题",
  "core_conflict": "核心冲突与对立面(人与人/人与社会/人与自我)",
  "plot_beats": ["节拍1:起势/铺垫", "节拍2:发展/冲突加剧", "节拍3:高潮/爆发", "节拍4:尾声/余韵"],
  "character_arc": "主角/核心视角的认知跃迁或心理异化",
  "foreshadowing": [
    {{
      "hook_name": "伏笔/悬念/核心意象名称",
      "action": "埋下(plant) 或 回收/呼应(resolve)",
      "description": "伏笔内容简述或意象呼应方式",
      "resolution_excerpt": "如果是回收类伏笔，摘录回收时的原文片段(100-200字)；如果是埋下类伏笔，留空"
    }}
  ],
  "state_changes": [
    {{
      "entity_name": "人物名或重要物品/意象名",
      "change_type": "变更类型(能力跃迁/受伤/获得资源/关系恶化/阵营背叛/心理异化/死亡退场)",
      "change_description": "本卷状态变更详述"
    }}
  ],
  "setting_evolutions": [
    {{
      "setting_module": "设定模块(如:力量体系/政治格局/社会规则)",
      "setting_entity": "具体设定实体",
      "evolution_type": "演变类型(规则打破/格局洗牌/体系升级/新增设定)",
      "before_state": "演变前状态(50字内)",
      "after_state": "演变后状态(50字内)",
      "trigger_event": "触发演变的关键事件(50字内)"
    }}
  ]
}}
(注意：state_changes 必须包含重要配角的动态变化！回收类伏笔必须附带原文摘录！setting_evolutions 追踪世界观设定在本卷中的变化！禁止反引号)"""

            try:
                resp = ollama_chat(prompt_e, 0.2, "E")
                data = safe_parse_json(resp)
                if data and data.get("volume_theme"):
                    safe_beats = data.get("plot_beats", [])
                    if not isinstance(safe_beats, list):
                        safe_beats = []

                    result["macro_outlines"].append({
                        "book_name": self.book_name,
                        "category": self.category,
                        "volume_index": vol_idx + 1,
                        "chapter_range": f"{start_chap}-{end_chap}",
                        "theme": data.get("volume_theme"),
                        "conflict": data.get("core_conflict", ""),
                        "beats": safe_beats,
                        "arc": data.get("character_arc", ""),
                    })

                    for fs in data.get("foreshadowing", []):
                        if isinstance(fs, dict) and fs.get("hook_name"):
                            action = fs.get("action", "plant")
                            status = "已填" if "resolve" in action.lower() or "填" in action else "未填"
                            result["plot_foreshadowing"].append({
                                "book_name": self.book_name,
                                "hook_name": fs["hook_name"],
                                "planted_chapter": f"{start_chap}-{end_chap}" if status == "未填" else "",
                                "planned_payoff": fs.get("description", ""),
                                "status": status,
                                "resolved_chapter": f"{start_chap}-{end_chap}" if status == "已填" else "",
                                "resolution_excerpt": fs.get("resolution_excerpt", ""),
                            })

                    for sc in data.get("state_changes", []):
                        if isinstance(sc, dict) and sc.get("entity_name") and sc.get("change_description"):
                            result["entity_state_tracker"].append({
                                "book_name": self.book_name,
                                "entity_name": sc["entity_name"],
                                "chapter_range": f"{start_chap}-{end_chap}",
                                "current_state_json": json.dumps({
                                    "type": sc.get("change_type", "状态变更"),
                                    "detail": sc["change_description"],
                                }, ensure_ascii=False),
                            })

                    # 解析设定演变
                    for se in data.get("setting_evolutions", []):
                        if isinstance(se, dict) and se.get("setting_module") and se.get("setting_entity"):
                            result["setting_evolutions"].append({
                                "book_name": self.book_name,
                                "setting_module": se.get("setting_module"),
                                "setting_entity": se.get("setting_entity"),
                                "chapter_range": f"{start_chap}-{end_chap}",
                                "evolution_type": se.get("evolution_type", "未知"),
                                "before_state": se.get("before_state", ""),
                                "after_state": se.get("after_state", ""),
                                "trigger_event": se.get("trigger_event", ""),
                            })
            except Exception as e:
                logger.warning(f"⚠️ [阶段E] 聚合第 {vol_idx+1} 卷失败: {e}")

            # 标记完成并保存断点
            completed_vol_outlines.add(vol_idx)
            self.save_cache({
                "completed_vol_outlines": list(completed_vol_outlines),
                "completed_vol_functions": list(completed_vol_functions),
                "result": result,
            })

        # 章节功能分类（按卷批量处理，减少 LLM 调用次数）
        logger.info(f"📋 [阶段E] 开始章节功能分类分析（按卷批量处理）...")
        
        for vol_idx, vol_chapters in enumerate(tqdm(volumes, desc="章节功能分类（按卷）")):
            start_chap = vol_idx * chapters_per_volume + 1
            
            # 每卷内再分批，每批最多 10 章
            batch_size = 10
            for batch_start in range(0, len(vol_chapters), batch_size):
                # 断点跳过已完成批次
                batch_key = f"{vol_idx}_{batch_start}"
                if batch_key in completed_vol_functions:
                    continue

                batch_chapters = vol_chapters[batch_start:batch_start + batch_size]
                
                # 构建批量章节文本
                chapters_text = "\n\n".join([
                    f"【第{ch.get('id', '未知')}章】\n摘要: {ch.get('summary', '无摘要')[:300]}\n正文片段: {ch.get('text', '')[:300]}"
                    for ch in batch_chapters
                ])
                
                prompt_func = f"""你是文学结构分析师。批量分析《{self.book_name}》({self.category})以下{len(batch_chapters)}章的结构功能。
【章节列表】
{chapters_text}

输出纯JSON（包含{len(batch_chapters)}个章节的分析结果）：
{{
  "chapter_functions": [
    {{
      "chapter_id": "章节ID",
      "function_type": "章节功能类型(战斗章/过渡章/高潮章/日常章/揭秘章/铺垫章/转折章)",
      "structure_pattern": {{
        "opening": "开头方式(场景切入/对话切入/悬念切入/回忆切入/动作切入)",
        "development": "发展方式(冲突升级/信息揭露/情感铺垫/多线交织)",
        "ending": "结尾方式(悬念收尾/情感余韵/转折钩子/平静过渡)"
      }},
      "hook_type": "章末钩子类型(悬念型/反转型/情感型/危机型/无钩子)",
      "hook_content": "钩子内容简述(20字内)",
      "information_gap": {{
        "reader_knows": ["读者知道但角色不知道的信息"],
        "character_knows": ["角色知道但读者不知道的信息"],
        "dramatic_effect": "产生的戏剧效果(如:戏剧性讽刺/悬念/期待)"
      }},
      "active_plotlines": ["本章推进的剧情线(如:主线-复仇/支线-感情线/支线-势力线/支线-成长线)"]
    }}
  ]
}}
(禁止反引号，如果没有信息差，对应数组留空。active_plotlines 必须标注本章推进了哪些剧情线。必须返回{len(batch_chapters)}个章节的分析结果)"""

                try:
                    resp = ollama_chat(prompt_func, 0.2, "E")
                    data = safe_parse_json(resp)
                    if data and isinstance(data.get("chapter_functions"), list):
                        for cf in data["chapter_functions"]:
                            if isinstance(cf, dict) and cf.get("chapter_id"):
                                result["chapter_functions"].append({
                                    "book_name": self.book_name,
                                    "chapter_id": cf.get("chapter_id", ""),
                                    "function_type": cf.get("function_type", "未知"),
                                    "structure_pattern": cf.get("structure_pattern", {}),
                                    "hook_type": cf.get("hook_type", "无钩子"),
                                    "hook_content": cf.get("hook_content", ""),
                                    "information_gap": cf.get("information_gap", {}),
                                    "active_plotlines": cf.get("active_plotlines", []),
                                })
                except Exception as e:
                    logger.warning(f"⚠️ [阶段E] 章节功能分类批量处理失败 (卷{vol_idx+1}, 批次{batch_start//batch_size+1}): {e}")

                # 标记完成并保存断点
                completed_vol_functions.add(batch_key)
                self.save_cache({
                    "completed_vol_outlines": list(completed_vol_outlines),
                    "completed_vol_functions": list(completed_vol_functions),
                    "result": result,
                })

        logger.info(
            f"✅ [阶段E战报] 卷大纲: {len(result['macro_outlines'])} 卷 | "
            f"伏笔: {len(result['plot_foreshadowing'])} 条 | "
            f"状态快照: {len(result['entity_state_tracker'])} 条 | "
            f"章节功能: {len(result['chapter_functions'])} 章 | "
            f"设定演变: {len(result['setting_evolutions'])} 条"
        )
        return result

    def insert(self, results: Dict[str, List[Dict]]) -> Dict[str, int]:
        """将 Stage E 结果写入数据库"""
        cursor = self.db.connect().cursor()
        stats = {"macro_outlines": 0, "foreshadowing": 0, "state_tracker": 0, "chapter_functions": 0, "setting_evolutions": 0}

        # 宏观大纲入库
        for m in results.get("macro_outlines", []):
            m_id = generate_id(m["book_name"], f"vol_{m['volume_index']}")
            beats_str = json.dumps(m.get("beats", []), ensure_ascii=False)
            cursor.execute(
                "INSERT OR REPLACE INTO macro_outlines VALUES (?,?,?,?,?,?,?,?,?)",
                (m_id, m["book_name"], m["category"], m["volume_index"],
                 m["chapter_range"], m["theme"], m["conflict"], beats_str, m["arc"]),
            )
            stats["macro_outlines"] += 1

        # ChromaDB: 宏观大纲
        m_ids, m_docs, m_metas = [], [], []
        for m in results.get("macro_outlines", []):
            mid = generate_id(m["book_name"], f"vol_{m['volume_index']}")
            m_ids.append(mid)
            beats_str = "\n".join([f"- {b}" for b in m.get("beats", [])])
            m_docs.append(f"卷主题:{m['theme']}\n冲突:{m['conflict']}\n弧光:{m['arc']}\n节拍:\n{beats_str}")
            m_metas.append({
                "book_name": m["book_name"],
                "category": m["category"],
                "volume_index": m["volume_index"],
                "chapter_range": m["chapter_range"],
                "module": "宏观卷大纲",
            })
        if m_ids:
            self.chroma.upsert_batch("macro_outlines_kb", m_ids, m_docs, m_metas)

        # 伏笔追踪入库（8个字段）
        for fs in results.get("plot_foreshadowing", []):
            fs_id = generate_id(fs["book_name"], fs["hook_name"], fs.get("planted_chapter", ""), fs.get("resolved_chapter", ""))
            cursor.execute(
                "INSERT OR IGNORE INTO plot_foreshadowing VALUES (?,?,?,?,?,?,?,?)",
                (fs_id, fs["book_name"], fs["hook_name"], fs.get("planted_chapter", ""),
                 fs.get("planned_payoff", ""), fs.get("status", ""), fs.get("resolved_chapter", ""),
                 fs.get("resolution_excerpt", "")),
            )
            stats["foreshadowing"] += 1

        # 实体状态快照入库
        for es in results.get("entity_state_tracker", []):
            es_id = generate_id(es["book_name"], es["entity_name"], es["chapter_range"])
            cursor.execute(
                "INSERT OR IGNORE INTO entity_state_tracker VALUES (?,?,?,?,?)",
                (es_id, es["book_name"], es["entity_name"], es["chapter_range"],
                 es.get("current_state_json", "")),
            )
            stats["state_tracker"] += 1

        # 章节功能分类入库（9个字段）
        for cf in results.get("chapter_functions", []):
            cf_id = generate_id(cf["book_name"], cf["chapter_id"])
            cursor.execute(
                "INSERT OR REPLACE INTO chapter_functions VALUES (?,?,?,?,?,?,?,?,?)",
                (cf_id, cf["book_name"], cf["chapter_id"], cf.get("function_type", ""),
                 json.dumps(cf.get("structure_pattern", {}), ensure_ascii=False),
                 cf.get("hook_type", ""), cf.get("hook_content", ""),
                 json.dumps(cf.get("information_gap", {}), ensure_ascii=False),
                 json.dumps(cf.get("active_plotlines", []), ensure_ascii=False)),
            )
            stats["chapter_functions"] += 1

        # 设定演变入库
        for se in results.get("setting_evolutions", []):
            se_id = generate_id(se["book_name"], se["setting_module"], se["setting_entity"], se["chapter_range"])
            cursor.execute(
                "INSERT OR REPLACE INTO setting_evolutions VALUES (?,?,?,?,?,?,?,?,?)",
                (se_id, se["book_name"], se["setting_module"], se["setting_entity"],
                 se["chapter_range"], se.get("evolution_type", ""),
                 se.get("before_state", ""), se.get("after_state", ""),
                 se.get("trigger_event", "")),
            )
            stats["setting_evolutions"] += 1

        self.db.commit()
        logger.info(
            f"   ✅ [阶段E战报] 卷大纲: {stats['macro_outlines']} | "
            f"伏笔: {stats['foreshadowing']} | 状态快照: {stats['state_tracker']} | "
            f"章节功能: {stats['chapter_functions']} | 设定演变: {stats['setting_evolutions']}"
        )
        return stats
