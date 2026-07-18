"""
Stage H: 全书级宏观分析
使用 qwen14b:latest 模型，提取三幕结构、主线支线、情感曲线、高潮/张力点分布、象征体系
"""

import json
import logging
from typing import List, Dict, Any
from stages.base import BaseStage
from core.ollama_client import ollama_chat, safe_parse_json
from core.utils import generate_id
from core.graph import get_graph_manager

logger = logging.getLogger(__name__)


class StageH(BaseStage):
    """Stage H: 全书级宏观分析"""

    def __init__(self, book_name: str, category: str):
        super().__init__("H", book_name, category)

    def run(
        self,
        stage_a_res: List[Dict],
        stage_e_res: Dict[str, List[Dict]],
        **kwargs,
    ) -> Dict[str, List[Dict]]:
        """
        执行 Stage H

        Args:
            stage_a_res: Stage A 的章节摘要结果
            stage_e_res: Stage E 的卷大纲结果

        Returns:
            包含 plot_lines, revelation_pacing, emotion_transition_patterns, information_management,
            climax_buildup_chains, conflict_escalation 的字典
        """
        logger.info(f"=== 阶段八：全书级宏观分析 ({self.book_name}) ===")

        result = {
            "plot_lines": [],
            "revelation_pacing": [],
            "emotion_transition_patterns": [],
            "information_management": [],
            "climax_buildup_chains": [],
            "conflict_escalation": [],
        }

        if not stage_a_res:
            logger.warning("⚠️ [阶段H] 没有 Stage A 数据，跳过全书宏观分析。")
            return result

        # 构建全书摘要文本（超过阈值时均匀采样，确保开头/中间/结尾都有覆盖）
        max_summary_chars = (
            4000  # num_ctx=14336, num_predict=4096, safe≈6493, 模板~2000, 留4000给摘要
        )
        summary_lines = [
            f"{ch.get('id', '未知章节')}: {ch.get('summary', '无摘要')}"
            for ch in stage_a_res
        ]
        summaries_text = "\n".join(summary_lines)

        if len(summaries_text) > max_summary_chars:
            # 均匀采样：前 25% + 中 50% + 后 25% 的容量
            total = len(summary_lines)
            # 估算每行平均长度
            avg_line_len = len(summaries_text) / max(total, 1)
            max_lines = max(int(max_summary_chars / avg_line_len), 20)

            # 采样策略：前 1/4、中间 1/2、后 1/4
            head_count = max_lines // 4
            tail_count = max_lines // 4
            mid_count = max_lines - head_count - tail_count

            head_indices = list(range(min(head_count, total)))
            tail_start = max(total - tail_count, head_count)
            tail_indices = list(range(tail_start, total))

            # 中间部分均匀采样
            mid_range = list(range(head_count, tail_start))
            if len(mid_range) > mid_count and mid_count > 0:
                step = len(mid_range) / mid_count
                mid_indices = [mid_range[int(i * step)] for i in range(mid_count)]
            else:
                mid_indices = mid_range

            sampled_indices = sorted(set(head_indices + mid_indices + tail_indices))
            sampled_lines = [summary_lines[i] for i in sampled_indices]
            summaries_text = "\n".join(sampled_lines)
            logger.info(
                f"[阶段H] 全书摘要超过 {max_summary_chars} 字符，"
                f"从 {total} 章中均匀采样 {len(sampled_indices)} 章"
            )

        # 构建卷大纲文本
        volumes_text = ""
        for m in stage_e_res.get("macro_outlines", []):
            volumes_text += f"\n【第{m.get('volume_index', '')}卷 ({m.get('chapter_range', '')})】\n"
            volumes_text += f"主题: {m.get('theme', '')}\n"
            volumes_text += f"冲突: {m.get('conflict', '')}\n"
            volumes_text += f"弧光: {m.get('arc', '')}\n"

        # 分三组顺序调用，前一组结果作为上下文传递给后续组
        logger.info("📊 [阶段H] 开始分组提取（结构组/技法组/类型组）...")

        # 断点恢复
        cache = self.load_cache()
        groups_done = set(cache.get("groups_done", [])) if cache else set()
        if cache and "result" in cache:
            result.update({k: v for k, v in cache["result"].items() if v})

        # 第一组：结构组 — 内联 plot_lines 提取
        if "structure" not in groups_done:
            prompt_pl = f"""你是顶级的叙事结构分析师。请从以下全书摘要中提取所有剧情线。

书名：{self.book_name}  分类：{self.category}
{summaries_text}

请输出纯 JSON 格式：{{{{"plot_lines": [{{{{"line_type": "main/romance/mystery/fear/backstory", "theme": "主题概括(30字内)", "chapter_distribution": "覆盖章节", "milestones": ["关键里程碑"]}}}}]}}}}
(禁止反引号)"""
            try:
                resp = ollama_chat(prompt_pl, 0.2, "H")
                data = safe_parse_json(resp)
                if data:
                    pl = data.get("plot_lines", [])
                    if pl:
                        result["plot_lines"] = [
                            {"book_name": self.book_name, **p} for p in pl
                        ]
            except Exception as e:
                logger.warning(f"⚠️ [阶段H] plot_lines提取失败: {{e}}")
            groups_done.add("structure")
            self.save_cache({"groups_done": list(groups_done), "result": result})
            structure_data = {"plot_lines": result.get("plot_lines", [])}
        else:
            logger.info("✅ [阶段H] 结构组已完成，跳过")
            cached_result = cache.get("result", {})
            structure_data = {"plot_lines": cached_result.get("plot_lines", [])}

        # 构建结构组摘要，供后续组参考
        structure_context = self._build_group_context(structure_data)

        # 第二组：技法组
        tk = [
            "revelation_pacing",
            "emotion_transition_patterns",
            "information_management",
            "climax_buildup_chains",
            "conflict_escalation",
        ]
        if "technique" not in groups_done:
            technique_data = self._extract_technique_group(
                summaries_text, volumes_text, structure_context
            )
            for key in tk:
                if key in technique_data:
                    result[key] = technique_data[key]
            groups_done.add("technique")
            self.save_cache({"groups_done": list(groups_done), "result": result})
        else:
            logger.info("✅ [阶段H] 技法组已完成，跳过")
        # 第三组：类型组全部子表已废弃，跳过

        logger.info(
            f"✅ [阶段H战报] 剧情线: {len(result['plot_lines'])} | "
            f"信息揭露: {len(result['revelation_pacing'])} | "
            f"情感铺垫: {len(result['emotion_transition_patterns'])} | "
            f"信息管理: {len(result['information_management'])} | "
            f"高潮构建: {len(result['climax_buildup_chains'])} | "
            f"冲突升级: {len(result['conflict_escalation'])}"
        )

        # 更新知识图谱：剧情线节点
        self._update_graph(result)

        return result

    def _update_graph(self, result: Dict[str, List[Dict]]):
        """将剧情线同步到知识图谱"""
        try:
            graph = get_graph_manager()

            for plot_line in result.get("plot_lines", []):
                line_type = plot_line.get("line_type", "unknown")
                theme = plot_line.get("theme", "unknown")
                line_id = f"plot:{self.book_name}:{line_type}:{theme}"

                graph.add_node(
                    line_id,
                    node_type="plot_line",
                    book_name=self.book_name,
                    line_type=line_type,
                    theme=theme,
                    chapter_distribution=plot_line.get("chapter_distribution", ""),
                )

            graph.save()
            logger.info(
                f"📊 [阶段H] 知识图谱已更新 {len(result.get('plot_lines', []))} 个剧情线节点"
            )
        except Exception as e:
            logger.warning(f"⚠️ [阶段H] 知识图谱更新失败: {e}")

    def _build_group_context(self, group_data: Dict[str, List[Dict]]) -> str:
        """构建前一组的分析摘要，供后续组参考"""
        context_lines = ["【前序分析结果摘要】"]

        for key, items in group_data.items():
            if not items:
                continue

            if key == "plot_lines":
                main_plots = [p for p in items if p.get("line_type") == "main"]
                if main_plots:
                    context_lines.append(
                        f"- 主线主题: {main_plots[0].get('theme', '')}"
                    )
                context_lines.append(
                    f"- 支线数量: {len([p for p in items if p.get('line_type') == 'subplot'])}"
                )

            elif key == "information_management":
                strategies = [item.get("strategy_type", "") for item in items[:3]]
                context_lines.append(f"- 信息管理策略: {', '.join(strategies)}")

            elif key == "climax_buildup_chains":
                climax_names = [item.get("climax_name", "") for item in items[:3]]
                context_lines.append(f"- 主要高潮: {', '.join(climax_names)}")

            elif key == "conflict_escalation":
                conflict_lines = [item.get("conflict_line", "") for item in items[:3]]
                context_lines.append(f"- 冲突线: {', '.join(conflict_lines)}")

        if len(context_lines) == 1:
            return ""

        return "\n".join(context_lines)

    def _extract_technique_group(
        self, summaries_text: str, volumes_text: str, prior_context: str = ""
    ) -> Dict[str, List[Dict]]:
        """提取技法组：revelation_pacing, chapter_patterns, emotion_transition_patterns, information_management, climax_buildup_chains, conflict_escalation"""
        result = {}

        # 构建包含前序分析结果的 Prompt
        prior_section = ""
        if prior_context:
            prior_section = f"\n{prior_context}\n请参考以上结构分析结果，确保技法分析与结构分析保持一致。\n"

        prompt = f"""你是顶级的文学评论家。请根据《{self.book_name}》({self.category})的全书摘要和卷大纲，分析写作技法。

【书名】{self.book_name} 【分类】{self.category}
{prior_section}
【全书章节摘要】
{summaries_text}

【卷大纲】
{volumes_text}

请输出纯 JSON 格式：
{{
  "revelation_pacing": [
    {{
      "revelation_name": "关键信息名称",
      "reveal_chapter": "揭露章节",
      "reveal_method": "揭露方式",
      "impact": "对剧情的影响(50字内)"
    }}
  ],
  "emotion_transition_patterns": [
    {{
      "transition_type": "情感转变类型",
      "foreshadowing_method": "铺垫方式(50字内)"
    }}
  ],
  "information_management": [
    {{
      "strategy_type": "信息管理策略类型",
      "target_info": "被管理的信息内容",
      "conceal_method": "隐瞒方式(50字内)",
      "reveal_timing": "揭露时机",
      "dramatic_purpose": "戏剧目的(50字内)"
    }}
  ],
  "climax_buildup_chains": [
    {{
      "climax_name": "高潮名称",
      "climax_chapter": "高潮章节",
      "buildup_steps": ["铺垫步骤1"],
      "tension_escalation": "张力升级方式(50字内)"
    }}
  ],
  "conflict_escalation": [
    {{
      "conflict_line": "冲突线名称",
      "escalation_steps": ["升级步骤1"],
      "escalation_pattern": "升级模式(50字内)"
    }}
  ]
}}
(⚠️核心要求：必须分析信息揭露节奏！必须总结章节模式！必须分析情感转变铺垫！必须分析信息管理策略！必须分析高潮构建链！必须分析冲突升级！禁止反引号)"""

        try:
            resp = ollama_chat(prompt, 0.2, "H")
            data = safe_parse_json(resp)
            if data:
                revelations = data.get("revelation_pacing", [])
                if revelations:
                    result["revelation_pacing"] = []
                    for rev in revelations:
                        if isinstance(rev, dict) and rev.get("revelation_name"):
                            result["revelation_pacing"].append(
                                {
                                    "book_name": self.book_name,
                                    "revelation_name": rev.get("revelation_name"),
                                    "reveal_chapter": rev.get("reveal_chapter", ""),
                                    "reveal_method": rev.get("reveal_method", ""),
                                    "impact": rev.get("impact", ""),
                                }
                            )

                emotion_transitions = data.get("emotion_transition_patterns", [])
                if emotion_transitions:
                    result["emotion_transition_patterns"] = []
                    for et in emotion_transitions:
                        if isinstance(et, dict) and et.get("transition_type"):
                            result["emotion_transition_patterns"].append(
                                {
                                    "book_name": self.book_name,
                                    "transition_type": et.get("transition_type"),
                                    "foreshadowing_method": et.get(
                                        "foreshadowing_method", ""
                                    ),
                                    "original_example": et.get("original_example", ""),
                                }
                            )

                info_mgmt = data.get("information_management", [])
                if info_mgmt:
                    result["information_management"] = []
                    for im in info_mgmt:
                        if (
                            isinstance(im, dict)
                            and im.get("strategy_type")
                            and im.get("target_info")
                        ):
                            result["information_management"].append(
                                {
                                    "book_name": self.book_name,
                                    "strategy_type": im.get("strategy_type"),
                                    "target_info": im.get("target_info"),
                                    "conceal_method": im.get("conceal_method", ""),
                                    "reveal_timing": im.get("reveal_timing", ""),
                                    "dramatic_purpose": im.get("dramatic_purpose", ""),
                                }
                            )

                climax_chains = data.get("climax_buildup_chains", [])
                if climax_chains:
                    result["climax_buildup_chains"] = []
                    for cc in climax_chains:
                        if isinstance(cc, dict) and cc.get("climax_name"):
                            result["climax_buildup_chains"].append(
                                {
                                    "book_name": self.book_name,
                                    "climax_name": cc.get("climax_name"),
                                    "climax_chapter": cc.get("climax_chapter", ""),
                                    "buildup_steps": cc.get("buildup_steps", []),
                                    "tension_escalation": cc.get(
                                        "tension_escalation", ""
                                    ),
                                }
                            )

                conflict_esc = data.get("conflict_escalation", [])
                if conflict_esc:
                    result["conflict_escalation"] = []
                    for ce in conflict_esc:
                        if isinstance(ce, dict) and ce.get("conflict_line"):
                            result["conflict_escalation"].append(
                                {
                                    "book_name": self.book_name,
                                    "conflict_line": ce.get("conflict_line"),
                                    "escalation_steps": ce.get("escalation_steps", []),
                                    "escalation_pattern": ce.get(
                                        "escalation_pattern", ""
                                    ),
                                }
                            )
        except Exception as e:
            logger.warning(f"⚠️ [阶段H] 技法组提取失败: {e}")

        return result

    def insert(self, results: Dict[str, List[Dict]]) -> Dict[str, int]:
        """将 Stage H 结果写入数据库"""
        cursor = self.db.connect().cursor()
        stats = {
            "plot_lines": 0,
            "revelation_pacing": 0,
            "emotion_transition_patterns": 0,
            "information_management": 0,
            "climax_buildup_chains": 0,
            "conflict_escalation": 0,
        }
        # 主线支线入库
        for pl in results.get("plot_lines", []):
            pl_id = generate_id(pl["book_name"], pl["line_type"], pl.get("theme", ""))
            cursor.execute(
                "INSERT OR REPLACE INTO plot_lines VALUES (?,?,?,?,?,?)",
                (
                    pl_id,
                    pl["book_name"],
                    pl["line_type"],
                    pl.get("theme", ""),
                    pl.get("chapter_distribution", ""),
                    json.dumps(pl.get("milestones", []), ensure_ascii=False),
                ),
            )
            stats["plot_lines"] += 1

        # 信息揭露节奏入库
        for rev in results.get("revelation_pacing", []):
            rev_id = generate_id(
                rev["book_name"], rev["revelation_name"], rev.get("reveal_chapter", "")
            )
            cursor.execute(
                "INSERT OR REPLACE INTO revelation_pacing VALUES (?,?,?,?,?,?)",
                (
                    rev_id,
                    rev["book_name"],
                    rev["revelation_name"],
                    rev.get("reveal_chapter", ""),
                    rev.get("reveal_method", ""),
                    rev.get("impact", ""),
                ),
            )
            stats["revelation_pacing"] += 1

        # 情感转变铺垫入库
        for et in results.get("emotion_transition_patterns", []):
            et_id = generate_id(et["book_name"], et["transition_type"])
            cursor.execute(
                "INSERT OR REPLACE INTO emotion_transition_patterns VALUES (?,?,?,?)",
                (
                    et_id,
                    et["book_name"],
                    et["transition_type"],
                    et.get("foreshadowing_method", ""),
                ),
            )
            stats["emotion_transition_patterns"] += 1

        # 信息管理策略入库
        for im in results.get("information_management", []):
            im_id = generate_id(im["book_name"], im["strategy_type"], im["target_info"])
            cursor.execute(
                "INSERT OR REPLACE INTO information_management VALUES (?,?,?,?,?,?,?)",
                (
                    im_id,
                    im["book_name"],
                    im["strategy_type"],
                    im["target_info"],
                    im.get("conceal_method", ""),
                    im.get("reveal_timing", ""),
                    im.get("dramatic_purpose", ""),
                ),
            )
            stats["information_management"] += 1

        # 高潮构建链入库
        for cc in results.get("climax_buildup_chains", []):
            cc_id = generate_id(cc["book_name"], cc["climax_name"])
            cursor.execute(
                "INSERT OR REPLACE INTO climax_buildup_chains VALUES (?,?,?,?,?,?)",
                (
                    cc_id,
                    cc["book_name"],
                    cc["climax_name"],
                    cc.get("climax_chapter", ""),
                    json.dumps(cc.get("buildup_steps", []), ensure_ascii=False),
                    cc.get("tension_escalation", ""),
                ),
            )
            stats["climax_buildup_chains"] += 1

        # 冲突升级阶梯入库
        for ce in results.get("conflict_escalation", []):
            ce_id = generate_id(ce["book_name"], ce["conflict_line"])
            cursor.execute(
                "INSERT OR REPLACE INTO conflict_escalation VALUES (?,?,?,?,?)",
                (
                    ce_id,
                    ce["book_name"],
                    ce["conflict_line"],
                    json.dumps(ce.get("escalation_steps", []), ensure_ascii=False),
                    ce.get("escalation_pattern", ""),
                ),
            )
            stats["conflict_escalation"] += 1

        self.db.commit()
        logger.info(
            f"   ✅ [阶段H战报] 剧情线: {stats['plot_lines']} | "
            f"信息揭露: {stats['revelation_pacing']} | "
            f"情感铺垫: {stats['emotion_transition_patterns']} | "
            f"信息管理: {stats['information_management']} | "
            f"高潮构建: {stats['climax_buildup_chains']} | "
            f"冲突升级: {stats['conflict_escalation']}"
        )
        return stats
