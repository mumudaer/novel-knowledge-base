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
            包含 book_structure, plot_lines, emotional_arc, climax_point_distribution, symbol_system,
            revelation_pacing, chapter_patterns, emotion_transition_patterns, information_management,
            climax_buildup_chains, conflict_escalation 的字典
        """
        logger.info(f"=== 阶段八：全书级宏观分析 ({self.book_name}) ===")

        result = {
            "book_structure": [],
            "plot_lines": [],
            "emotional_arc": [],
            "climax_point_distribution": [],
            "symbol_system": [],
            "revelation_pacing": [],
            "chapter_patterns": [],
            "emotion_transition_patterns": [],
            "information_management": [],
            "climax_buildup_chains": [],
            "conflict_escalation": [],
            "romance_lines": [],
            "mystery_clues": [],
            "fear_building": [],
            "progression_systems": [],
            "genre_specific_techniques": [],
            "pov_switching_patterns": [],
        }

        if not stage_a_res:
            logger.warning("⚠️ [阶段H] 没有 Stage A 数据，跳过全书宏观分析。")
            return result

        # 构建全书摘要文本（超过阈值时均匀采样，确保开头/中间/结尾都有覆盖）
        max_summary_chars = 5000  # 配合 num_ctx=14336, num_predict=2048, safe≈11788 tokens
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

        # 第一组：结构组
        structure_data = self._extract_structure_group(summaries_text, volumes_text)
        for key in [
            "book_structure",
            "plot_lines",
            "emotional_arc",
            "climax_point_distribution",
            "symbol_system",
        ]:
            if key in structure_data:
                result[key] = structure_data[key]

        # 构建结构组摘要，供后续组参考
        structure_context = self._build_group_context(structure_data)

        # 第二组：技法组（传入结构组上下文）
        technique_data = self._extract_technique_group(
            summaries_text, volumes_text, structure_context
        )
        for key in [
            "revelation_pacing",
            "chapter_patterns",
            "emotion_transition_patterns",
            "information_management",
            "climax_buildup_chains",
            "conflict_escalation",
        ]:
            if key in technique_data:
                result[key] = technique_data[key]

        # 构建技法组摘要，供类型组参考
        technique_context = self._build_group_context(technique_data)
        combined_context = structure_context + "\n" + technique_context

        # 第三组：类型组（传入结构组+技法组上下文）
        genre_data = self._extract_genre_group(
            summaries_text, volumes_text, combined_context
        )
        for key in [
            "romance_lines",
            "mystery_clues",
            "fear_building",
            "progression_systems",
            "genre_specific_techniques",
            "pov_switching_patterns",
        ]:
            if key in genre_data:
                result[key] = genre_data[key]

        logger.info(
            f"✅ [阶段H战报] 结构: {len(result['book_structure'])} | "
            f"剧情线: {len(result['plot_lines'])} | "
            f"情感曲线: {len(result['emotional_arc'])} | "
            f"高潮点分布: {len(result['climax_point_distribution'])} | "
            f"象征体系: {len(result['symbol_system'])} | "
            f"信息揭露: {len(result['revelation_pacing'])} | "
            f"章节模式: {len(result['chapter_patterns'])} | "
            f"情感铺垫: {len(result['emotion_transition_patterns'])} | "
            f"信息管理: {len(result['information_management'])} | "
            f"高潮构建: {len(result['climax_buildup_chains'])} | "
            f"冲突升级: {len(result['conflict_escalation'])} | "
            f"感情线: {len(result['romance_lines'])} | "
            f"线索推理: {len(result['mystery_clues'])} | "
            f"恐惧构建: {len(result['fear_building'])} | "
            f"升级体系: {len(result['progression_systems'])} | "
            f"类型技法: {len(result['genre_specific_techniques'])} | "
            f"视角切换: {len(result['pov_switching_patterns'])}"
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

            if key == "book_structure" and items:
                struct = items[0]
                context_lines.append(
                    f"- 结构类型: {struct.get('structure_type', '未知')}"
                )
                context_lines.append(f"- 表层主题: {struct.get('surface_theme', '')}")
                context_lines.append(f"- 深层主题: {struct.get('deep_theme', '')}")

            elif key == "plot_lines":
                main_plots = [p for p in items if p.get("line_type") == "main"]
                if main_plots:
                    context_lines.append(
                        f"- 主线主题: {main_plots[0].get('theme', '')}"
                    )
                context_lines.append(
                    f"- 支线数量: {len([p for p in items if p.get('line_type') == 'subplot'])}"
                )

            elif key == "emotional_arc" and items:
                arc_data = items[0].get("arc_data", [])
                if arc_data:
                    emotions = [a.get("dominant_emotion", "") for a in arc_data[:3]]
                    context_lines.append(f"- 主要情感阶段: {', '.join(emotions)}")

            elif key == "climax_point_distribution" and items:
                dist = items[0]
                climax_points = dist.get("distribution", [])
                context_lines.append(f"- 高潮点数量: {len(climax_points)}")
                context_lines.append(f"- 节奏模式: {dist.get('rhythm_pattern', '')}")

            elif key == "symbol_system" and items:
                symbols = items[0].get("symbols", [])
                symbol_names = [s.get("symbol", "") for s in symbols[:3]]
                context_lines.append(f"- 核心象征: {', '.join(symbol_names)}")

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

    def _extract_structure_group(
        self, summaries_text: str, volumes_text: str
    ) -> Dict[str, List[Dict]]:
        """提取结构组：book_structure, plot_lines, emotional_arc, climax_point_distribution, symbol_system"""
        result = {}
        prompt = f"""你是顶级的文学评论家。请根据《{self.book_name}》({self.category})的全书摘要和卷大纲，分析全书结构。

【书名】{self.book_name} 【分类】{self.category}
【全书章节摘要】
{summaries_text}

【卷大纲】
{volumes_text}

请输出纯 JSON 格式：
{{
  "book_structure": {{
    "structure_type": "结构类型(三幕结构/英雄之旅/多线交织/环形结构/非线性叙事)",
    "surface_theme": "表层主题(20字内)",
    "deep_theme": "深层主题(30字内)",
    "act_breakdown": [
      {{
        "act_name": "幕名",
        "chapter_range": "章节范围",
        "key_function": "核心功能",
        "turning_point": "转折点"
      }}
    ]
  }},
  "plot_lines": {{
    "main_plot": {{
      "theme": "主线主题",
      "chapter_distribution": "章节分布",
      "key_milestones": ["里程碑1", "里程碑2"]
    }},
    "subplots": [
      {{
        "name": "支线名称",
        "theme": "支线主题",
        "chapter_distribution": "章节分布",
        "key_milestones": ["里程碑1"]
      }}
    ]
  }},
  "emotional_arc": [
    {{
      "chapter_range": "章节范围",
      "dominant_emotion": "主导情绪",
      "emotional_intensity": 5,
      "key_events": ["关键事件1"]
    }}
  ],
  "climax_point_distribution": {{
    "climax_points": [
      {{
        "chapter": 30,
        "type": "高潮/张力点类型(打脸/升级/获宝/逆袭/揭秘/复仇/情感爆发/悬念释放)",
        "description": "高潮点描述(如:主角击败曾经看不起他的人)",
        "intensity": 7
      }}
    ],
    "rhythm_pattern": "节奏模式"
  }},
  "symbol_system": [
    {{
      "symbol": "核心象征",
      "meaning": "象征意义",
      "occurrences": ["出现位置1"],
      "thematic_significance": "主题意义(50字内)"
    }}
  ]
}}
(⚠️核心要求：必须识别结构类型和幕次划分！必须分离主线支线！必须绘制情感曲线！必须识别高潮/张力点分布！必须提取象征体系！禁止反引号)"""

        try:
            resp = ollama_chat(prompt, 0.2, "H")
            data = safe_parse_json(resp)
            if data:
                structure = data.get("book_structure", {})
                if structure:
                    result["book_structure"] = [
                        {
                            "book_name": self.book_name,
                            "structure_type": structure.get("structure_type", "未知"),
                            "surface_theme": structure.get("surface_theme", ""),
                            "deep_theme": structure.get("deep_theme", ""),
                            "act_breakdown": structure.get("act_breakdown", []),
                        }
                    ]

                plot_lines = data.get("plot_lines", {})
                if plot_lines:
                    result["plot_lines"] = []
                    main_plot = plot_lines.get("main_plot", {})
                    if main_plot:
                        result["plot_lines"].append(
                            {
                                "book_name": self.book_name,
                                "line_type": "main",
                                "theme": main_plot.get("theme", ""),
                                "chapter_distribution": main_plot.get(
                                    "chapter_distribution", ""
                                ),
                                "milestones": main_plot.get("key_milestones", []),
                            }
                        )
                    for subplot in plot_lines.get("subplots", []):
                        result["plot_lines"].append(
                            {
                                "book_name": self.book_name,
                                "line_type": "subplot",
                                "theme": subplot.get("theme", ""),
                                "chapter_distribution": subplot.get(
                                    "chapter_distribution", ""
                                ),
                                "milestones": subplot.get("key_milestones", []),
                                "name": subplot.get("name", ""),
                            }
                        )

                emotional_arc = data.get("emotional_arc", [])
                if emotional_arc:
                    result["emotional_arc"] = [
                        {"book_name": self.book_name, "arc_data": emotional_arc}
                    ]

                climax_points = data.get("climax_point_distribution", {})
                if climax_points:
                    result["climax_point_distribution"] = [
                        {
                            "book_name": self.book_name,
                            "distribution": climax_points.get("climax_points", []),
                            "rhythm_pattern": climax_points.get("rhythm_pattern", ""),
                        }
                    ]

                symbols = data.get("symbol_system", [])
                if symbols:
                    result["symbol_system"] = [
                        {"book_name": self.book_name, "symbols": symbols}
                    ]
        except Exception as e:
            logger.warning(f"⚠️ [阶段H] 结构组提取失败: {e}")

        return result

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
  "chapter_patterns": {{
    "opening_patterns": ["常见开头模式1"],
    "ending_patterns": ["常见结尾模式1"],
    "common_transitions": ["常见转场手法1"]
  }},
  "emotion_transition_patterns": [
    {{
      "transition_type": "情感转变类型",
      "foreshadowing_method": "铺垫方式(50字内)",
      "original_example": "原文示例(100-200字)"
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

                chapter_patterns = data.get("chapter_patterns", {})
                if chapter_patterns:
                    result["chapter_patterns"] = [
                        {
                            "book_name": self.book_name,
                            "opening_patterns": chapter_patterns.get(
                                "opening_patterns", []
                            ),
                            "ending_patterns": chapter_patterns.get(
                                "ending_patterns", []
                            ),
                            "common_transitions": chapter_patterns.get(
                                "common_transitions", []
                            ),
                        }
                    ]

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

    def _extract_genre_group(
        self, summaries_text: str, volumes_text: str, prior_context: str = ""
    ) -> Dict[str, List[Dict]]:
        """提取类型组：romance_lines, mystery_clues, fear_building, progression_systems, genre_specific_techniques, pov_switching_patterns"""
        result = {}

        # 构建包含前序分析结果的 Prompt
        prior_section = ""
        if prior_context:
            prior_section = f"\n{prior_context}\n请参考以上结构和技法分析结果，确保类型元素分析与前序分析保持一致。\n"

        prompt = f"""你是顶级的文学评论家。请根据《{self.book_name}》({self.category})的全书摘要和卷大纲，分析类型特定元素。

【书名】{self.book_name} 【分类】{self.category}
{prior_section}
【全书章节摘要】
{summaries_text}

【卷大纲】
{volumes_text}

请输出纯 JSON 格式：
{{
  "romance_lines": [
    {{
      "couple_a": "CP角色A姓名",
      "couple_b": "CP角色B姓名",
      "line_type": "感情线类型",
      "development_stages": ["阶段1"],
      "sweet_points": ["甜点场景1"],
      "angst_points": ["虐点场景1"],
      "interaction_patterns": ["互动模式1"],
      "resolution": "感情线结局(50字内)"
    }}
  ],
  "mystery_clues": [
    {{
      "clue_name": "线索名称",
      "clue_type": "线索类型",
      "planted_chapter": "埋设章节",
      "payoff_chapter": "回收章节",
      "red_herring": 0,
      "misdirection_method": "误导手法(50字内)",
      "reasoning_chain": ["推理步骤1"],
      "twist_design": "反转设计(50字内)"
    }}
  ],
  "fear_building": [
    {{
      "fear_type": "恐惧类型",
      "building_steps": ["步骤1"],
      "atmosphere_techniques": ["技法1"],
      "climax_moment": "恐惧高潮时刻(50字内)",
      "original_example": "原文氛围营造示例(100-200字)"
    }}
  ],
  "progression_systems": [
    {{
      "system_type": "体系类型",
      "levels": ["等级1"],
      "upgrade_conditions": ["升级条件1"],
      "power_comparison": ["实力对比1"],
      "milestones": ["成长里程碑1"],
      "growth_pattern": "成长模式(50字内)"
    }}
  ],
  "genre_specific_techniques": [
    {{
      "genre_tag": "类型标签",
      "technique_name": "技法名称",
      "technique_category": "技法分类",
      "analysis": "技法分析(100字内)",
      "original_example": "原文示例(100-200字)",
      "applicable_scenarios": "适用场景(50字内)"
    }}
  ],
  "pov_switching_patterns": [
    {{
      "pattern_type": "视角模式类型",
      "pov_characters": ["视角角色1"],
      "switching_triggers": "切换触发条件(50字内)",
      "frequency": "切换频率(50字内)",
      "original_example": "视角切换示例(100-200字)"
    }}
  ]
}}
(⚠️核心要求：如果有感情线必须提取！如果是悬疑类必须提取线索！如果是恐怖类必须提取恐惧构建！如果有升级体系必须提取！必须提取类型特定技法！必须分析视角切换！没有的元素返回空数组！禁止反引号)"""

        try:
            resp = ollama_chat(prompt, 0.2, "H")
            data = safe_parse_json(resp)
            if data:
                romance = data.get("romance_lines", [])
                if romance:
                    result["romance_lines"] = []
                    for rl in romance:
                        if isinstance(rl, dict) and rl.get("couple_a"):
                            result["romance_lines"].append(
                                {
                                    "book_name": self.book_name,
                                    "couple_a": rl.get("couple_a"),
                                    "couple_b": rl.get("couple_b", ""),
                                    "line_type": rl.get("line_type", ""),
                                    "development_stages": rl.get(
                                        "development_stages", []
                                    ),
                                    "sweet_points": rl.get("sweet_points", []),
                                    "angst_points": rl.get("angst_points", []),
                                    "interaction_patterns": rl.get(
                                        "interaction_patterns", []
                                    ),
                                    "resolution": rl.get("resolution", ""),
                                }
                            )

                mystery = data.get("mystery_clues", [])
                if mystery:
                    result["mystery_clues"] = []
                    for mc in mystery:
                        if isinstance(mc, dict) and mc.get("clue_name"):
                            result["mystery_clues"].append(
                                {
                                    "book_name": self.book_name,
                                    "clue_name": mc.get("clue_name"),
                                    "clue_type": mc.get("clue_type", ""),
                                    "planted_chapter": mc.get("planted_chapter", ""),
                                    "payoff_chapter": mc.get("payoff_chapter", ""),
                                    "red_herring": mc.get("red_herring", 0),
                                    "misdirection_method": mc.get(
                                        "misdirection_method", ""
                                    ),
                                    "reasoning_chain": mc.get("reasoning_chain", []),
                                    "twist_design": mc.get("twist_design", ""),
                                }
                            )

                fear = data.get("fear_building", [])
                if fear:
                    result["fear_building"] = []
                    for fb in fear:
                        if isinstance(fb, dict) and fb.get("fear_type"):
                            result["fear_building"].append(
                                {
                                    "book_name": self.book_name,
                                    "fear_type": fb.get("fear_type"),
                                    "building_steps": fb.get("building_steps", []),
                                    "atmosphere_techniques": fb.get(
                                        "atmosphere_techniques", []
                                    ),
                                    "climax_moment": fb.get("climax_moment", ""),
                                    "original_example": fb.get("original_example", ""),
                                }
                            )

                progression = data.get("progression_systems", [])
                if progression:
                    result["progression_systems"] = []
                    for ps in progression:
                        if isinstance(ps, dict) and ps.get("system_type"):
                            result["progression_systems"].append(
                                {
                                    "book_name": self.book_name,
                                    "system_type": ps.get("system_type"),
                                    "levels": ps.get("levels", []),
                                    "upgrade_conditions": ps.get(
                                        "upgrade_conditions", []
                                    ),
                                    "power_comparison": ps.get("power_comparison", []),
                                    "milestones": ps.get("milestones", []),
                                    "growth_pattern": ps.get("growth_pattern", ""),
                                }
                            )

                genre_tech = data.get("genre_specific_techniques", [])
                if genre_tech:
                    result["genre_specific_techniques"] = []
                    for gt in genre_tech:
                        if isinstance(gt, dict) and gt.get("technique_name"):
                            result["genre_specific_techniques"].append(
                                {
                                    "book_name": self.book_name,
                                    "genre_tag": gt.get("genre_tag", self.category),
                                    "technique_name": gt.get("technique_name"),
                                    "technique_category": gt.get(
                                        "technique_category", ""
                                    ),
                                    "analysis": gt.get("analysis", ""),
                                    "original_example": gt.get("original_example", ""),
                                    "applicable_scenarios": gt.get(
                                        "applicable_scenarios", ""
                                    ),
                                }
                            )

                pov_patterns = data.get("pov_switching_patterns", [])
                if pov_patterns:
                    result["pov_switching_patterns"] = []
                    for pp in pov_patterns:
                        if isinstance(pp, dict) and pp.get("pattern_type"):
                            result["pov_switching_patterns"].append(
                                {
                                    "book_name": self.book_name,
                                    "pattern_type": pp.get("pattern_type"),
                                    "pov_characters": pp.get("pov_characters", []),
                                    "switching_triggers": pp.get(
                                        "switching_triggers", ""
                                    ),
                                    "frequency": pp.get("frequency", ""),
                                    "original_example": pp.get("original_example", ""),
                                }
                            )
        except Exception as e:
            logger.warning(f"⚠️ [阶段H] 类型组提取失败: {e}")

        return result

    def insert(self, results: Dict[str, List[Dict]]) -> Dict[str, int]:
        """将 Stage H 结果写入数据库"""
        cursor = self.db.connect().cursor()
        stats = {
            "book_structure": 0,
            "plot_lines": 0,
            "emotional_arc": 0,
            "climax_point_distribution": 0,
            "symbol_system": 0,
            "revelation_pacing": 0,
            "chapter_patterns": 0,
            "emotion_transition_patterns": 0,
            "information_management": 0,
            "climax_buildup_chains": 0,
            "conflict_escalation": 0,
            "romance_lines": 0,
            "mystery_clues": 0,
            "fear_building": 0,
            "progression_systems": 0,
            "genre_specific_techniques": 0,
            "pov_switching_patterns": 0,
        }

        # 全书结构入库（6个字段）
        for bs in results.get("book_structure", []):
            bs_id = generate_id(bs["book_name"], "structure")
            cursor.execute(
                "INSERT OR REPLACE INTO book_structure VALUES (?,?,?,?,?,?)",
                (
                    bs_id,
                    bs["book_name"],
                    bs.get("structure_type", ""),
                    json.dumps(bs.get("act_breakdown", []), ensure_ascii=False),
                    bs.get("surface_theme", ""),
                    bs.get("deep_theme", ""),
                ),
            )
            stats["book_structure"] += 1

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

        # 情感曲线入库
        for ea in results.get("emotional_arc", []):
            ea_id = generate_id(ea["book_name"], "emotional_arc")
            cursor.execute(
                "INSERT OR REPLACE INTO emotional_arc VALUES (?,?,?)",
                (
                    ea_id,
                    ea["book_name"],
                    json.dumps(ea.get("arc_data", []), ensure_ascii=False),
                ),
            )
            stats["emotional_arc"] += 1

        # 高潮点分布入库
        for cpd in results.get("climax_point_distribution", []):
            cpd_id = generate_id(cpd["book_name"], "climax_points")
            cursor.execute(
                "INSERT OR REPLACE INTO climax_point_distribution VALUES (?,?,?,?)",
                (
                    cpd_id,
                    cpd["book_name"],
                    json.dumps(cpd.get("distribution", []), ensure_ascii=False),
                    cpd.get("rhythm_pattern", ""),
                ),
            )
            stats["climax_point_distribution"] += 1

        # 象征体系入库
        for ss in results.get("symbol_system", []):
            ss_id = generate_id(ss["book_name"], "symbols")
            cursor.execute(
                "INSERT OR REPLACE INTO symbol_system VALUES (?,?,?)",
                (
                    ss_id,
                    ss["book_name"],
                    json.dumps(ss.get("symbols", []), ensure_ascii=False),
                ),
            )
            stats["symbol_system"] += 1

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

        # 章节模式入库
        for cp in results.get("chapter_patterns", []):
            cp_id = generate_id(cp["book_name"], "patterns")
            cursor.execute(
                "INSERT OR REPLACE INTO chapter_patterns VALUES (?,?,?,?,?)",
                (
                    cp_id,
                    cp["book_name"],
                    json.dumps(cp.get("opening_patterns", []), ensure_ascii=False),
                    json.dumps(cp.get("ending_patterns", []), ensure_ascii=False),
                    json.dumps(cp.get("common_transitions", []), ensure_ascii=False),
                ),
            )
            stats["chapter_patterns"] += 1

        # 情感转变铺垫入库
        for et in results.get("emotion_transition_patterns", []):
            et_id = generate_id(et["book_name"], et["transition_type"])
            cursor.execute(
                "INSERT OR REPLACE INTO emotion_transition_patterns VALUES (?,?,?,?,?)",
                (
                    et_id,
                    et["book_name"],
                    et["transition_type"],
                    et.get("foreshadowing_method", ""),
                    et.get("original_example", ""),
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

        # 感情线追踪入库
        for rl in results.get("romance_lines", []):
            rl_id = generate_id(rl["book_name"], rl["couple_a"], rl.get("couple_b", ""))
            cursor.execute(
                "INSERT OR REPLACE INTO romance_lines VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    rl_id,
                    rl["book_name"],
                    rl["couple_a"],
                    rl.get("couple_b", ""),
                    rl.get("line_type", ""),
                    json.dumps(rl.get("development_stages", []), ensure_ascii=False),
                    json.dumps(rl.get("sweet_points", []), ensure_ascii=False),
                    json.dumps(rl.get("angst_points", []), ensure_ascii=False),
                    json.dumps(rl.get("interaction_patterns", []), ensure_ascii=False),
                    rl.get("resolution", ""),
                ),
            )
            stats["romance_lines"] += 1

        # 线索与推理链入库
        for mc in results.get("mystery_clues", []):
            mc_id = generate_id(mc["book_name"], mc["clue_name"])
            cursor.execute(
                "INSERT OR REPLACE INTO mystery_clues VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    mc_id,
                    mc["book_name"],
                    mc["clue_name"],
                    mc.get("clue_type", ""),
                    mc.get("planted_chapter", ""),
                    mc.get("payoff_chapter", ""),
                    mc.get("red_herring", 0),
                    mc.get("misdirection_method", ""),
                    json.dumps(mc.get("reasoning_chain", []), ensure_ascii=False),
                    mc.get("twist_design", ""),
                ),
            )
            stats["mystery_clues"] += 1

        # 恐惧/氛围构建链入库
        for fb in results.get("fear_building", []):
            fb_id = generate_id(fb["book_name"], fb["fear_type"])
            cursor.execute(
                "INSERT OR REPLACE INTO fear_building VALUES (?,?,?,?,?,?,?)",
                (
                    fb_id,
                    fb["book_name"],
                    fb["fear_type"],
                    json.dumps(fb.get("building_steps", []), ensure_ascii=False),
                    json.dumps(fb.get("atmosphere_techniques", []), ensure_ascii=False),
                    fb.get("climax_moment", ""),
                    fb.get("original_example", ""),
                ),
            )
            stats["fear_building"] += 1

        # 升级/成长体系入库
        for ps in results.get("progression_systems", []):
            ps_id = generate_id(ps["book_name"], ps["system_type"])
            cursor.execute(
                "INSERT OR REPLACE INTO progression_systems VALUES (?,?,?,?,?,?,?,?)",
                (
                    ps_id,
                    ps["book_name"],
                    ps["system_type"],
                    json.dumps(ps.get("levels", []), ensure_ascii=False),
                    json.dumps(ps.get("upgrade_conditions", []), ensure_ascii=False),
                    json.dumps(ps.get("power_comparison", []), ensure_ascii=False),
                    json.dumps(ps.get("milestones", []), ensure_ascii=False),
                    ps.get("growth_pattern", ""),
                ),
            )
            stats["progression_systems"] += 1

        # 类型特定技法入库
        for gt in results.get("genre_specific_techniques", []):
            gt_id = generate_id(gt["book_name"], gt["technique_name"])
            cursor.execute(
                "INSERT OR REPLACE INTO genre_specific_techniques VALUES (?,?,?,?,?,?,?,?)",
                (
                    gt_id,
                    gt["book_name"],
                    gt.get("genre_tag", ""),
                    gt["technique_name"],
                    gt.get("technique_category", ""),
                    gt.get("analysis", ""),
                    gt.get("original_example", ""),
                    gt.get("applicable_scenarios", ""),
                ),
            )
            stats["genre_specific_techniques"] += 1

        # 多视角切换模式入库
        for pp in results.get("pov_switching_patterns", []):
            pp_id = generate_id(pp["book_name"], pp["pattern_type"])
            cursor.execute(
                "INSERT OR REPLACE INTO pov_switching_patterns VALUES (?,?,?,?,?,?,?)",
                (
                    pp_id,
                    pp["book_name"],
                    pp["pattern_type"],
                    json.dumps(pp.get("pov_characters", []), ensure_ascii=False),
                    pp.get("switching_triggers", ""),
                    pp.get("frequency", ""),
                    pp.get("original_example", ""),
                ),
            )
            stats["pov_switching_patterns"] += 1

        self.db.commit()
        logger.info(
            f"   ✅ [阶段H战报] 结构: {stats['book_structure']} | "
            f"剧情线: {stats['plot_lines']} | 情感曲线: {stats['emotional_arc']} | "
            f"高潮点分布: {stats['climax_point_distribution']} | 象征体系: {stats['symbol_system']} | "
            f"信息揭露: {stats['revelation_pacing']} | 章节模式: {stats['chapter_patterns']} | "
            f"情感铺垫: {stats['emotion_transition_patterns']} | "
            f"信息管理: {stats['information_management']} | "
            f"高潮构建: {stats['climax_buildup_chains']} | "
            f"冲突升级: {stats['conflict_escalation']} | "
            f"感情线: {stats['romance_lines']} | "
            f"线索推理: {stats['mystery_clues']} | "
            f"恐惧构建: {stats['fear_building']} | "
            f"升级体系: {stats['progression_systems']} | "
            f"类型技法: {stats['genre_specific_techniques']} | "
            f"视角切换: {stats['pov_switching_patterns']}"
        )
        return stats
