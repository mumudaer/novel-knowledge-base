"""
Stage L: 跨书对比分析模块
从知识库中已有的多本书数据中提取共同模式和差异
让知识库能回答"多本标杆作品在同一个问题上是怎么处理的，有什么异同"
"""
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from stages.base import BaseStage
from core.ollama_client import ollama_chat, safe_parse_json
from core.utils import generate_id

logger = logging.getLogger(__name__)


class StageL(BaseStage):
    """Stage L: 跨书对比分析"""

    def __init__(self, book_name: str = "cross_book", category: str = "fiction"):
        super().__init__("L", book_name, category)

    def check_dimension_data(self, dimension: str) -> int:
        """
        检查指定维度的数据量
        
        Args:
            dimension: 对比维度名称
            
        Returns:
            该维度相关的数据条数
        """
        from core.db import get_db_manager
        
        db = get_db_manager()
        cursor = db.connect().cursor()
        
        # 根据维度映射到对应的表
        dimension_table_map = {
            "感情线设计": ("romance_lines", "couple_a"),
            "高潮铺垫方式": ("climax_buildup_chains", "climax_name"),
            "冲突升级模式": ("conflict_escalation", "conflict_line"),
            "人物塑造": ("character_profiles", "name"),
            "世界观设计": ("world_settings", "module"),
            "对话风格": ("character_speech_style", "character_name"),
            "描写技法": ("description_samples", "description_type"),
            "结构编排": ("book_structure", "surface_theme"),
            "信息管理": ("information_management", "strategy_type"),
            "伏笔设计": ("plot_foreshadowing", "hook_name"),
        }
        
        table_info = dimension_table_map.get(dimension)
        if not table_info:
            return 0
        
        table_name, key_field = table_info
        
        try:
            cursor.execute(f"SELECT COUNT(DISTINCT book_name) FROM {table_name}")
            result = cursor.fetchone()
            return result[0] if result else 0
        except Exception:
            return 0

    def run(
        self,
        comparison_dimension: str,
        book_names: Optional[List[str]] = None,
        category: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        执行跨书对比分析

        Args:
            comparison_dimension: 对比维度（如"感情线设计"、"高潮铺垫方式"、"冲突升级模式"）
            book_names: 可选，指定对比哪些书（默认选择该维度数据最丰富的 3-5 本）
            category: 可选，按分类筛选书籍

        Returns:
            对比分析结果字典
        """
        logger.info(f"=== 阶段L：跨书对比分析 (维度: {comparison_dimension}) ===")

        # 1. 确定要对比的书籍
        if not book_names:
            book_names = self._select_books_for_comparison(comparison_dimension, category)
        
        if len(book_names) < 2:
            logger.warning(f"⚠️ [阶段L] 可用书籍不足 2 本，无法进行对比分析")
            return {"comparison_dimension": comparison_dimension, "books_analyzed": [], "analysis": {}}

        # 2. 收集各书在该维度的数据
        dimension_data = self._collect_dimension_data(comparison_dimension, book_names)
        
        if not dimension_data:
            logger.warning(f"⚠️ [阶段L] 未收集到任何数据")
            return {"comparison_dimension": comparison_dimension, "books_analyzed": book_names, "analysis": {}}

        # 3. 调用 LLM 进行对比分析
        prompt = self._build_comparison_prompt(comparison_dimension, dimension_data)

        try:
            resp = ollama_chat(prompt, 0.3, "L")
            data = safe_parse_json(resp)
            if not data:
                logger.warning("⚠️ [阶段L] LLM 返回解析失败")
                return {"comparison_dimension": comparison_dimension, "books_analyzed": book_names, "analysis": {}}

            result = {
                "comparison_dimension": comparison_dimension,
                "books_analyzed": book_names,
                "analysis": data,
            }
            
            logger.info(
                f"✅ [阶段L战报] 对比书籍: {len(book_names)} | "
                f"共同模式: {len(data.get('common_patterns', []))} | "
                f"独特特色: {len(data.get('unique_features', []))}"
            )
            return result

        except Exception as exc:
            logger.error(f"❌ [阶段L] 对比分析失败: {exc}")
            return {"comparison_dimension": comparison_dimension, "books_analyzed": book_names, "analysis": {}}

    def insert(self, results: Dict[str, Any]) -> Dict[str, int]:
        """将对比分析结果写入数据库"""
        cursor = self.db.connect().cursor()
        stats = {"cross_book_comparisons": 0}

        analysis = results.get("analysis", {})
        if not analysis:
            return stats

        # ID 不含时间戳，保证幂等性：重复运行相同维度会覆盖旧记录
        comparison_id = generate_id(
            results["comparison_dimension"],
            "_".join(results["books_analyzed"]),
        )

        cursor.execute(
            "INSERT OR REPLACE INTO cross_book_comparisons VALUES (?,?,?,?,?,?,?)",
            (
                comparison_id,
                results["comparison_dimension"],
                json.dumps(results["books_analyzed"], ensure_ascii=False),
                json.dumps(analysis.get("common_patterns", []), ensure_ascii=False),
                json.dumps(analysis.get("unique_features", []), ensure_ascii=False),
                analysis.get("best_practices", ""),
                datetime.now().isoformat(),
            ),
        )
        stats["cross_book_comparisons"] += 1

        self.db.commit()
        logger.info(f"   ✅ [阶段L] 对比分析已写入")
        return stats

    def _select_books_for_comparison(
        self,
        comparison_dimension: str,
        category: Optional[str] = None,
    ) -> List[str]:
        """选择用于对比的书籍（优先选择该维度数据最丰富的）"""
        cursor = self.db.connect().cursor()
        
        # 根据维度映射到对应的数据表
        dimension_to_tables = {
            "感情线设计": ["romance_lines"],
            "高潮铺垫方式": ["climax_buildup_chains", "climax_excerpts"],
            "冲突升级模式": ["conflict_escalation"],
            "人物塑造": ["character_profiles"],
            "世界观设计": ["world_settings"],
            "对话风格": ["dialogue_samples", "character_speech_style"],
            "描写技法": ["description_samples"],
            "结构编排": ["book_structure", "macro_outlines"],
            "信息管理": ["information_management"],
            "伏笔设计": ["plot_foreshadowing"],
        }
        
        tables = dimension_to_tables.get(comparison_dimension, ["book_metadata"])
        
        # 统计各书在这些表中的数据量
        book_scores = {}
        for table in tables:
            try:
                sql = f"SELECT book_name, COUNT(*) as cnt FROM {table}"
                if category:
                    sql += " WHERE book_name IN (SELECT book_name FROM book_metadata WHERE category LIKE ?)"
                    cursor.execute(sql + " GROUP BY book_name", (f"%{category}%",))
                else:
                    cursor.execute(sql + " GROUP BY book_name")
                
                for row in cursor.fetchall():
                    book_name, count = row
                    book_scores[book_name] = book_scores.get(book_name, 0) + count
            except Exception:
                continue
        
        # 按数据量排序，选择前 3-5 本
        sorted_books = sorted(book_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [book for book, count in sorted_books[:5] if count > 0]
        
        return selected

    def _collect_dimension_data(
        self,
        comparison_dimension: str,
        book_names: List[str],
    ) -> Dict[str, str]:
        """收集各书在指定维度的数据摘要"""
        cursor = self.db.connect().cursor()
        dimension_data = {}
        
        for book_name in book_names:
            book_data_parts = []
            
            # 根据维度收集相关数据
            if comparison_dimension == "感情线设计":
                cursor.execute(
                    "SELECT couple_a, couple_b, line_type, development_stages_json, sweet_points_json, angst_points_json FROM romance_lines WHERE book_name=? LIMIT 5",
                    (book_name,),
                )
                for row in cursor.fetchall():
                    stages = json.loads(row[3]) if row[3] else []
                    book_data_parts.append(
                        f"CP: {row[0]} & {row[1]} | 类型: {row[2]} | "
                        f"发展阶段: {len(stages)}个 | "
                        f"甜点: {json.loads(row[4]) if row[4] else []} | "
                        f"虐点: {json.loads(row[5]) if row[5] else []}"
                    )
            
            elif comparison_dimension == "高潮铺垫方式":
                cursor.execute(
                    "SELECT climax_name, climax_chapter, buildup_steps_json, tension_escalation FROM climax_buildup_chains WHERE book_name=? LIMIT 5",
                    (book_name,),
                )
                for row in cursor.fetchall():
                    steps = json.loads(row[2]) if row[2] else []
                    book_data_parts.append(
                        f"高潮: {row[0]} (第{row[1]}章) | "
                        f"铺垫步骤: {len(steps)}步 | "
                        f"张力升级: {row[3]}"
                    )
            
            elif comparison_dimension == "冲突升级模式":
                cursor.execute(
                    "SELECT conflict_line, escalation_steps_json, escalation_pattern FROM conflict_escalation WHERE book_name=? LIMIT 5",
                    (book_name,),
                )
                for row in cursor.fetchall():
                    steps = json.loads(row[1]) if row[1] else []
                    book_data_parts.append(
                        f"冲突线: {row[0]} | "
                        f"升级步骤: {len(steps)}步 | "
                        f"升级模式: {row[2]}"
                    )
            
            elif comparison_dimension == "人物塑造":
                cursor.execute(
                    "SELECT name, role_type, desire_vs_need, fatal_flaw, arc_trajectory FROM character_profiles WHERE book_name=? LIMIT 5",
                    (book_name,),
                )
                for row in cursor.fetchall():
                    book_data_parts.append(
                        f"角色: {row[0]} ({row[1]}) | "
                        f"欲望vs需求: {row[2]} | "
                        f"致命缺陷: {row[3]} | "
                        f"弧光: {row[4]}"
                    )
            
            elif comparison_dimension == "世界观设计":
                cursor.execute(
                    "SELECT module, entity, content FROM world_settings WHERE book_name=? LIMIT 10",
                    (book_name,),
                )
                for row in cursor.fetchall():
                    book_data_parts.append(f"{row[0]}:{row[1]} - {row[2][:100]}")
            
            elif comparison_dimension == "对话风格":
                cursor.execute(
                    "SELECT character_name, catchphrases, vocabulary_preference, sentence_pattern FROM character_speech_style WHERE book_name=? LIMIT 5",
                    (book_name,),
                )
                for row in cursor.fetchall():
                    book_data_parts.append(
                        f"角色: {row[0]} | "
                        f"口头禅: {row[1]} | "
                        f"词汇偏好: {row[2]} | "
                        f"句式: {row[3]}"
                    )
            
            elif comparison_dimension == "描写技法":
                cursor.execute(
                    "SELECT description_type, technique_analysis, sensory_details FROM description_samples WHERE book_name=? LIMIT 5",
                    (book_name,),
                )
                for row in cursor.fetchall():
                    book_data_parts.append(
                        f"类型: {row[0]} | "
                        f"技法: {row[1]} | "
                        f"感官: {row[2]}"
                    )
            
            elif comparison_dimension == "结构编排":
                cursor.execute(
                    "SELECT act_breakdown_json, surface_theme, deep_theme FROM book_structure WHERE book_name=? LIMIT 1",
                    (book_name,),
                )
                row = cursor.fetchone()
                if row:
                    acts = json.loads(row[1]) if row[1] else []
                    book_data_parts.append(
                        f"结构类型: {row[0]} | "
                        f"幕次: {len(acts)}幕 | "
                        f"表层主题: {row[2]} | "
                        f"深层主题: {row[3]}"
                    )
            
            elif comparison_dimension == "信息管理":
                cursor.execute(
                    "SELECT strategy_type, target_info, conceal_method, reveal_timing, dramatic_purpose FROM information_management WHERE book_name=? LIMIT 5",
                    (book_name,),
                )
                for row in cursor.fetchall():
                    book_data_parts.append(
                        f"策略: {row[0]} | "
                        f"目标信息: {row[1]} | "
                        f"隐瞒方式: {row[2]} | "
                        f"揭露时机: {row[3]} | "
                        f"戏剧目的: {row[4]}"
                    )
            
            elif comparison_dimension == "伏笔设计":
                cursor.execute(
                    "SELECT hook_name, planted_chapter, planned_payoff, status, resolution_excerpt FROM plot_foreshadowing WHERE book_name=? LIMIT 5",
                    (book_name,),
                )
                for row in cursor.fetchall():
                    book_data_parts.append(
                        f"伏笔: {row[0]} | "
                        f"埋设: 第{row[1]}章 | "
                        f"计划回收: {row[2]} | "
                        f"状态: {row[3]} | "
                        f"回收片段: {row[4][:50] if row[4] else ''}"
                    )
            
            else:
                # 通用兜底：收集书籍基本信息
                cursor.execute(
                    "SELECT category, genre_tags, total_chapters, description FROM book_metadata WHERE book_name=? LIMIT 1",
                    (book_name,),
                )
                row = cursor.fetchone()
                if row:
                    book_data_parts.append(
                        f"分类: {row[0]} | "
                        f"标签: {row[1]} | "
                        f"总章数: {row[2]} | "
                        f"简介: {row[3][:100] if row[3] else ''}"
                    )
            
            if book_data_parts:
                dimension_data[book_name] = "\n".join(book_data_parts)
        
        return dimension_data

    def _build_comparison_prompt(
        self,
        comparison_dimension: str,
        dimension_data: Dict[str, str],
    ) -> str:
        """构建跨书对比分析 Prompt"""
        books_text = "\n\n".join([
            f"【《{book_name}》】\n{data}"
            for book_name, data in dimension_data.items()
        ])

        return f"""你是资深的文学评论家和创作顾问。请对比分析以下多本标杆作品在"{comparison_dimension}"这个维度上的处理方式，提炼共同模式、各自特色和最佳实践。

【对比维度】{comparison_dimension}

【各书数据】
{books_text}

请输出纯 JSON 格式：
{{
  "common_patterns": [
    {{
      "pattern_name": "共同模式名称(如:5阶段感情线发展)",
      "description": "模式描述(如:初遇→误会→和解→危机→确认)",
      "frequency": "出现频率(如:3/5本书采用)",
      "effectiveness": "效果分析(如:这种模式能让读者有代入感，情感积累充分)"
    }}
  ],
  "unique_features": [
    {{
      "book_name": "书名",
      "unique_approach": "独特做法(如:采用欢喜冤家模式而非日久生情)",
      "advantage": "优势(如:前期冲突多，节奏快，读者留存率高)",
      "applicable_scenario": "适用场景(如:适合快节奏网文，不适合慢热言情)"
    }}
  ],
  "best_practices": "最佳实践建议(综合各书优点，给出可操作的创作建议，200字内)"
}}
(⚠️核心要求：
1. 共同模式必须是跨书出现的规律，不是单本书的特点！
2. 独特特色必须是该书独有的、值得学习的做法！
3. 最佳实践必须综合各书优点，给出可操作的建议！
4. 分析必须基于提供的数据，不要凭空想象！
5. 禁止使用反引号，必须输出合法JSON)"""
