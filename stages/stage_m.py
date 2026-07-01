"""
Stage M: 常见错误模式提取模块
从 Stage J 的评审历史数据中提炼高频问题
让知识库能告诉 AI "这种写法是常见的错误，应该避免"
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


class StageM(BaseStage):
    """Stage M: 常见错误模式提取"""

    # issue_type 归一化映射：将具体问题类型映射到标杆查找维度
    DIMENSION_NORMALIZE = {
        "节奏拖沓": "节奏", "对话生硬": "对话", "描写不足": "描写",
        "人物失真": "人物", "情节漏洞": "情节", "过度告知": "描写",
        "视角混乱": "情节",
    }

    def __init__(self, book_name: str = "common_mistakes", category: str = "fiction"):
        super().__init__("M", book_name, category)

    def run(
        self,
        min_frequency: int = 2,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        执行常见错误模式提取

        Args:
            min_frequency: 最小出现频率阈值（默认 2 次以上才纳入）

        Returns:
            错误模式分析结果字典
        """
        logger.info(f"=== 阶段M：常见错误模式提取 ===")

        # 1. 聚合所有评审中的问题数据
        issues_data = self._aggregate_review_issues()
        
        if not issues_data:
            logger.warning("⚠️ [阶段M] 未找到评审历史数据，无法提取错误模式")
            return {"mistakes": []}

        # 2. 按维度分类统计
        dimension_issues = self._group_issues_by_dimension(issues_data)

        # 3. 调用 LLM 归纳错误模式
        prompt = self._build_mistake_analysis_prompt(dimension_issues, min_frequency)

        try:
            resp = ollama_chat(prompt, 0.3, "M")
            data = safe_parse_json(resp)
            if not data:
                logger.warning("⚠️ [阶段M] LLM 返回解析失败")
                return {"mistakes": []}

            result = {
                "mistakes": data.get("mistakes", []),
                "total_reviews_analyzed": len(issues_data),
            }
            
            logger.info(
                f"✅ [阶段M战报] 分析评审数: {len(issues_data)} | "
                f"错误模式数: {len(result['mistakes'])}"
            )
            return result

        except Exception as exc:
            logger.error(f"❌ [阶段M] 错误模式提取失败: {exc}")
            return {"mistakes": []}

    def insert(self, results: Dict[str, Any]) -> Dict[str, int]:
        """将错误模式写入数据库"""
        cursor = self.db.connect().cursor()
        stats = {"common_mistakes": 0}

        for mistake in results.get("mistakes", []):
            if not mistake.get("mistake_name"):
                continue

            # ID 不含时间戳，保证幂等性
            mistake_id = generate_id(
                mistake.get("dimension", ""),
                mistake["mistake_name"],
            )

            # 查找标杆范文作为正确示范
            benchmark_example, benchmark_book = self._find_benchmark_example(
                mistake.get("dimension", "")
            )

            cursor.execute(
                "INSERT OR REPLACE INTO common_mistakes VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    mistake_id,
                    mistake.get("dimension", ""),
                    mistake["mistake_name"],
                    mistake.get("typical_manifestation", ""),
                    mistake.get("frequency", 0),
                    mistake.get("correction_direction", ""),
                    benchmark_example,
                    benchmark_book,
                    datetime.now().isoformat(),
                ),
            )
            stats["common_mistakes"] += 1

        self.db.commit()
        logger.info(f"   ✅ [阶段M] {stats['common_mistakes']} 条错误模式已写入")
        return stats

    def _aggregate_review_issues(self) -> List[Dict]:
        """聚合所有评审中的问题数据"""
        cursor = self.db.connect().cursor()
        
        try:
            cursor.execute(
                "SELECT project_name, chapter_index, overall_score, dimension_scores_json, issues_json FROM chapter_reviews ORDER BY reviewed_at DESC LIMIT 100"
            )
            rows = cursor.fetchall()
        except Exception as exc:
            logger.warning(f"⚠️ [阶段M] 查询评审数据失败: {exc}")
            return []

        issues_data = []
        for row in rows:
            project_name, chapter_index, overall_score, scores_json, issues_json = row
            try:
                issues = json.loads(issues_json) if issues_json else []
                scores = json.loads(scores_json) if scores_json else {}
                
                for issue in issues:
                    if isinstance(issue, dict):
                        # stage_j 写入的字段是 issue_type/severity，不是 dimension/type
                        issue_type = issue.get("issue_type", issue.get("type", ""))
                        # 归一化映射：将 issue_type（如“节奏拖沓”）归一化为标杆查找维度（如“节奏”）
                        raw_dimension = issue.get("dimension", issue_type or "未知")
                        normalized_dim = self.DIMENSION_NORMALIZE.get(issue_type, raw_dimension)
                        issues_data.append({
                            "project_name": project_name,
                            "chapter_index": chapter_index,
                            "overall_score": overall_score,
                            "dimension": normalized_dim,
                            "issue_type": issue_type,
                            "description": issue.get("description", issue.get("location", "")),
                            "severity": issue.get("severity", "medium"),
                            "location": issue.get("location", ""),
                        })
            except Exception:
                continue

        return issues_data

    def _group_issues_by_dimension(self, issues_data: List[Dict]) -> Dict[str, List[Dict]]:
        """按维度分组问题数据"""
        dimension_issues = defaultdict(list)
        
        for issue in issues_data:
            dimension = issue.get("dimension", "未知")
            dimension_issues[dimension].append(issue)
        
        return dict(dimension_issues)

    def _build_mistake_analysis_prompt(
        self,
        dimension_issues: Dict[str, List[Dict]],
        min_frequency: int,
    ) -> str:
        """构建错误模式分析 Prompt"""
        issues_text_parts = []
        
        for dimension, issues in dimension_issues.items():
            if len(issues) < min_frequency:
                continue
            
            issues_text_parts.append(f"\n【{dimension}维度】(共 {len(issues)} 个问题)")
            
            # 按问题类型分组统计
            type_counts = defaultdict(int)
            type_examples = defaultdict(list)
            
            for issue in issues:
                issue_type = issue.get("issue_type", "其他")
                type_counts[issue_type] += 1
                if len(type_examples[issue_type]) < 3:
                    type_examples[issue_type].append(issue.get("description", ""))
            
            for issue_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
                examples = type_examples[issue_type]
                issues_text_parts.append(
                    f"  - {issue_type}: 出现 {count} 次\n"
                    f"    典型描述: {'; '.join(examples[:2])}"
                )

        issues_text = "\n".join(issues_text_parts)

        return f"""你是资深的写作教练和文学编辑。请分析以下评审历史数据中的高频问题，归纳为"常见错误模式"，并给出修正方向和标杆示范。

【评审历史数据统计】
{issues_text}

请输出纯 JSON 格式：
{{
  "mistakes": [
    {{
      "dimension": "所属维度(节奏/对话/描写/人物/情节)",
      "mistake_name": "错误模式名称(如:Tell过度/对话无辨识度/伏笔遗忘)",
      "typical_manifestation": "典型表现(如:连续3段以上使用告知式叙述，缺乏行为和感官细节)",
      "frequency": 5,
      "correction_direction": "修正方向(如:在情感转折点使用Show，通过行为和感官细节传达情绪，参考标杆作品的处理方式)"
    }}
  ]
}}
(⚠️核心要求：
1. 错误模式名称必须简洁有力，便于记忆！
2. 典型表现必须具体，能直接识别！
3. 修正方向必须可操作，不能空泛！
4. frequency 必须是实际统计的出现次数！
5. 只归纳出现 {min_frequency} 次以上的错误模式！
6. 禁止使用反引号，必须输出合法JSON)"""

    def _find_benchmark_example(self, dimension: str) -> tuple:
        """从知识库中查找该维度的标杆范文"""
        cursor = self.db.connect().cursor()
        
        # 根据维度映射到对应的范文表
        dimension_to_table = {
            "对话": ("dialogue_samples", "original_text", "book_name"),
            "描写": ("description_samples", "original_text", "book_name"),
            "节奏": ("climax_excerpts", "original_text", "book_name"),
            "人物": ("character_profiles", "speech_samples", "book_name"),
            "情节": ("climax_buildup_chains", "buildup_steps_json", "book_name"),
        }
        
        table_info = dimension_to_table.get(dimension)
        if not table_info:
            return ("", "")
        
        table, text_col, book_col = table_info
        
        try:
            cursor.execute(
                f"SELECT {text_col}, {book_col} FROM {table} LIMIT 1"
            )
            row = cursor.fetchone()
            if row:
                return (row[0][:200] if row[0] else "", row[1] if row[1] else "")
        except Exception:
            pass
        
        return ("", "")
