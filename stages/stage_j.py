"""
Stage J: 正文质量评审模块
对标知识库标杆作品，对用户小说章节进行多维度评审
输出：打分 + 问题标记 + 修改建议 + 改写示范
"""
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from stages.base import BaseStage
from core.ollama_client import ollama_chat, safe_parse_json
from core.utils import generate_id

logger = logging.getLogger(__name__)


class StageJ(BaseStage):
    """Stage J: 正文质量评审（对标知识库标杆）"""

    def __init__(self, book_name: str = "creative", category: str = "fiction"):
        super().__init__("J", book_name, category)

    def run(
        self,
        chapter_text: str,
        project_name: str,
        chapter_index: int,
        benchmark_books: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        执行正文质量评审

        Args:
            chapter_text: 章节正文
            project_name: 创作项目名称
            chapter_index: 章节序号
            benchmark_books: 标杆作品列表（书名），为空则自动匹配

        Returns:
            评审结果字典
        """
        logger.info(f"=== 阶段J：正文质量评审 (项目: {project_name}, 第{chapter_index}章) ===")

        # 1. 从知识库检索标杆数据
        benchmark_data = self._retrieve_benchmark_data(benchmark_books)

        # 2. 构建评审 Prompt
        prompt = self._build_review_prompt(chapter_text, project_name, chapter_index, benchmark_data)

        # 3. 调用 LLM 进行评审
        try:
            resp = ollama_chat(prompt, 0.2, "J")
            data = safe_parse_json(resp)
            if not data:
                logger.warning("⚠️ [阶段J] LLM 返回解析失败")
                return self._empty_result(project_name, chapter_index)

            result = self._parse_review_result(data, project_name, chapter_index, benchmark_books)
            logger.info(
                f"✅ [阶段J战报] 总分: {result.get('overall_score', 'N/A')} | "
                f"问题数: {len(result.get('issues', []))} | "
                f"建议数: {len(result.get('suggestions', []))} | "
                f"改写示范: {len(result.get('rewrite_samples', []))}"
            )
            return result

        except Exception as exc:
            logger.error(f"❌ [阶段J] 评审失败: {exc}")
            return self._empty_result(project_name, chapter_index)

    def insert(self, results: Dict[str, Any]) -> Dict[str, int]:
        """将评审结果写入数据库"""
        cursor = self.db.connect().cursor()
        stats = {"chapter_reviews": 0}

        review_id = generate_id(
            results["project_name"],
            str(results["chapter_index"]),
            "review",
        )
        cursor.execute(
            "INSERT OR REPLACE INTO chapter_reviews VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                review_id,
                results["project_name"],
                results["chapter_index"],
                results.get("overall_score", 0.0),
                json.dumps(results.get("dimension_scores", {}), ensure_ascii=False),
                json.dumps(results.get("issues", []), ensure_ascii=False),
                json.dumps(results.get("suggestions", []), ensure_ascii=False),
                json.dumps(results.get("rewrite_samples", []), ensure_ascii=False),
                results.get("benchmark_books", ""),
                results.get("reviewed_at", datetime.now().isoformat()),
            ),
        )
        stats["chapter_reviews"] += 1

        self.db.commit()
        logger.info(f"   ✅ [阶段J] 评审结果已写入: {results['project_name']} 第{results['chapter_index']}章")
        return stats

    def _retrieve_benchmark_data(self, benchmark_books: Optional[List[str]] = None) -> str:
        """从知识库检索标杆作品的相关数据"""
        if not benchmark_books:
            return ""

        benchmark_text_parts = []
        cursor = self.db.connect().cursor()

        for book_name in benchmark_books:
            book_data = []

            # 检索对话样本
            cursor.execute(
                "SELECT scene_type, original_text, emotional_tension FROM dialogue_samples WHERE book_name=? LIMIT 3",
                (book_name,),
            )
            dialogue_rows = cursor.fetchall()
            if dialogue_rows:
                book_data.append("【对话样本】")
                for row in dialogue_rows:
                    book_data.append(f"  场景:{row[0]} | 张力:{row[2]}\n  原文:{row[1][:200]}")

            # 检索描写样本
            cursor.execute(
                "SELECT description_type, original_text, technique_analysis FROM description_samples WHERE book_name=? LIMIT 3",
                (book_name,),
            )
            desc_rows = cursor.fetchall()
            if desc_rows:
                book_data.append("【描写样本】")
                for row in desc_rows:
                    book_data.append(f"  类型:{row[0]} | 技法:{row[2]}\n  原文:{row[1][:200]}")

            # 检索章节模式
            cursor.execute(

                (book_name,),
            )
            pattern_rows = cursor.fetchall()
            if pattern_rows:
                book_data.append(f"【章节模式】开头:{pattern_rows[0][0]} | 结尾:{pattern_rows[0][1]} | 转场:{pattern_rows[0][2]}")

            # 检索叙事距离
            cursor.execute(
                "SELECT distance_type, trigger_reason, original_example FROM narrative_distance WHERE book_name=? LIMIT 2",
                (book_name,),
            )
            nd_rows = cursor.fetchall()
            if nd_rows:
                book_data.append("【叙事距离控制】")
                for row in nd_rows:
                    book_data.append(f"  距离:{row[0]} | 触发:{row[1]}\n  示例:{row[2][:150]}")

            if book_data:
                benchmark_text_parts.append(f"\n===== 标杆作品《{book_name}》=====\n" + "\n".join(book_data))

        return "\n".join(benchmark_text_parts) if benchmark_text_parts else ""

    def _build_review_prompt(
        self,
        chapter_text: str,
        project_name: str,
        chapter_index: int,
        benchmark_data: str,
    ) -> str:
        """构建评审 Prompt"""
        truncated_text = chapter_text[:6000] if len(chapter_text) > 6000 else chapter_text

        benchmark_section = ""
        if benchmark_data:
            benchmark_section = f"""
【知识库标杆参考】
{benchmark_data}

请在评审时对标上述标杆作品的技法水平，指出差距并给出改进方向。
"""

        return f"""你是顶级的文学编辑与写作教练，拥有丰富的小说评审经验。请对以下章节进行全方位质量评审。

【创作项目】{project_name}
【章节序号】第{chapter_index}章
{benchmark_section}
【待评审正文】
{truncated_text}

请输出纯 JSON 格式的评审报告：
{{
  "overall_score": 7.5,
  "dimension_scores": {{
    "pacing": {{"score": 7, "comment": "节奏评价(50字内)"}},
    "dialogue_quality": {{"score": 8, "comment": "对话质量评价(50字内)"}},
    "description_quality": {{"score": 6, "comment": "描写质量评价(50字内)"}},
    "character_consistency": {{"score": 7, "comment": "人物一致性评价(50字内)"}},
    "plot_logic": {{"score": 8, "comment": "情节逻辑评价(50字内)"}},
    "emotional_impact": {{"score": 7, "comment": "情感冲击力评价(50字内)"}},
    "show_vs_tell": {{"score": 6, "comment": "Show vs Tell 平衡评价(50字内)"}}
  }},
  "issues": [
    {{
      "issue_type": "问题类型(节奏拖沓/对话生硬/描写不足/人物失真/情节漏洞/过度告知/视角混乱)",
      "severity": "严重程度(高/中/低)",
      "location": "问题所在位置描述(如:章节中段对话部分)",
      "description": "问题详细描述(100字内)",
      "original_excerpt": "问题原文摘录(50-100字)"
    }}
  ],
  "suggestions": [
    {{
      "suggestion_type": "建议类型(节奏调整/对话优化/描写增强/人物深化/情节修补/Show化改写)",
      "priority": "优先级(高/中/低)",
      "description": "建议详细描述(100字内)",
      "expected_effect": "预期效果(50字内)"
    }}
  ],
  "rewrite_samples": [
    {{
      "original_text": "需要改写的原文片段(50-100字)",
      "rewritten_text": "改写后的示范文本(100-200字)",
      "rewrite_reason": "改写理由(50字内)",
      "technique_applied": "应用的技法(如:Show化/节奏压缩/对话潜台词强化)"
    }}
  ],
  "benchmark_comparison": "与标杆作品的整体对比评价(100字内)"
}}
(⚠️核心要求：
1. 打分范围 1-10 分，必须客观公正！
2. 问题标记必须具体到位置，附带原文摘录！
3. 修改建议必须可操作，说明预期效果！
4. 改写示范必须展示具体的技法应用！
5. 如有标杆数据，必须进行对标分析！
6. 禁止使用反引号，必须输出合法JSON)"""

    def _parse_review_result(
        self,
        data: Dict,
        project_name: str,
        chapter_index: int,
        benchmark_books: Optional[List[str]],
    ) -> Dict[str, Any]:
        """解析评审结果"""
        dimension_scores = data.get("dimension_scores", {})
        scores_summary = {}
        for dim_name, dim_data in dimension_scores.items():
            if isinstance(dim_data, dict):
                scores_summary[dim_name] = {
                    "score": dim_data.get("score", 0),
                    "comment": dim_data.get("comment", ""),
                }

        issues = []
        for issue in data.get("issues", []):
            if isinstance(issue, dict) and issue.get("issue_type"):
                issues.append({
                    "issue_type": issue["issue_type"],
                    "severity": issue.get("severity", "中"),
                    "location": issue.get("location", ""),
                    "description": issue.get("description", ""),
                    "original_excerpt": issue.get("original_excerpt", ""),
                })

        suggestions = []
        for suggestion in data.get("suggestions", []):
            if isinstance(suggestion, dict) and suggestion.get("suggestion_type"):
                suggestions.append({
                    "suggestion_type": suggestion["suggestion_type"],
                    "priority": suggestion.get("priority", "中"),
                    "description": suggestion.get("description", ""),
                    "expected_effect": suggestion.get("expected_effect", ""),
                })

        rewrite_samples = []
        for sample in data.get("rewrite_samples", []):
            if isinstance(sample, dict) and sample.get("rewritten_text"):
                rewrite_samples.append({
                    "original_text": sample.get("original_text", ""),
                    "rewritten_text": sample["rewritten_text"],
                    "rewrite_reason": sample.get("rewrite_reason", ""),
                    "technique_applied": sample.get("technique_applied", ""),
                })

        return {
            "project_name": project_name,
            "chapter_index": chapter_index,
            "overall_score": data.get("overall_score", 0.0),
            "dimension_scores": scores_summary,
            "issues": issues,
            "suggestions": suggestions,
            "rewrite_samples": rewrite_samples,
            "benchmark_comparison": data.get("benchmark_comparison", ""),
            "benchmark_books": "|".join(benchmark_books) if benchmark_books else "",
            "reviewed_at": datetime.now().isoformat(),
        }

    def _empty_result(self, project_name: str, chapter_index: int) -> Dict[str, Any]:
        """返回空评审结果"""
        return {
            "project_name": project_name,
            "chapter_index": chapter_index,
            "overall_score": 0.0,
            "dimension_scores": {},
            "issues": [],
            "suggestions": [],
            "rewrite_samples": [],
            "benchmark_comparison": "",
            "benchmark_books": "",
            "reviewed_at": datetime.now().isoformat(),
        }
