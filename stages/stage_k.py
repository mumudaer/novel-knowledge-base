"""
Stage K: 知识库引用推荐模块
根据创作项目的题材/类型，从知识库中检索最相关的标杆作品
为世界观/人物/大纲/风格各维度推荐参考素材
"""
import logging
from typing import List, Dict, Any, Optional
from stages.base import BaseStage
from core.ollama_client import ollama_chat, safe_parse_json
from core.utils import generate_id

logger = logging.getLogger(__name__)


class StageK(BaseStage):
    """Stage K: 知识库引用推荐"""

    def __init__(self, book_name: str = "creative", category: str = "fiction"):
        super().__init__("K", book_name, category)

    def run(
        self,
        project_name: str,
        genre: str = "",
        premise: str = "",
        target_dimensions: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        执行知识库引用推荐

        Args:
            project_name: 创作项目名称
            genre: 题材/类型（如：玄幻、都市、科幻）
            premise: 故事前提/简介
            target_dimensions: 目标维度列表，默认全部

        Returns:
            推荐结果字典
        """
        logger.info(f"=== 阶段K：知识库引用推荐 (项目: {project_name}) ===")

        if target_dimensions is None:
            target_dimensions = ["world_settings", "character_profiles", "plot_structure", "writing_style"]

        # 1. 获取知识库中所有可用书籍
        available_books = self._get_available_books()
        if not available_books:
            logger.warning("⚠️ [阶段K] 知识库为空，无法生成推荐")
            return {"project_name": project_name, "recommendations": []}

        # 2. 收集各书籍的概要信息
        book_summaries = self._collect_book_summaries(available_books)

        # 3. 调用 LLM 进行题材匹配和维度推荐
        prompt = self._build_recommend_prompt(project_name, genre, premise, target_dimensions, book_summaries)

        try:
            resp = ollama_chat(prompt, 0.2, "K")
            data = safe_parse_json(resp)
            if not data:
                logger.warning("⚠️ [阶段K] LLM 返回解析失败")
                return {"project_name": project_name, "recommendations": []}

            result = self._parse_recommendations(data, project_name, genre)
            logger.info(
                f"✅ [阶段K战报] 推荐条目: {len(result.get('recommendations', []))} | "
                f"覆盖维度: {len(set(r.get('dimension', '') for r in result.get('recommendations', [])))}"
            )
            return result

        except Exception as exc:
            logger.error(f"❌ [阶段K] 推荐失败: {exc}")
            return {"project_name": project_name, "recommendations": []}

    def insert(self, results: Dict[str, Any]) -> Dict[str, int]:
        """将推荐结果写入数据库"""
        cursor = self.db.connect().cursor()
        stats = {"kb_references": 0}

        for rec in results.get("recommendations", []):
            ref_id = generate_id(
                results["project_name"],
                rec.get("dimension", ""),
                rec.get("ref_book", ""),
                rec.get("ref_table", ""),
            )
            cursor.execute(
                "INSERT OR REPLACE INTO kb_references VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    ref_id,
                    results["project_name"],
                    rec.get("target_type", ""),
                    rec.get("target_id", ""),
                    rec.get("ref_book", ""),
                    rec.get("ref_table", ""),
                    rec.get("ref_id", ""),
                    rec.get("ref_content", ""),
                    rec.get("usage_purpose", ""),
                ),
            )
            stats["kb_references"] += 1

        self.db.commit()
        logger.info(f"   ✅ [阶段K] {stats['kb_references']} 条推荐已写入")
        return stats

    def _get_available_books(self) -> List[str]:
        """获取知识库中所有可用的书籍名称"""
        cursor = self.db.connect().cursor()
        cursor.execute("SELECT DISTINCT book_name FROM book_metadata")
        rows = cursor.fetchall()
        return [row[0] for row in rows if row[0]]

    def _collect_book_summaries(self, book_names: List[str]) -> str:
        """收集各书籍的概要信息，用于题材匹配"""
        cursor = self.db.connect().cursor()
        summary_parts = []

        for book_name in book_names:
            book_info = []

            # 获取书籍分类
            cursor.execute(
                "SELECT category FROM book_metadata WHERE book_name=? LIMIT 1",
                (book_name,),
            )
            cat_row = cursor.fetchone()
            category = cat_row[0] if cat_row else "未知"
            book_info.append(f"分类: {category}")

            # 获取世界观模块
            cursor.execute(
                "SELECT module, entity FROM world_settings WHERE book_name=? LIMIT 5",
                (book_name,),
            )
            ws_rows = cursor.fetchall()
            if ws_rows:
                modules = [f"{r[0]}:{r[1]}" for r in ws_rows]
                book_info.append(f"世界观: {', '.join(modules)}")

            # 获取主要人物
            cursor.execute(
                "SELECT name, role_type FROM character_profiles WHERE book_name=? LIMIT 5",
                (book_name,),
            )
            cp_rows = cursor.fetchall()
            if cp_rows:
                chars = [f"{r[0]}({r[1]})" for r in cp_rows]
                book_info.append(f"人物: {', '.join(chars)}")

            # 获取全书结构
            cursor.execute(
                "SELECT structure_type, surface_theme, deep_theme FROM book_structure WHERE book_name=? LIMIT 1",
                (book_name,),
            )
            bs_row = cursor.fetchone()
            if bs_row:
                book_info.append(f"结构: {bs_row[0]} | 表层主题: {bs_row[1]} | 深层主题: {bs_row[2]}")

            # 获取主线
            cursor.execute(
                "SELECT theme FROM plot_lines WHERE book_name=? AND line_type='main' LIMIT 1",
                (book_name,),
            )
            pl_row = cursor.fetchone()
            if pl_row:
                book_info.append(f"主线: {pl_row[0]}")

            summary_parts.append(f"【《{book_name}》】\n" + "\n".join(book_info))

        return "\n\n".join(summary_parts)

    def _build_recommend_prompt(
        self,
        project_name: str,
        genre: str,
        premise: str,
        target_dimensions: List[str],
        book_summaries: str,
    ) -> str:
        """构建推荐 Prompt"""
        dimension_desc = {
            "world_settings": "世界观设定（力量体系/社会结构/地理空间/规则体系）",
            "character_profiles": "人物档案（性格设计/弧光轨迹/关系网络/对话风格）",
            "plot_structure": "剧情结构（大纲设计/冲突升级/伏笔管理/节奏控制）",
            "writing_style": "写作风格（叙事距离/Show vs Tell/描写技法/对话风格）",
        }
        dimensions_text = "\n".join([
            f"  - {dim}: {dimension_desc.get(dim, dim)}"
            for dim in target_dimensions
        ])

        return f"""你是资深的文学顾问，精通各类小说创作。请根据用户的创作项目信息，从知识库中推荐最相关的标杆作品和具体参考素材。

【创作项目】{project_name}
【题材类型】{genre}
【故事前提】{premise}

【需要推荐的维度】
{dimensions_text}

【知识库可用书籍及其概要】
{book_summaries}

请输出纯 JSON 格式：
{{
  "recommendations": [
    {{
      "dimension": "推荐维度(world_settings/character_profiles/plot_structure/writing_style)",
      "ref_book": "推荐参考的书名",
      "ref_table": "推荐参考的数据表(world_settings/character_profiles/dialogue_samples/description_samples/macro_outlines等)",
      "ref_content_summary": "推荐内容的摘要(100字内)",
      "relevance_score": 8,
      "relevance_reason": "推荐理由：为什么这本书的这个维度与用户项目最相关(100字内)",
      "usage_suggestion": "使用建议：用户应如何参考这个素材来创作(100字内)"
    }}
  ]
}}
(⚠️核心要求：
1. 每个维度至少推荐 1-2 个最相关的参考！
2. 相关性打分 1-10，必须基于题材匹配度！
3. 推荐理由必须具体，说明为什么相关！
4. 使用建议必须可操作！
5. 如果知识库中没有高度相关的作品，也要推荐最接近的并说明差距！
6. 禁止使用反引号，必须输出合法JSON)"""

    def _parse_recommendations(
        self,
        data: Dict,
        project_name: str,
        genre: str,
    ) -> Dict[str, Any]:
        """解析推荐结果"""
        recommendations = []
        for rec in data.get("recommendations", []):
            if isinstance(rec, dict) and rec.get("ref_book") and rec.get("dimension"):
                recommendations.append({
                    "dimension": rec["dimension"],
                    "target_type": "project",
                    "target_id": project_name,
                    "ref_book": rec["ref_book"],
                    "ref_table": rec.get("ref_table", ""),
                    "ref_id": "",
                    "ref_content": rec.get("ref_content_summary", ""),
                    "usage_purpose": rec.get("usage_suggestion", ""),
                    "relevance_score": rec.get("relevance_score", 0),
                    "relevance_reason": rec.get("relevance_reason", ""),
                })

        return {
            "project_name": project_name,
            "genre": genre,
            "recommendations": recommendations,
        }
