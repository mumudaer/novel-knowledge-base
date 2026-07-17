"""
混合检索工具模块
提供统一的混合检索接口，支持向量召回 + SQL 过滤 + 去重合并
"""
import re
import logging
from typing import Optional, List, Dict, Any
from core.db import get_db_manager
from core.chroma_client import get_chroma_manager

logger = logging.getLogger(__name__)

# 表名白名单，防止 SQL 注入
def _get_allowed_tables():
    from core.db import DatabaseManager
    return set(DatabaseManager.TABLE_SCHEMAS.keys())
ALLOWED_TABLES = _get_allowed_tables()
# (was hardcoded, now auto-generated from TABLE_SCHEMAS)
_OLD_ALLOWED_TABLES = {
    # Stage A
    "plot_arcs",
    # Stage B
    "skills",
    # Stage C
    "author_fingerprints", "sensory_mappings",
    # Stage D
    "world_settings", "character_profiles", "world_timeline", "golden_finger",
    "faction_networks", "setting_evolutions",
    # Stage E
    "macro_outlines", "plot_foreshadowing", "entity_state_tracker", "chapter_functions",
    # Stage F
    "dialogue_samples", "description_samples", "transition_samples", "style_summaries",
    "action_scene_samples", "climax_excerpts", "memorable_quotes",
    "chapter_opening_ending_samples", "narrative_distance", "show_tell_patterns",
    # Stage G
    "character_speech_style", "character_behavior_marks", "character_relationship_dynamics",
    # Stage H
    "plot_lines", "revelation_pacing",
    "emotion_transition_patterns", "information_management",
    "climax_buildup_chains", "conflict_escalation",
    # Stage I
    "book_statistics",
    # 通用类型补强
    # 高级功能
    "cross_book_comparisons", "common_mistakes", "technique_combinations",
    # 元数据与服务层
    "book_metadata", "chapter_reviews", "kb_references", "search_logs", "quality_checks",
    # Stage O: 事件因果图谱
    "story_events", "event_causal_edges",
    # 后处理聚合
}

# 字段名白名单，防止 SQL 注入（仅允许字母、数字、下划线）
_FIELD_NAME_RE = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


def hybrid_search(
    table: str,
    collection: str,
    columns: List[str],
    query: Optional[str] = None,
    filters: Dict[str, Any] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    混合检索：向量召回 + SQL 过滤 + 去重合并

    1. 如果有 query：先 ChromaDB 向量召回 Top-K
    2. 如果有过滤条件：SQL 精确过滤补充
    3. 按 id 去重，向量结果优先
    4. 返回统一的 results 列表

    Args:
        table: SQLite 表名（必须在白名单中）
        collection: ChromaDB 集合名
        columns: 列名列表
        query: 语义搜索关键词
        filters: SQL 过滤条件字典 {字段名: 值}
        limit: 返回数量上限

    Returns:
        去重合并后的结果列表

    Raises:
        ValueError: 表名不在白名单中
    """
    # 表名白名单验证
    if table not in ALLOWED_TABLES:
        raise ValueError(f"表名 {table} 不在白名单中，允许的表：{', '.join(sorted(ALLOWED_TABLES))}")

    db = get_db_manager()
    cursor = db.connect().cursor()
    seen_ids = set()
    results = []

    # 1. 向量召回（优先）
    if query:
        try:
            chroma = get_chroma_manager()
            where_filter = None
            if filters and "book_name" in filters:
                where_filter = {"book_name": filters["book_name"]}

            chroma_res = chroma.query(
                collection,
                query_texts=[query],
                n_results=limit,
                where=where_filter,
            )

            if chroma_res and chroma_res.get("ids"):
                for i, doc_id in enumerate(chroma_res["ids"][0]):
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        result_item = {
                            "id": doc_id,
                            "source": "semantic",
                            "text": chroma_res["documents"][0][i] if chroma_res.get("documents") else "",
                            "metadata": chroma_res["metadatas"][0][i] if chroma_res.get("metadatas") else {},
                        }
                        results.append(result_item)
        except Exception:
            pass

    # 2. SQL 过滤补充
    col_str = ", ".join(columns) if columns else "*"
    sql_query = f"SELECT {col_str} FROM {table} WHERE 1=1"
    params = []

    if filters:
        for field, value in filters.items():
            if value:
                # 字段名白名单校验，防止 SQL 注入
                if not _FIELD_NAME_RE.match(field):
                    logger.warning(f"跳过非法字段名: {field}")
                    continue
                if field == "book_name":
                    sql_query += f" AND {field} = ?"
                else:
                    sql_query += f" AND {field} LIKE ?"
                    value = f"%{value}%"
                params.append(value)

    sql_query += f" LIMIT {limit}"
    cursor.execute(sql_query, params)
    rows = cursor.fetchall()

    # 使用 cursor.description 获取实际列名，避免与传入的 columns 顺序不匹配
    actual_columns = [desc[0] for desc in cursor.description] if cursor.description else columns

    for row in rows:
        row_dict = dict(zip(actual_columns, row))
        row_id = row_dict.get("id", "")
        if row_id and row_id not in seen_ids:
            seen_ids.add(row_id)
            row_dict["source"] = "structured"
            results.append(row_dict)

    return results[:limit]
