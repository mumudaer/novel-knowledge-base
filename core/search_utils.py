"""
混合检索工具模块
提供统一的混合检索接口，支持向量召回 + SQL 过滤 + 去重合并
"""
from typing import Optional, List, Dict, Any
from core.db import get_db_manager
from core.chroma_client import get_chroma_manager

# 表名白名单，防止 SQL 注入
ALLOWED_TABLES = {
    "world_settings",
    "character_profiles",
    "dialogue_samples",
    "description_samples",
    "transition_samples",
    "narrative_distance",
    "show_tell_patterns",
    "style_summaries",
    "genre_specific_techniques",
    "fear_building",
    "plot_arcs",
    "book_structure",
    "macro_outlines",
    "climax_buildup_chains",
    "conflict_escalation",
    "information_management",
    "action_scene_samples",
    "character_speech_style",
    "character_behavior_marks",
}


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
    sql_query = f"SELECT * FROM {table} WHERE 1=1"
    params = []

    if filters:
        for field, value in filters.items():
            if value:
                if field == "book_name":
                    sql_query += f" AND {field} = ?"
                else:
                    sql_query += f" AND {field} LIKE ?"
                    value = f"%{value}%"
                params.append(value)

    sql_query += f" LIMIT {limit}"
    cursor.execute(sql_query, params)
    rows = cursor.fetchall()

    for row in rows:
        row_dict = dict(zip(columns, row))
        row_id = row_dict.get("id", "")
        if row_id and row_id not in seen_ids:
            seen_ids.add(row_id)
            row_dict["source"] = "structured"
            results.append(row_dict)

    return results[:limit]
