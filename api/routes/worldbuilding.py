"""
世界观查询接口
"""
import json
from typing import Optional
from fastapi import APIRouter, Query
from core.db import get_db_manager
from core.chroma_client import get_chroma_manager
from core.search_utils import hybrid_search

router = APIRouter()


@router.get("/settings")
def get_world_settings(
    query: Optional[str] = Query(None, description="语义搜索关键词（如：玄幻力量体系设计）"),
    book_name: Optional[str] = Query(None, description="书名"),
    module: Optional[str] = Query(None, description="设定模块(如:力量体系/社会阶层)"),
    tags: Optional[str] = Query(None, description="标签(逗号分隔)"),
    limit: int = Query(20, ge=1, le=100, description="返回数量"),
):
    """查询世界观设定（混合检索）"""
    columns = ["id", "book_name", "author", "category", "module", "entity",
               "content", "tags", "daily_life", "taboos", "conflict_roots",
               "geography", "economy", "culture", "causal_chain", "rules_exceptions"]
    
    filters = {}
    if book_name:
        filters["book_name"] = book_name
    if module:
        filters["module"] = module
    if tags:
        filters["tags"] = tags
    
    results = hybrid_search(
        table="world_settings",
        collection="world_settings_kb",
        columns=columns,
        query=query,
        filters=filters,
        limit=limit,
    )
    
    return {"success": True, "data": results, "total": len(results)}


@router.get("/timeline")
def get_world_timeline(
    book_name: str = Query(..., description="书名"),
    limit: int = Query(50, ge=1, le=500, description="返回数量"),
    offset: int = Query(0, ge=0, description="偏移量"),
):
    """查询编年史"""
    db = get_db_manager()
    cursor = db.connect().cursor()
    cursor.execute(
        "SELECT * FROM world_timeline WHERE book_name = ? ORDER BY era_or_year LIMIT ? OFFSET ?",
        (book_name, limit, offset),
    )
    rows = cursor.fetchall()

    columns = ["id", "book_name", "era_or_year", "event_name", "event_description", "impact"]
    results = [dict(zip(columns, row)) for row in rows]
    return {"success": True, "data": results, "total": len(results)}


@router.get("/conflicts")
def get_conflicts(
    book_name: str = Query(..., description="书名"),
):
    """查询冲突根源"""
    db = get_db_manager()
    cursor = db.connect().cursor()
    cursor.execute(
        "SELECT book_name, module, entity, conflict_roots FROM world_settings WHERE book_name = ? AND conflict_roots != ''",
        (book_name,),
    )
    rows = cursor.fetchall()

    results = [
        {"book_name": r[0], "module": r[1], "entity": r[2], "conflict_roots": r[3]}
        for r in rows
    ]
    return {"success": True, "data": results, "total": len(results)}


@router.get("/search")
def search_world_settings(
    query: str = Query(..., description="语义搜索关键词"),
    book_name: Optional[str] = Query(None, description="书名(可选)"),
    limit: int = Query(10, ge=1, le=50, description="返回数量"),
):
    """语义搜索世界观设定（混合检索）"""
    columns = ["id", "book_name", "author", "category", "module", "entity",
               "content", "tags", "daily_life", "taboos", "conflict_roots",
               "geography", "economy", "culture", "causal_chain", "rules_exceptions"]
    
    filters = {}
    if book_name:
        filters["book_name"] = book_name
    
    results = hybrid_search(
        table="world_settings",
        collection="world_settings_kb",
        columns=columns,
        query=query,
        filters=filters,
        limit=limit,
    )
    
    return {"success": True, "data": results, "total": len(results)}


@router.get("/factions")
def get_faction_networks(
    book_name: str = Query(..., description="书名"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """查询势力关系网络"""
    db = get_db_manager()
    cursor = db.connect().cursor()
    cursor.execute(
        "SELECT * FROM faction_networks WHERE book_name = ? LIMIT ? OFFSET ?",
        (book_name, limit, offset),
    )
    rows = cursor.fetchall()

    columns = ["id", "book_name", "faction_a", "faction_b",
               "relation_type", "relation_detail", "stability", "key_events"]
    results = [dict(zip(columns, row)) for row in rows]

    return {"success": True, "data": results, "total": len(results)}


@router.get("/setting-evolutions")
def get_setting_evolutions(
    book_name: str = Query(..., description="书名"),
    setting_module: Optional[str] = Query(None, description="设定模块"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """查询设定演变追踪"""
    db = get_db_manager()
    cursor = db.connect().cursor()

    query = "SELECT * FROM setting_evolutions WHERE book_name = ? LIMIT ? OFFSET ?"
    params = [book_name]

    if setting_module:
        query += " AND setting_module LIKE ?"
        params.append(f"%{setting_module}%")

    cursor.execute(query, params)
    rows = cursor.fetchall()

    columns = ["id", "book_name", "setting_module", "setting_entity",
               "chapter_range", "evolution_type", "before_state",
               "after_state", "trigger_event"]
    results = [dict(zip(columns, row)) for row in rows]

    return {"success": True, "data": results, "total": len(results)}
