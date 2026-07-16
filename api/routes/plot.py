"""
大纲/细纲查询接口
"""
import json
from typing import Optional
from fastapi import APIRouter, Query
from core.db import get_db_manager
from core.chroma_client import get_chroma_manager
from core.search_utils import hybrid_search

router = APIRouter()


    
    




    
    




@router.get("/subplots")
def get_subplots(
    book_name: str = Query(..., description="书名"),
    limit: int = Query(50, ge=1, le=500, description="返回数量"),
    offset: int = Query(0, ge=0, description="偏移量"),
):
    """查询支线剧情"""
    db = get_db_manager()
    cursor = db.connect().cursor()
    cursor.execute(
        "SELECT * FROM plot_lines WHERE book_name = ? LIMIT ? OFFSET ? AND line_type = 'subplot' LIMIT ? OFFSET ?",
        (book_name, limit, offset),
    )
    rows = cursor.fetchall()

    columns = ["id", "book_name", "line_type", "theme", "chapter_distribution", "milestones_json"]
    results = []
    for row in rows:
        item = dict(zip(columns, row))
        try:
            item["milestones"] = json.loads(item.get("milestones_json", "[]"))
        except Exception:
            item["milestones"] = []
        results.append(item)

    return {"success": True, "data": results, "total": len(results)}












@router.get("/foreshadowing")
def get_foreshadowing(
    book_name: str = Query(..., description="书名"),
    status: Optional[str] = Query(None, description="状态(未填/已填)"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """查询伏笔追踪"""
    db = get_db_manager()
    cursor = db.connect().cursor()

    query = "SELECT * FROM plot_foreshadowing WHERE book_name = ? LIMIT ? OFFSET ?"
    params = [book_name]

    if status:
        query += " AND status = ?"
        params.append(status)

    cursor.execute(query, params)
    rows = cursor.fetchall()

    columns = ["id", "book_name", "hook_name", "planted_chapter",
               "planned_payoff", "status", "resolved_chapter"]
    results = [dict(zip(columns, row)) for row in rows]

    return {"success": True, "data": results, "total": len(results)}







@router.get("/chapter-functions")
def get_chapter_functions(
    book_name: str = Query(..., description="书名"),
    function_type: Optional[str] = Query(None, description="章节功能类型"),
    limit: int = Query(50, ge=1, le=500, description="返回数量"),
):
    """查询章节功能分类"""
    db = get_db_manager()
    cursor = db.connect().cursor()

    query = "SELECT * FROM chapter_functions WHERE book_name = ? LIMIT ? OFFSET ?"
    params = [book_name]

    if function_type:
        query += " AND function_type = ?"
        params.append(function_type)

    query += f" LIMIT {limit}"
    cursor.execute(query, params)
    rows = cursor.fetchall()

    columns = ["id", "book_name", "chapter_id", "function_type",
               "structure_pattern_json", "hook_type", "hook_content",
               "information_gap_json", "active_plotlines"]

    results = []
    for row in rows:
        item = dict(zip(columns, row))
        try:
            item["structure_pattern"] = json.loads(item.get("structure_pattern_json", "{}"))
        except Exception:
            item["structure_pattern"] = {}
        try:
            item["information_gap"] = json.loads(item.get("information_gap_json", "{}"))
        except Exception:
            item["information_gap"] = {}
        try:
            item["active_plotlines"] = json.loads(item.get("active_plotlines", "[]"))
        except Exception:
            item["active_plotlines"] = []
        results.append(item)

    return {"success": True, "data": results, "total": len(results)}


@router.get("/volume-outlines")
def get_volume_outlines(
    book_name: str = Query(..., description="书名"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """查询卷大纲"""
    db = get_db_manager()
    cursor = db.connect().cursor()
    cursor.execute(
        "SELECT * FROM macro_outlines WHERE book_name = ? ORDER BY volume_index LIMIT ? OFFSET ?",
        (book_name, limit, offset),
    )
    rows = cursor.fetchall()

    columns = ["id", "book_name", "category", "volume_index", "chapter_range",
               "theme", "conflict", "beats_json", "arc"]

    results = []
    for row in rows:
        item = dict(zip(columns, row))
        try:
            item["beats"] = json.loads(item.get("beats_json", "[]"))
        except Exception:
            item["beats"] = []
        results.append(item)

    return {"success": True, "data": results, "total": len(results)}


@router.get("/revelation-pacing")
def get_revelation_pacing(
    book_name: str = Query(..., description="书名"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """查询信息揭露节奏"""
    db = get_db_manager()
    cursor = db.connect().cursor()
    cursor.execute(
        "SELECT * FROM revelation_pacing WHERE book_name = ? LIMIT ? OFFSET ?",
        (book_name, limit, offset),
    )
    rows = cursor.fetchall()

    columns = ["id", "book_name", "revelation_name", "reveal_chapter",
               "reveal_method", "impact"]
    results = [dict(zip(columns, row)) for row in rows]

    return {"success": True, "data": results, "total": len(results)}







@router.get("/emotion-transitions")
def get_emotion_transitions(
    book_name: str = Query(..., description="书名"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """查询情感转变铺垫模式"""
    db = get_db_manager()
    cursor = db.connect().cursor()
    cursor.execute(
        "SELECT * FROM emotion_transition_patterns WHERE book_name = ? LIMIT ? OFFSET ?",
        (book_name, limit, offset),
    )
    rows = cursor.fetchall()

    columns = ["id", "book_name", "transition_type", "foreshadowing_method"]
    results = [dict(zip(columns, row)) for row in rows]

    return {"success": True, "data": results, "total": len(results)}


@router.get("/information-management")
def get_information_management(
    book_name: str = Query(..., description="书名"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """查询全书信息管理策略"""
    db = get_db_manager()
    cursor = db.connect().cursor()
    cursor.execute(
        "SELECT * FROM information_management WHERE book_name = ? LIMIT ? OFFSET ?",
        (book_name, limit, offset),
    )
    rows = cursor.fetchall()

    columns = ["id", "book_name", "strategy_type", "target_info",
               "conceal_method", "reveal_timing", "dramatic_purpose"]
    results = [dict(zip(columns, row)) for row in rows]

    return {"success": True, "data": results, "total": len(results)}


@router.get("/climax-buildup")
def get_climax_buildup(
    book_name: str = Query(..., description="书名"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """查询高潮构建链"""
    db = get_db_manager()
    cursor = db.connect().cursor()
    cursor.execute(
        "SELECT * FROM climax_buildup_chains WHERE book_name = ? LIMIT ? OFFSET ?",
        (book_name, limit, offset),
    )
    rows = cursor.fetchall()

    columns = ["id", "book_name", "climax_name", "climax_chapter",
               "buildup_steps_json", "tension_escalation"]
    results = []
    for row in rows:
        item = dict(zip(columns, row))
        try:
            item["buildup_steps"] = json.loads(item.get("buildup_steps_json", "[]"))
        except Exception:
            item["buildup_steps"] = []
        results.append(item)

    return {"success": True, "data": results, "total": len(results)}


@router.get("/conflict-escalation")
def get_conflict_escalation(
    book_name: str = Query(..., description="书名"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """查询冲突升级阶梯"""
    db = get_db_manager()
    cursor = db.connect().cursor()
    cursor.execute(
        "SELECT * FROM conflict_escalation WHERE book_name = ? LIMIT ? OFFSET ?",
        (book_name, limit, offset),
    )
    rows = cursor.fetchall()

    columns = ["id", "book_name", "conflict_line",
               "escalation_steps_json", "escalation_pattern"]
    results = []
    for row in rows:
        item = dict(zip(columns, row))
        try:
            item["escalation_steps"] = json.loads(item.get("escalation_steps_json", "[]"))
        except Exception:
            item["escalation_steps"] = []
        results.append(item)

    return {"success": True, "data": results, "total": len(results)}
