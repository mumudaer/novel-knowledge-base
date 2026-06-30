"""
写作风格查询接口
"""
import json
from typing import Optional
from fastapi import APIRouter, Query
from core.db import get_db_manager
from core.chroma_client import get_chroma_manager
from core.search_utils import hybrid_search

router = APIRouter()


@router.get("/fingerprint")
def get_style_fingerprint(
    book_name: Optional[str] = Query(None, description="书名"),
    query: Optional[str] = Query(None, description="语义搜索关键词（如：文风特点/叙事风格）"),
    limit: int = Query(20, ge=1, le=100, description="返回数量"),
):
    """查询文风指纹 - 支持混合检索"""
    columns = ["id", "book_name", "category", "verbs", "adjectives",
               "imagery", "transitions", "negative_prompts",
               "narrative_perspective", "sentence_rhythm"]
    
    filters = {}
    if book_name:
        filters["book_name"] = book_name
    
    results = hybrid_search(
        table="author_fingerprints",
        collection="author_fingerprints_kb",
        columns=columns,
        query=query,
        filters=filters,
        limit=limit,
    )

    # 将逗号分隔的字符串转为列表
    for item in results:
        for field in ["verbs", "adjectives", "imagery", "transitions"]:
            val = item.get(field, "")
            item[field] = [v.strip() for v in val.split(",") if v.strip()] if val else []

    return {"success": True, "data": results, "total": len(results)}


@router.get("/sensory")
def get_sensory_mappings(
    book_name: Optional[str] = Query(None, description="书名"),
    emotion: Optional[str] = Query(None, description="情绪类型"),
    limit: int = Query(20, ge=1, le=100, description="返回数量"),
):
    """查询感官映射"""
    db = get_db_manager()
    cursor = db.connect().cursor()

    query = "SELECT * FROM sensory_mappings WHERE 1=1"
    params = []

    if book_name:
        query += " AND book_name = ?"
        params.append(book_name)
    if emotion:
        query += " AND emotion LIKE ?"
        params.append(f"%{emotion}%")

    query += f" LIMIT {limit}"
    cursor.execute(query, params)
    rows = cursor.fetchall()

    columns = ["id", "book_name", "chapter_id", "category", "emotion",
               "show_not_tell", "analysis"]
    results = [dict(zip(columns, row)) for row in rows]

    return {"success": True, "data": results, "total": len(results)}


@router.get("/dialogue-samples")
def get_dialogue_samples(
    book_name: Optional[str] = Query(None, description="书名"),
    scene_type: Optional[str] = Query(None, description="场景类型(争吵/告白/谈判/日常/教导)"),
    query: Optional[str] = Query(None, description="语义搜索关键词（如：紧张对话/潜台词对话）"),
    limit: int = Query(10, ge=1, le=50, description="返回数量"),
):
    """查询对话样本 - 支持混合检索"""
    columns = ["id", "book_name", "chapter_id", "scene_type",
               "original_text", "emotional_tension", "subtext", "plot_function"]
    
    filters = {}
    if book_name:
        filters["book_name"] = book_name
    if scene_type:
        filters["scene_type"] = scene_type
    
    results = hybrid_search(
        table="dialogue_samples",
        collection="dialogue_samples_kb",
        columns=columns,
        query=query,
        filters=filters,
        limit=limit,
    )

    return {"success": True, "data": results, "total": len(results)}


@router.get("/description-samples")
def get_description_samples(
    book_name: Optional[str] = Query(None, description="书名"),
    description_type: Optional[str] = Query(None, description="描写类型(打斗/环境/心理/外貌/细节)"),
    query: Optional[str] = Query(None, description="语义搜索关键词（如：环境描写/心理描写）"),
    limit: int = Query(10, ge=1, le=50, description="返回数量"),
):
    """查询描写样本 - 支持混合检索"""
    columns = ["id", "book_name", "chapter_id", "description_type",
               "original_text", "technique_analysis", "sensory_details"]
    
    filters = {}
    if book_name:
        filters["book_name"] = book_name
    if description_type:
        filters["description_type"] = description_type
    
    results = hybrid_search(
        table="description_samples",
        collection="description_samples_kb",
        columns=columns,
        query=query,
        filters=filters,
        limit=limit,
    )

    return {"success": True, "data": results, "total": len(results)}


@router.get("/skills")
def get_narrative_skills(
    book_name: Optional[str] = Query(None, description="书名"),
    scene_type: Optional[str] = Query(None, description="场景类型"),
    skill_name: Optional[str] = Query(None, description="技法名称"),
    limit: int = Query(20, ge=1, le=100, description="返回数量"),
):
    """查询叙事技法"""
    db = get_db_manager()
    cursor = db.connect().cursor()

    query = "SELECT * FROM skills WHERE 1=1"
    params = []

    if book_name:
        query += " AND book_name = ?"
        params.append(book_name)
    if scene_type:
        query += " AND scene_type LIKE ?"
        params.append(f"%{scene_type}%")
    if skill_name:
        query += " AND skill_name LIKE ?"
        params.append(f"%{skill_name}%")

    query += f" LIMIT {limit}"
    cursor.execute(query, params)
    rows = cursor.fetchall()

    columns = ["id", "book_name", "chapter_id", "category", "scene_type",
               "skill_name", "analysis", "original_example", "tags"]
    results = [dict(zip(columns, row)) for row in rows]

    return {"success": True, "data": results, "total": len(results)}


@router.get("/transitions")
def get_transition_samples(
    book_name: Optional[str] = Query(None, description="书名"),
    transition_type: Optional[str] = Query(None, description="转场类型"),
    limit: int = Query(10, ge=1, le=50, description="返回数量"),
):
    """查询转场样本"""
    db = get_db_manager()
    cursor = db.connect().cursor()

    query = "SELECT * FROM transition_samples WHERE 1=1"
    params = []

    if book_name:
        query += " AND book_name = ?"
        params.append(book_name)
    if transition_type:
        query += " AND transition_type LIKE ?"
        params.append(f"%{transition_type}%")

    query += f" LIMIT {limit}"
    cursor.execute(query, params)
    rows = cursor.fetchall()

    columns = ["id", "book_name", "chapter_id", "transition_type",
               "original_text", "technique_analysis"]
    results = [dict(zip(columns, row)) for row in rows]

    return {"success": True, "data": results, "total": len(results)}


@router.get("/summaries")
def get_style_summaries(
    book_name: Optional[str] = Query(None, description="书名"),
    summary_type: Optional[str] = Query(None, description="总结类型(dialogue/description/transition)"),
    limit: int = Query(20, ge=1, le=100, description="返回数量"),
):
    """查询风格总结"""
    db = get_db_manager()
    cursor = db.connect().cursor()

    query = "SELECT * FROM style_summaries WHERE 1=1"
    params = []

    if book_name:
        query += " AND book_name = ?"
        params.append(book_name)
    if summary_type:
        query += " AND summary_type = ?"
        params.append(summary_type)

    query += f" LIMIT {limit}"
    cursor.execute(query, params)
    rows = cursor.fetchall()

    columns = ["id", "book_name", "category", "summary_type",
               "scene_or_desc_type", "style_description", "key_features"]
    results = [dict(zip(columns, row)) for row in rows]

    return {"success": True, "data": results, "total": len(results)}


@router.get("/statistics")
def get_book_statistics(
    book_name: str = Query(..., description="书名"),
):
    """查询书籍统计指标"""
    db = get_db_manager()
    cursor = db.connect().cursor()
    cursor.execute(
        "SELECT * FROM book_statistics WHERE book_name = ?",
        (book_name,),
    )
    rows = cursor.fetchall()

    columns = ["id", "book_name", "total_words", "avg_chapter_words",
               "min_chapter_words", "max_chapter_words", "median_chapter_words",
               "dialogue_ratio", "description_ratio", "avg_paragraph_length",
               "short_para_ratio", "medium_para_ratio", "long_para_ratio",
               "rhythm_pattern"]
    results = [dict(zip(columns, row)) for row in rows]

    return {"success": True, "data": results, "total": len(results)}


@router.get("/narrative-distance")
def get_narrative_distance(
    book_name: Optional[str] = Query(None, description="书名"),
    distance_type: Optional[str] = Query(None, description="叙事距离类型"),
    limit: int = Query(20, ge=1, le=100, description="返回数量"),
):
    """查询叙事距离控制"""
    db = get_db_manager()
    cursor = db.connect().cursor()

    query = "SELECT * FROM narrative_distance WHERE 1=1"
    params = []

    if book_name:
        query += " AND book_name = ?"
        params.append(book_name)
    if distance_type:
        query += " AND distance_type LIKE ?"
        params.append(f"%{distance_type}%")

    query += f" LIMIT {limit}"
    cursor.execute(query, params)
    rows = cursor.fetchall()

    columns = ["id", "book_name", "chapter_id", "distance_type",
               "trigger_reason", "original_example"]
    results = [dict(zip(columns, row)) for row in rows]

    return {"success": True, "data": results, "total": len(results)}


@router.get("/show-tell")
def get_show_tell_patterns(
    book_name: Optional[str] = Query(None, description="书名"),
    pattern_type: Optional[str] = Query(None, description="模式类型"),
    limit: int = Query(20, ge=1, le=100, description="返回数量"),
):
    """查询 Show vs Tell 策略"""
    db = get_db_manager()
    cursor = db.connect().cursor()

    query = "SELECT * FROM show_tell_patterns WHERE 1=1"
    params = []

    if book_name:
        query += " AND book_name = ?"
        params.append(book_name)
    if pattern_type:
        query += " AND pattern_type LIKE ?"
        params.append(f"%{pattern_type}%")

    query += f" LIMIT {limit}"
    cursor.execute(query, params)
    rows = cursor.fetchall()

    columns = ["id", "book_name", "chapter_id", "pattern_type",
               "ratio_estimate", "switching_triggers", "original_example"]
    results = [dict(zip(columns, row)) for row in rows]

    return {"success": True, "data": results, "total": len(results)}
