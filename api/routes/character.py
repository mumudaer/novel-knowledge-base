"""
人物查询接口
"""
import json
from typing import Optional
from fastapi import APIRouter, Query
from core.db import get_db_manager
from core.chroma_client import get_chroma_manager
from core.search_utils import hybrid_search

router = APIRouter()


@router.get("/profile")
def get_character_profile(
    query: Optional[str] = Query(None, description="语义搜索关键词（如：反派设计/导师型角色）"),
    book_name: Optional[str] = Query(None, description="书名"),
    character_name: Optional[str] = Query(None, description="人物名"),
    role_type: Optional[str] = Query(None, description="角色定位"),
    limit: int = Query(20, ge=1, le=100, description="返回数量"),
):
    """查询人物档案（混合检索）"""
    columns = [
        "id", "book_name", "author", "category", "name", "role_type",
        "appearance", "quirks", "identity", "motivation", "internal_conflict",
        "fatal_flaw", "symbolism", "personality", "relation_to_mc",
        "relations_to_others", "climax_or_fate", "background",
        "desire_vs_need", "secrets", "fears", "social_masks",
        "growth_cost", "speech_samples", "behavior_samples",
        "relationship_evolution", "abilities", "arc_trajectory", "internal_dilemma",
        "decision_pattern", "cognitive_bias", "transformation_trigger", "contrast_design",
    ]
    
    filters = {}
    if book_name:
        filters["book_name"] = book_name
    if character_name:
        filters["name"] = character_name
    if role_type:
        filters["role_type"] = role_type
    
    results = hybrid_search(
        table="character_profiles",
        collection="character_profiles_kb",
        columns=columns,
        query=query,
        filters=filters,
        limit=limit,
    )
    
    return {"success": True, "data": results, "total": len(results)}


@router.get("/speech-style")
def get_speech_style(
    book_name: str = Query(..., description="书名"),
    character_name: Optional[str] = Query(None, description="人物名"),
):
    """查询人物语言风格"""
    columns = ["id", "book_name", "character_name", "catchphrases",
               "vocabulary_preference", "sentence_pattern",
               "tone_contexts_json", "dialogue_samples_json"]
    
    filters = {"book_name": book_name}
    if character_name:
        filters["character_name"] = character_name
    
    results = hybrid_search(
        table="character_speech_style",
        collection="character_speech_style_kb",
        columns=columns,
        query=None,
        filters=filters,
        limit=100,
    )
    
    # 解析 JSON 字段
    for item in results:
        try:
            item["tone_contexts"] = json.loads(item.get("tone_contexts_json", "{}"))
        except Exception:
            item["tone_contexts"] = {}
        try:
            item["dialogue_samples"] = json.loads(item.get("dialogue_samples_json", "[]"))
        except Exception:
            item["dialogue_samples"] = []

    return {"success": True, "data": results, "total": len(results)}


@router.get("/behavior")
def get_behavior_marks(
    book_name: str = Query(..., description="书名"),
    character_name: Optional[str] = Query(None, description="人物名"),
):
    """查询人物行为标志"""
    columns = ["id", "book_name", "character_name", "habitual_actions",
               "micro_expressions", "defense_mechanisms", "behavior_samples_json"]
    
    filters = {"book_name": book_name}
    if character_name:
        filters["character_name"] = character_name
    
    results = hybrid_search(
        table="character_behavior_marks",
        collection="character_behavior_marks_kb",
        columns=columns,
        query=None,
        filters=filters,
        limit=100,
    )
    
    # 解析 JSON 字段
    for item in results:
        try:
            item["behavior_samples"] = json.loads(item.get("behavior_samples_json", "[]"))
        except Exception:
            item["behavior_samples"] = []

    return {"success": True, "data": results, "total": len(results)}


@router.get("/relationship")
def get_relationship(
    book_name: str = Query(..., description="书名"),
    character_a: Optional[str] = Query(None, description="角色A"),
    character_b: Optional[str] = Query(None, description="角色B"),
):
    """查询人物关系动态"""
    db = get_db_manager()
    cursor = db.connect().cursor()

    query = "SELECT * FROM character_relationship_dynamics WHERE book_name = ?"
    params = [book_name]

    if character_a:
        query += " AND (character_a LIKE ? OR character_b LIKE ?)"
        params.extend([f"%{character_a}%", f"%{character_a}%"])
    if character_b:
        query += " AND (character_a LIKE ? OR character_b LIKE ?)"
        params.extend([f"%{character_b}%", f"%{character_b}%"])

    cursor.execute(query, params)
    rows = cursor.fetchall()

    columns = ["id", "book_name", "character_a", "character_b", "timeline_json"]

    results = []
    for row in rows:
        item = dict(zip(columns, row))
        try:
            item["timeline"] = json.loads(item.get("timeline_json", "[]"))
        except Exception:
            item["timeline"] = []
        results.append(item)

    return {"success": True, "data": results, "total": len(results)}


@router.get("/search")
def search_characters(
    query: str = Query(..., description="语义搜索关键词"),
    book_name: Optional[str] = Query(None, description="书名(可选)"),
    limit: int = Query(10, ge=1, le=50, description="返回数量"),
):
    """语义搜索人物档案"""
    chroma = get_chroma_manager()
    where_filter = {"book_name": book_name} if book_name else None

    results = chroma.query(
        "character_profiles_kb",
        query_texts=[query],
        n_results=limit,
        where=where_filter,
    )

    items = []
    if results and results.get("ids"):
        for i, doc_id in enumerate(results["ids"][0]):
            items.append({
                "id": doc_id,
                "text": results["documents"][0][i] if results.get("documents") else "",
                "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
            })

    return {"success": True, "data": items, "total": len(items)}
