"""
知识库搜索引擎 API 路由
为 Reasonix 创作 skill 提供多维度知识库搜索/检索接口
覆盖：世界观/人物档案/写作风格/大纲结构/正文样本 等创作维度
"""
import json
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel, Field
import networkx as nx
from core.db import get_db_manager
from core.chroma_client import get_chroma_manager
from core.graph import get_graph_manager
from core.search_utils import hybrid_search

router = APIRouter()


# ===================== Pydantic 模型 =====================

class ComprehensiveSearchRequest(BaseModel):
    text: str = Field(..., description="用户创作内容文本（如世界观设定、人物描述等）")
    dimensions: List[str] = Field(
        default=["world", "character", "style", "plot"],
        description="搜索维度列表: world/character/style/plot/excerpt",
    )
    limit: int = Field(default=5, ge=1, le=20, description="每个维度返回的最大结果数")


class ReviewRequest(BaseModel):
    chapter_text: str = Field(..., description="待评审的章节正文")
    chapter_index: int = Field(default=1, description="章节序号")
    project_name: str = Field(default="default", description="项目标识（用于区分不同创作项目）")
    benchmark_books: Optional[List[str]] = Field(default=None, description="标杆作品书名列表")


class RecommendRequest(BaseModel):
    genre: str = Field(default="", description="题材/类型")
    premise: str = Field(default="", description="故事前提/简介")
    project_name: str = Field(default="default", description="项目标识")
    target_dimensions: Optional[List[str]] = Field(
        default=None,
        description="目标维度: world_settings/character_profiles/plot_structure/writing_style",
    )


class CompareRequest(BaseModel):
    dimension: str = Field(..., description="对比维度（如：感情线设计/反派塑造/力量体系）")
    book_names: Optional[List[str]] = Field(default=None, description="参与对比的书名列表，为空则使用全部标杆书")
    category: Optional[str] = Field(default=None, description="限定分类")


class ContextPushRequest(BaseModel):
    context_text: str = Field(..., description="当前创作上下文文本")
    creation_stage: Optional[str] = Field(default=None, description="创作阶段（如：大纲/人物设计/正文写作）")
    genre: Optional[str] = Field(default=None, description="题材/类型")
    project_name: str = Field(default="default", description="项目标识")


# ===================== 混合检索工具函数 =====================

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
        table: SQLite 表名
        collection: ChromaDB 集合名
        columns: 列名列表
        query: 语义搜索关键词
        filters: SQL 过滤条件字典 {字段名: 值}
        limit: 返回数量上限

    Returns:
        去重合并后的结果列表
    """
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


# ===================== 世界观搜索 =====================

@router.get("/search/world")
def search_world(
    query: Optional[str] = Query(None, description="语义搜索关键词（如：玄幻力量体系设计）"),
    module: Optional[str] = Query(None, description="设定模块过滤（如：力量体系/社会阶层/地理空间）"),
    book_name: Optional[str] = Query(None, description="限定书名"),
    limit: int = Query(10, ge=1, le=50, description="返回数量"),
):
    """
    世界观知识搜索（混合检索）

    向量召回优先 + SQL 精确过滤补充 + 去重合并。
    适用场景：创作世界观时，搜索标杆作品的力量体系/社会结构/地理设定等。
    """
    columns = ["id", "book_name", "author", "category", "module", "entity",
               "content", "tags", "daily_life", "taboos", "conflict_roots",
               "geography", "economy", "culture", "causal_chain", "rules_exceptions"]
    
    filters = {}
    if book_name:
        filters["book_name"] = book_name
    if module:
        filters["module"] = module
    
    results = hybrid_search(
        table="world_settings",
        collection="world_settings_kb",
        columns=columns,
        query=query,
        filters=filters,
        limit=limit,
    )

    _log_search("default", "world", query or module or "", len(results))

    return {
        "success": True,
        "data": results,
        "total": len(results),
    }


# ===================== 人物档案搜索 =====================

@router.get("/search/character")
def search_character(
    query: Optional[str] = Query(None, description="语义搜索关键词（如：反派设计/导师型角色）"),
    role_type: Optional[str] = Query(None, description="角色定位过滤（主角/反派/导师/配角）"),
    character_name: Optional[str] = Query(None, description="人物名模糊匹配"),
    book_name: Optional[str] = Query(None, description="限定书名"),
    limit: int = Query(10, ge=1, le=50, description="返回数量"),
):
    """
    人物档案搜索（混合检索）

    向量召回优先 + SQL 精确过滤补充 + 去重合并。
    适用场景：创作人物档案时，搜索标杆作品的同类角色设计。
    """
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
    if role_type:
        filters["role_type"] = role_type
    if character_name:
        filters["name"] = character_name
    
    results = hybrid_search(
        table="character_profiles",
        collection="character_profiles_kb",
        columns=columns,
        query=query,
        filters=filters,
        limit=limit,
    )

    _log_search("default", "character", query or role_type or "", len(results))

    return {
        "success": True,
        "data": results,
        "total": len(results),
    }


# ===================== 写作风格搜索 =====================

@router.get("/search/style")
def search_style(
    query: Optional[str] = Query(None, description="语义搜索关键词（如：如何写好紧张氛围/对话潜台词）"),
    technique_type: Optional[str] = Query(
        None,
        description="技法类型过滤（dialogue/description/transition/narrative_distance/show_tell）",
    ),
    book_name: Optional[str] = Query(None, description="限定书名"),
    limit: int = Query(10, ge=1, le=50, description="返回数量"),
):
    """
    写作风格搜索（混合检索）

    聚合对话样本、描写样本、转场样本、叙事距离、Show vs Tell 策略。
    向量召回优先 + SQL 精确过滤补充 + 去重合并。
    适用场景：创作正文时，搜索标杆作品的写作技法和范文。
    """
    db = get_db_manager()
    cursor = db.connect().cursor()
    result_sections = {}
    
    filters = {}
    if book_name:
        filters["book_name"] = book_name

    # 对话样本（混合检索）
    if not technique_type or technique_type == "dialogue":
        cols = ["id", "book_name", "chapter_id", "scene_type",
                "original_text", "emotional_tension", "subtext", "plot_function"]
        result_sections["dialogue_samples"] = hybrid_search(
            table="dialogue_samples",
            collection="dialogue_samples_kb",
            columns=cols,
            query=query,
            filters=filters,
            limit=limit,
        )

    # 描写样本（混合检索）
    if not technique_type or technique_type == "description":
        cols = ["id", "book_name", "chapter_id", "description_type",
                "original_text", "technique_analysis", "sensory_details"]
        result_sections["description_samples"] = hybrid_search(
            table="description_samples",
            collection="description_samples_kb",
            columns=cols,
            query=query,
            filters=filters,
            limit=limit,
        )

    # 转场样本（混合检索）
    if not technique_type or technique_type == "transition":
        cols = ["id", "book_name", "chapter_id", "transition_type",
                "original_text", "technique_analysis"]
        result_sections["transition_samples"] = hybrid_search(
            table="transition_samples",
            collection="transition_samples_kb",
            columns=cols,
            query=query,
            filters=filters,
            limit=limit,
        )

    # 叙事距离控制（纯 SQL，无向量集合）
    if not technique_type or technique_type == "narrative_distance":
        sql = "SELECT * FROM narrative_distance WHERE 1=1"
        params = []
        if book_name:
            sql += " AND book_name = ?"
            params.append(book_name)
        sql += f" LIMIT {limit}"
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        cols = ["id", "book_name", "chapter_id", "distance_type",
                "trigger_reason", "original_example"]
        result_sections["narrative_distance"] = [dict(zip(cols, r)) for r in rows]

    # Show vs Tell 策略（纯 SQL，无向量集合）
    if not technique_type or technique_type == "show_tell":
        sql = "SELECT * FROM show_tell_patterns WHERE 1=1"
        params = []
        if book_name:
            sql += " AND book_name = ?"
            params.append(book_name)
        sql += f" LIMIT {limit}"
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        cols = ["id", "book_name", "chapter_id", "pattern_type",
                "ratio_estimate", "switching_triggers", "original_example"]
        result_sections["show_tell_patterns"] = [dict(zip(cols, r)) for r in rows]

    # 风格总结（纯 SQL）
    sql = "SELECT * FROM style_summaries WHERE 1=1"
    params = []
    if book_name:
        sql += " AND book_name = ?"
        params.append(book_name)
    sql += f" LIMIT {limit}"
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    cols = ["id", "book_name", "category", "summary_type",
            "scene_or_desc_type", "style_description", "key_features"]
    result_sections["style_summaries"] = [dict(zip(cols, r)) for r in rows]

    # 类型特定技法（纯 SQL）
    sql = "SELECT * FROM genre_specific_techniques WHERE 1=1"
    params = []
    if book_name:
        sql += " AND book_name = ?"
        params.append(book_name)
    sql += f" LIMIT {limit}"
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    cols = ["id", "book_name", "genre_tag", "technique_name", "technique_category",
            "analysis", "original_example", "applicable_scenarios"]
    result_sections["genre_specific_techniques"] = [dict(zip(cols, r)) for r in rows]

    # 恐惧/氛围构建链（纯 SQL）
    sql = "SELECT * FROM fear_building WHERE 1=1"
    params = []
    if book_name:
        sql += " AND book_name = ?"
        params.append(book_name)
    sql += f" LIMIT {limit}"
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    cols = ["id", "book_name", "fear_type", "building_steps_json",
            "atmosphere_techniques_json", "climax_moment", "original_example"]
    fear = []
    for row in rows:
        item = dict(zip(cols, row))
        for json_field in ["building_steps_json", "atmosphere_techniques_json"]:
            key = json_field.replace("_json", "")
            try:
                item[key] = json.loads(item.get(json_field, "[]"))
            except Exception:
                item[key] = []
        fear.append(item)
    result_sections["fear_building"] = fear

    total = sum(len(v) for v in result_sections.values())
    _log_search("default", "style", query or technique_type or "", total)

    return {
        "success": True,
        "data": result_sections,
        "total": total,
    }


# ===================== 大纲/结构搜索 =====================

@router.get("/search/plot")
def search_plot(
    query: Optional[str] = Query(None, description="语义搜索关键词（如：三幕结构设计/冲突升级方式）"),
    structure_type: Optional[str] = Query(None, description="结构类型过滤（三幕结构/英雄之旅/多线交织）"),
    book_name: Optional[str] = Query(None, description="限定书名"),
    limit: int = Query(10, ge=1, le=50, description="返回数量"),
):
    """
    大纲/结构搜索

    聚合全书结构、主线支线、卷大纲、高潮构建链、冲突升级阶梯、信息管理策略。
    适用场景：设计大纲时，搜索标杆作品的结构设计和剧情编排。
    """
    db = get_db_manager()
    cursor = db.connect().cursor()
    result_sections = {}

    # 全书结构
    sql = "SELECT * FROM book_structure WHERE 1=1"
    params = []
    if book_name:
        sql += " AND book_name = ?"
        params.append(book_name)
    if structure_type:
        sql += " AND structure_type LIKE ?"
        params.append(f"%{structure_type}%")
    sql += f" LIMIT {limit}"
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    cols = ["id", "book_name", "structure_type", "act_breakdown_json", "surface_theme", "deep_theme"]
    structures = []
    for row in rows:
        item = dict(zip(cols, row))
        try:
            item["act_breakdown"] = json.loads(item.pop("act_breakdown_json", "[]"))
        except Exception:
            item["act_breakdown"] = []
        structures.append(item)
    result_sections["book_structure"] = structures

    # 主线支线
    sql = "SELECT * FROM plot_lines WHERE 1=1"
    params = []
    if book_name:
        sql += " AND book_name = ?"
        params.append(book_name)
    sql += f" LIMIT {limit}"
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    cols = ["id", "book_name", "line_type", "theme", "chapter_distribution", "milestones_json"]
    plot_lines = []
    for row in rows:
        item = dict(zip(cols, row))
        try:
            item["milestones"] = json.loads(item.pop("milestones_json", "[]"))
        except Exception:
            item["milestones"] = []
        plot_lines.append(item)
    result_sections["plot_lines"] = plot_lines

    # 卷大纲
    sql = "SELECT * FROM macro_outlines WHERE 1=1"
    params = []
    if book_name:
        sql += " AND book_name = ?"
        params.append(book_name)
    sql += " ORDER BY volume_index"
    sql += f" LIMIT {limit}"
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    cols = ["id", "book_name", "category", "volume_index", "chapter_range",
            "theme", "conflict", "beats_json", "arc"]
    outlines = []
    for row in rows:
        item = dict(zip(cols, row))
        try:
            item["beats"] = json.loads(item.pop("beats_json", "[]"))
        except Exception:
            item["beats"] = []
        outlines.append(item)
    result_sections["macro_outlines"] = outlines

    # 高潮构建链
    sql = "SELECT * FROM climax_buildup_chains WHERE 1=1"
    params = []
    if book_name:
        sql += " AND book_name = ?"
        params.append(book_name)
    sql += f" LIMIT {limit}"
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    cols = ["id", "book_name", "climax_name", "climax_chapter", "buildup_steps_json", "tension_escalation"]
    climax_chains = []
    for row in rows:
        item = dict(zip(cols, row))
        try:
            item["buildup_steps"] = json.loads(item.pop("buildup_steps_json", "[]"))
        except Exception:
            item["buildup_steps"] = []
        climax_chains.append(item)
    result_sections["climax_buildup_chains"] = climax_chains

    # 冲突升级阶梯
    sql = "SELECT * FROM conflict_escalation WHERE 1=1"
    params = []
    if book_name:
        sql += " AND book_name = ?"
        params.append(book_name)
    sql += f" LIMIT {limit}"
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    cols = ["id", "book_name", "conflict_line", "escalation_steps_json", "escalation_pattern"]
    conflict_esc = []
    for row in rows:
        item = dict(zip(cols, row))
        try:
            item["escalation_steps"] = json.loads(item.pop("escalation_steps_json", "[]"))
        except Exception:
            item["escalation_steps"] = []
        conflict_esc.append(item)
    result_sections["conflict_escalation"] = conflict_esc

    # 感情线追踪
    sql = "SELECT * FROM romance_lines WHERE 1=1"
    params = []
    if book_name:
        sql += " AND book_name = ?"
        params.append(book_name)
    sql += f" LIMIT {limit}"
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    cols = ["id", "book_name", "couple_a", "couple_b", "line_type",
            "development_stages_json", "sweet_points_json", "angst_points_json",
            "interaction_patterns_json", "resolution"]
    romance = []
    for row in rows:
        item = dict(zip(cols, row))
        for json_field in ["development_stages_json", "sweet_points_json", "angst_points_json", "interaction_patterns_json"]:
            key = json_field.replace("_json", "")
            try:
                item[key] = json.loads(item.get(json_field, "[]"))
            except Exception:
                item[key] = []
        romance.append(item)
    result_sections["romance_lines"] = romance

    # 线索与推理链
    sql = "SELECT * FROM mystery_clues WHERE 1=1"
    params = []
    if book_name:
        sql += " AND book_name = ?"
        params.append(book_name)
    sql += f" LIMIT {limit}"
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    cols = ["id", "book_name", "clue_name", "clue_type", "planted_chapter",
            "payoff_chapter", "red_herring", "misdirection_method",
            "reasoning_chain_json", "twist_design"]
    mystery = []
    for row in rows:
        item = dict(zip(cols, row))
        try:
            item["reasoning_chain"] = json.loads(item.pop("reasoning_chain_json", "[]"))
        except Exception:
            item["reasoning_chain"] = []
        mystery.append(item)
    result_sections["mystery_clues"] = mystery

    # 信息管理策略
    sql = "SELECT * FROM information_management WHERE 1=1"
    params = []
    if book_name:
        sql += " AND book_name = ?"
        params.append(book_name)
    sql += f" LIMIT {limit}"
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    cols = ["id", "book_name", "strategy_type", "target_info",
            "conceal_method", "reveal_timing", "dramatic_purpose"]
    result_sections["information_management"] = [dict(zip(cols, r)) for r in rows]

    total = sum(len(v) for v in result_sections.values())
    _log_search("default", "plot", query or structure_type or "", total)

    return {
        "success": True,
        "data": {"structured": result_sections},
        "total": total,
    }


# ===================== 正文样本搜索 =====================

@router.get("/search/excerpt")
def search_excerpt(
    query: Optional[str] = Query(None, description="语义搜索关键词（如：打脸高潮点写法/告白场景对话）"),
    sample_type: Optional[str] = Query(
        None,
        description="样本类型过滤（dialogue/description/transition）",
    ),
    scene_type: Optional[str] = Query(None, description="场景类型过滤（争吵/告白/打斗/环境）"),
    book_name: Optional[str] = Query(None, description="限定书名"),
    limit: int = Query(10, ge=1, le=50, description="返回数量"),
):
    """
    正文样本搜索

    聚合对话样本、描写样本、转场样本，支持按场景类型过滤。
    适用场景：创作正文时，搜索标杆作品的范文段落作为参考。
    """
    db = get_db_manager()
    cursor = db.connect().cursor()
    result_sections = {}

    # 对话样本
    if not sample_type or sample_type == "dialogue":
        sql = "SELECT * FROM dialogue_samples WHERE 1=1"
        params = []
        if book_name:
            sql += " AND book_name = ?"
            params.append(book_name)
        if scene_type:
            sql += " AND scene_type LIKE ?"
            params.append(f"%{scene_type}%")
        sql += f" LIMIT {limit}"
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        cols = ["id", "book_name", "chapter_id", "scene_type",
                "original_text", "emotional_tension", "subtext", "plot_function"]
        result_sections["dialogue_samples"] = [dict(zip(cols, r)) for r in rows]

    # 描写样本
    if not sample_type or sample_type == "description":
        sql = "SELECT * FROM description_samples WHERE 1=1"
        params = []
        if book_name:
            sql += " AND book_name = ?"
            params.append(book_name)
        if scene_type:
            sql += " AND description_type LIKE ?"
            params.append(f"%{scene_type}%")
        sql += f" LIMIT {limit}"
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        cols = ["id", "book_name", "chapter_id", "description_type",
                "original_text", "technique_analysis", "sensory_details"]
        result_sections["description_samples"] = [dict(zip(cols, r)) for r in rows]

    # 转场样本
    if not sample_type or sample_type == "transition":
        sql = "SELECT * FROM transition_samples WHERE 1=1"
        params = []
        if book_name:
            sql += " AND book_name = ?"
            params.append(book_name)
        sql += f" LIMIT {limit}"
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        cols = ["id", "book_name", "chapter_id", "transition_type",
                "original_text", "technique_analysis"]
        result_sections["transition_samples"] = [dict(zip(cols, r)) for r in rows]

    # 动作/战斗场景范文
    if not sample_type or sample_type == "action":
        sql = "SELECT * FROM action_scene_samples WHERE 1=1"
        params = []
        if book_name:
            sql += " AND book_name = ?"
            params.append(book_name)
        if scene_type:
            sql += " AND action_type LIKE ?"
            params.append(f"%{scene_type}%")
        sql += f" LIMIT {limit}"
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        cols = ["id", "book_name", "chapter_id", "action_type",
                "original_text", "technique_analysis", "pacing_analysis", "sensory_details"]
        result_sections["action_scene_samples"] = [dict(zip(cols, r)) for r in rows]

    # 高潮段落/名场面原文
    if not sample_type or sample_type == "climax":
        sql = "SELECT * FROM climax_excerpts WHERE 1=1"
        params = []
        if book_name:
            sql += " AND book_name = ?"
            params.append(book_name)
        if scene_type:
            sql += " AND excerpt_type LIKE ?"
            params.append(f"%{scene_type}%")
        sql += f" LIMIT {limit}"
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        cols = ["id", "book_name", "chapter_id", "excerpt_type",
                "original_text", "technique_analysis", "emotional_impact"]
        result_sections["climax_excerpts"] = [dict(zip(cols, r)) for r in rows]

    # 章节开头/结尾范文
    if not sample_type or sample_type == "opening_ending":
        sql = "SELECT * FROM chapter_opening_ending_samples WHERE 1=1"
        params = []
        if book_name:
            sql += " AND book_name = ?"
            params.append(book_name)
        sql += f" LIMIT {limit}"
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        cols = ["id", "book_name", "chapter_id", "sample_position",
                "original_text", "technique_analysis", "hook_type"]
        result_sections["chapter_opening_ending_samples"] = [dict(zip(cols, r)) for r in rows]

    # 金句/名句
    if not sample_type or sample_type == "quotes":
        sql = "SELECT * FROM memorable_quotes WHERE 1=1"
        params = []
        if book_name:
            sql += " AND book_name = ?"
            params.append(book_name)
        if scene_type:
            sql += " AND quote_type LIKE ?"
            params.append(f"%{scene_type}%")
        sql += f" LIMIT {limit}"
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        cols = ["id", "book_name", "chapter_id", "quote_text",
                "context", "technique_analysis", "quote_type"]
        result_sections["memorable_quotes"] = [dict(zip(cols, r)) for r in rows]

    # 语义搜索
    semantic_results = []
    if query:
        chroma = get_chroma_manager()
        where_filter = {"book_name": book_name} if book_name else None
        for collection_name in ["dialogue_samples_kb", "description_samples_kb", "transition_samples_kb", "action_scene_samples_kb", "climax_excerpts_kb", "memorable_quotes_kb"]:
            chroma_res = chroma.query(
                collection_name,
                query_texts=[query],
                n_results=limit,
                where=where_filter,
            )
            if chroma_res and chroma_res.get("ids"):
                for i, doc_id in enumerate(chroma_res["ids"][0]):
                    semantic_results.append({
                        "id": doc_id,
                        "source": collection_name,
                        "text": chroma_res["documents"][0][i] if chroma_res.get("documents") else "",
                        "metadata": chroma_res["metadatas"][0][i] if chroma_res.get("metadatas") else {},
                    })

    total = sum(len(v) for v in result_sections.values()) + len(semantic_results)
    _log_search("default", "excerpt", query or sample_type or "", total)

    return {
        "success": True,
        "data": {
            "structured": result_sections,
            "semantic": semantic_results,
        },
        "total": total,
    }


# ===================== 综合语义搜索 =====================

@router.post("/search/comprehensive")
def search_comprehensive(req: ComprehensiveSearchRequest):
    """
    综合语义搜索

    发送一段创作内容文本，按指定维度返回最相关的标杆知识。
    适用场景：创作 skill 生成初稿后，发送初稿文本让知识库返回最相似的标杆内容供参考。
    """
    chroma = get_chroma_manager()
    db = get_db_manager()
    cursor = db.connect().cursor()
    dimension_results = {}

    # 维度 → ChromaDB collection 映射
    dimension_collections = {
        "world": ["world_settings_kb"],
        "character": ["character_profiles_kb"],
        "style": ["dialogue_samples_kb", "description_samples_kb"],
        "plot": [],
        "excerpt": ["dialogue_samples_kb", "description_samples_kb", "transition_samples_kb"],
    }

    for dimension in req.dimensions:
        collections = dimension_collections.get(dimension, [])
        dim_items = []

        # ChromaDB 语义搜索
        for collection_name in collections:
            chroma_res = chroma.query(
                collection_name,
                query_texts=[req.text],
                n_results=req.limit,
            )
            if chroma_res and chroma_res.get("ids"):
                for i, doc_id in enumerate(chroma_res["ids"][0]):
                    dim_items.append({
                        "id": doc_id,
                        "source": collection_name,
                        "text": chroma_res["documents"][0][i] if chroma_res.get("documents") else "",
                        "metadata": chroma_res["metadatas"][0][i] if chroma_res.get("metadatas") else {},
                    })

        # plot 维度走结构化查询（基于关键词提取）
        if dimension == "plot":
            cursor.execute(
                "SELECT book_name, structure_type, surface_theme, deep_theme FROM book_structure LIMIT ?",
                (req.limit,),
            )
            rows = cursor.fetchall()
            for row in rows:
                dim_items.append({
                    "source": "book_structure",
                    "text": f"结构:{row[1]} | 表层主题:{row[2]} | 深层主题:{row[3]}",
                    "metadata": {"book_name": row[0]},
                })

        dimension_results[dimension] = dim_items

    total = sum(len(v) for v in dimension_results.values())
    _log_search("default", "comprehensive", req.text[:200], total)

    return {
        "success": True,
        "data": dimension_results,
        "total": total,
    }


# ===================== 按书名检索全部知识 =====================

@router.get("/search/by-book")
def search_by_book(
    book_name: str = Query(..., description="书名"),
):
    """
    按书名检索全部知识

    返回该书在知识库中的所有维度数据概览。
    适用场景：想深入了解某本标杆作品的全貌。
    """
    db = get_db_manager()
    cursor = db.connect().cursor()
    overview = {"book_name": book_name, "sections": {}}

    # 世界观
    cursor.execute("SELECT module, entity FROM world_settings WHERE book_name=? LIMIT 20", (book_name,))
    overview["sections"]["world_settings"] = [{"module": r[0], "entity": r[1]} for r in cursor.fetchall()]

    # 人物
    cursor.execute("SELECT name, role_type FROM character_profiles WHERE book_name=? LIMIT 20", (book_name,))
    overview["sections"]["character_profiles"] = [{"name": r[0], "role_type": r[1]} for r in cursor.fetchall()]

    # 全书结构
    cursor.execute("SELECT structure_type, surface_theme, deep_theme FROM book_structure WHERE book_name=? LIMIT 1", (book_name,))
    row = cursor.fetchone()
    if row:
        overview["sections"]["book_structure"] = {"structure_type": row[0], "surface_theme": row[1], "deep_theme": row[2]}

    # 主线
    cursor.execute("SELECT theme FROM plot_lines WHERE book_name=? AND line_type='main' LIMIT 1", (book_name,))
    row = cursor.fetchone()
    if row:
        overview["sections"]["main_plot"] = {"theme": row[0]}

    # 卷大纲数量
    cursor.execute("SELECT COUNT(*) FROM macro_outlines WHERE book_name=?", (book_name,))
    overview["sections"]["volume_count"] = cursor.fetchone()[0]

    # 对话样本数量
    cursor.execute("SELECT COUNT(*) FROM dialogue_samples WHERE book_name=?", (book_name,))
    overview["sections"]["dialogue_sample_count"] = cursor.fetchone()[0]

    # 描写样本数量
    cursor.execute("SELECT COUNT(*) FROM description_samples WHERE book_name=?", (book_name,))
    overview["sections"]["description_sample_count"] = cursor.fetchone()[0]

    # 势力网络
    cursor.execute("SELECT faction_a, faction_b, relation_type FROM faction_networks WHERE book_name=? LIMIT 10", (book_name,))
    overview["sections"]["faction_networks"] = [
        {"faction_a": r[0], "faction_b": r[1], "relation_type": r[2]} for r in cursor.fetchall()
    ]

    _log_search("default", "by_book", book_name, 1)

    return {"success": True, "data": overview}


# ===================== 正文质量评审 =====================

@router.post("/review")
def review_chapter(req: ReviewRequest):
    """
    正文质量评审（对标知识库标杆）

    对章节正文进行多维度评审：节奏/对话/描写/人物/情节，
    与知识库标杆作品对标分析，输出打分+问题标记+修改建议+改写示范。
    """
    from stages.stage_j import StageJ
    stage = StageJ()
    result = stage.run(
        chapter_text=req.chapter_text,
        project_name=req.project_name,
        chapter_index=req.chapter_index,
        benchmark_books=req.benchmark_books,
    )
    stage.insert(result)
    return {"success": True, "data": result}


@router.get("/review/{project_name}/{chapter_index}")
def get_review_result(project_name: str, chapter_index: int):
    """查询评审结果"""
    db = get_db_manager()
    cursor = db.connect().cursor()
    cursor.execute(
        "SELECT * FROM chapter_reviews WHERE project_name = ? AND chapter_index = ? ORDER BY reviewed_at DESC LIMIT 1",
        (project_name, chapter_index),
    )
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="评审结果不存在")
    columns = ["id", "project_name", "chapter_index", "overall_score",
                "dimension_scores_json", "issues_json", "suggestions_json",
                "rewrite_samples_json", "benchmark_books", "reviewed_at"]
    item = dict(zip(columns, row))
    for json_field in ["dimension_scores_json", "issues_json", "suggestions_json", "rewrite_samples_json"]:
        key = json_field.replace("_json", "")
        try:
            item[key] = json.loads(item.get(json_field, "{}" if "scores" in json_field else "[]"))
        except Exception:
            item[key] = {} if "scores" in json_field else []
    return {"success": True, "data": item}


# ===================== 知识库引用推荐 =====================

@router.post("/recommend")
def recommend_references(req: RecommendRequest):
    """
    知识库引用推荐（按题材匹配）

    根据创作项目的题材/类型，从知识库中检索最相关的标杆作品，
    按世界观/人物/大纲/风格各维度推荐参考素材。
    """
    from stages.stage_k import StageK
    stage = StageK()
    result = stage.run(
        project_name=req.project_name,
        genre=req.genre,
        premise=req.premise,
        target_dimensions=req.target_dimensions,
    )
    stage.insert(result)
    return {"success": True, "data": result}


# ===================== 知识图谱查询 =====================

@router.get("/graph/characters")
def query_character_graph(
    character_name: Optional[str] = Query(None, description="人物名称（精确匹配）"),
    book_name: Optional[str] = Query(None, description="限定书名"),
    depth: int = Query(2, ge=1, le=5, description="关系深度（1-5跳）"),
):
    """
    知识图谱：人物关系查询

    查询指定人物在知识图谱中的关系网络，包括与其他人物的共同出场、互动关系等。
    适用场景：了解人物关系网络、分析人物互动模式。
    """
    try:
        graph_mgr = get_graph_manager()
        graph = graph_mgr.load()
        
        results = []
        
        # 查找匹配的人物节点
        target_nodes = []
        for node_id, attrs in graph.nodes(data=True):
            if attrs.get("node_type") == "character":
                if character_name and character_name in node_id:
                    target_nodes.append(node_id)
                elif not character_name:
                    if book_name:
                        book_list = attrs.get("book_list", "")
                        if book_name in book_list:
                            target_nodes.append(node_id)
                    else:
                        target_nodes.append(node_id)
        
        # 对每个目标节点，提取关系网络
        for target_node in target_nodes[:10]:  # 限制返回数量
            node_data = {
                "node_id": target_node,
                "attributes": dict(graph.nodes[target_node]),
                "relationships": [],
            }
            
            # BFS 提取指定深度的关系
            visited = {target_node}
            current_level = [target_node]
            
            for current_depth in range(depth):
                next_level = []
                for node in current_level:
                    # 出边
                    for neighbor in graph.successors(node):
                        if neighbor not in visited:
                            edge_data = graph[node][neighbor]
                            node_data["relationships"].append({
                                "source": node,
                                "target": neighbor,
                                "direction": "out",
                                "depth": current_depth + 1,
                                "attributes": dict(edge_data),
                            })
                            visited.add(neighbor)
                            next_level.append(neighbor)
                    
                    # 入边
                    for predecessor in graph.predecessors(node):
                        if predecessor not in visited:
                            edge_data = graph[predecessor][node]
                            node_data["relationships"].append({
                                "source": predecessor,
                                "target": node,
                                "direction": "in",
                                "depth": current_depth + 1,
                                "attributes": dict(edge_data),
                            })
                            visited.add(predecessor)
                            next_level.append(predecessor)
                
                current_level = next_level
                if not current_level:
                    break
            
            results.append(node_data)
        
        return {
            "success": True,
            "data": {
                "total_characters": len(results),
                "characters": results,
            },
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@router.get("/graph/chapters")
def query_chapter_graph(
    book_name: str = Query(..., description="书名"),
    chapter_range: Optional[str] = Query(None, description="章节范围（如：1-50）"),
):
    """
    知识图谱：章节关联查询

    查询指定书籍的章节关联图谱，包括章节间的人物状态传递、情节延续等。
    适用场景：分析章节间的叙事连贯性、人物状态演变。
    """
    try:
        graph_mgr = get_graph_manager()
        graph = graph_mgr.load()
        
        # 解析章节范围
        start_chapter = 1
        end_chapter = 999999
        if chapter_range and "-" in chapter_range:
            parts = chapter_range.split("-")
            start_chapter = int(parts[0])
            end_chapter = int(parts[1])
        
        # 查找章节节点
        chapter_nodes = []
        for node_id, attrs in graph.nodes(data=True):
            if attrs.get("node_type") == "chapter":
                book_list = attrs.get("book_list", "")
                if book_name in book_list:
                    # 提取章节号
                    chapter_num = attrs.get("chapter_index", 0)
                    if start_chapter <= chapter_num <= end_chapter:
                        chapter_nodes.append(node_id)
        
        # 提取章节间的边
        edges = []
        for node in chapter_nodes:
            for neighbor in graph.successors(node):
                if neighbor in chapter_nodes:
                    edge_data = graph[node][neighbor]
                    edges.append({
                        "source": node,
                        "target": neighbor,
                        "attributes": dict(edge_data),
                    })
        
        return {
            "success": True,
            "data": {
                "book_name": book_name,
                "chapter_range": f"{start_chapter}-{end_chapter}",
                "total_chapters": len(chapter_nodes),
                "total_edges": len(edges),
                "chapters": chapter_nodes,
                "edges": edges,
            },
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@router.get("/graph/stats")
def graph_statistics():
    """
    知识图谱统计信息

    返回知识图谱的整体统计：节点数、边数、连通分量等。
    适用场景：了解知识库的整体规模和结构。
    """
    try:
        graph_mgr = get_graph_manager()
        graph = graph_mgr.load()
        
        # 基础统计
        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()
        
        # 按类型统计节点
        node_types = {}
        for node_id, attrs in graph.nodes(data=True):
            node_type = attrs.get("node_type", "unknown")
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        # 按书名统计
        book_stats = {}
        for node_id, attrs in graph.nodes(data=True):
            book_list = attrs.get("book_list", "")
            if book_list:
                for book in book_list.split("|"):
                    book = book.strip()
                    if book:
                        book_stats[book] = book_stats.get(book, 0) + 1
        
        # 连通分量（无向图）
        undirected = graph.to_undirected()
        components = list(nx.connected_components(undirected))
        
        return {
            "success": True,
            "data": {
                "total_nodes": node_count,
                "total_edges": edge_count,
                "node_types": node_types,
                "books": book_stats,
                "connected_components": len(components),
                "largest_component_size": max(len(c) for c in components) if components else 0,
            },
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


# ===================== 搜索历史 =====================

# ===================== 感情线搜索 =====================

@router.get("/search/romance")
def search_romance(
    query: Optional[str] = Query(None, description="语义搜索关键词（如：虐恋/欢喜冤家/暗恋）"),
    couple_a: Optional[str] = Query(None, description="CP角色A姓名模糊匹配"),
    couple_b: Optional[str] = Query(None, description="CP角色B姓名模糊匹配"),
    line_type: Optional[str] = Query(None, description="感情线类型过滤（主CP/副CP/暗恋/单恋）"),
    book_name: Optional[str] = Query(None, description="限定书名"),
    limit: int = Query(10, ge=1, le=50, description="返回数量"),
):
    """
    感情线搜索

    搜索标杆作品的感情线设计，包括CP配对、发展阶段、甜点/虐点、互动模式。
    适用场景：创作言情/爱情线时，参考标杆作品的感情线编排。
    """
    db = get_db_manager()
    cursor = db.connect().cursor()

    sql = "SELECT * FROM romance_lines WHERE 1=1"
    params = []
    if book_name:
        sql += " AND book_name = ?"
        params.append(book_name)
    if couple_a:
        sql += " AND (couple_a LIKE ? OR couple_b LIKE ?)"
        params.extend([f"%{couple_a}%", f"%{couple_a}%"])
    if couple_b:
        sql += " AND (couple_a LIKE ? OR couple_b LIKE ?)"
        params.extend([f"%{couple_b}%", f"%{couple_b}%"])
    if line_type:
        sql += " AND line_type LIKE ?"
        params.append(f"%{line_type}%")
    sql += f" LIMIT {limit}"
    cursor.execute(sql, params)
    rows = cursor.fetchall()

    cols = ["id", "book_name", "couple_a", "couple_b", "line_type",
            "development_stages_json", "sweet_points_json", "angst_points_json",
            "interaction_patterns_json", "resolution"]
    results = []
    for row in rows:
        item = dict(zip(cols, row))
        for json_field in ["development_stages_json", "sweet_points_json", "angst_points_json", "interaction_patterns_json"]:
            key = json_field.replace("_json", "")
            try:
                item[key] = json.loads(item.get(json_field, "[]"))
            except Exception:
                item[key] = []
        results.append(item)

    _log_search("default", "romance", query or couple_a or "", len(results))
    return {"success": True, "data": results, "total": len(results)}


# ===================== 线索/推理搜索 =====================

@router.get("/search/mystery")
def search_mystery(
    query: Optional[str] = Query(None, description="语义搜索关键词（如：红鲱鱼/反转设计/推理链）"),
    clue_type: Optional[str] = Query(None, description="线索类型过滤（关键线索/红鲱鱼/辅助线索）"),
    book_name: Optional[str] = Query(None, description="限定书名"),
    limit: int = Query(10, ge=1, le=50, description="返回数量"),
):
    """
    线索/推理搜索

    搜索标杆作品的线索设计、误导手法、推理链和反转设计。
    适用场景：创作悬疑/推理类作品时，参考标杆作品的线索编排。
    """
    db = get_db_manager()
    cursor = db.connect().cursor()

    sql = "SELECT * FROM mystery_clues WHERE 1=1"
    params = []
    if book_name:
        sql += " AND book_name = ?"
        params.append(book_name)
    if clue_type:
        sql += " AND clue_type LIKE ?"
        params.append(f"%{clue_type}%")
    sql += f" LIMIT {limit}"
    cursor.execute(sql, params)
    rows = cursor.fetchall()

    cols = ["id", "book_name", "clue_name", "clue_type", "planted_chapter",
            "payoff_chapter", "red_herring", "misdirection_method",
            "reasoning_chain_json", "twist_design"]
    results = []
    for row in rows:
        item = dict(zip(cols, row))
        try:
            item["reasoning_chain"] = json.loads(item.pop("reasoning_chain_json", "[]"))
        except Exception:
            item["reasoning_chain"] = []
        results.append(item)

    _log_search("default", "mystery", query or clue_type or "", len(results))
    return {"success": True, "data": results, "total": len(results)}


# ===================== 升级体系搜索 =====================

@router.get("/search/progression")
def search_progression(
    query: Optional[str] = Query(None, description="语义搜索关键词（如：修真境界/游戏等级/职场晋升）"),
    system_type: Optional[str] = Query(None, description="体系类型过滤（修真境界/魔法等级/游戏等级/职场晋升/武道境界）"),
    book_name: Optional[str] = Query(None, description="限定书名"),
    limit: int = Query(10, ge=1, le=50, description="返回数量"),
):
    """
    升级/成长体系搜索

    搜索标杆作品的升级体系设计，包括境界层级、升级条件、实力对比、成长里程碑。
    适用场景：创作玄幻/仙侠/游戏竞技/职场类作品时，参考标杆作品的成长体系。
    """
    db = get_db_manager()
    cursor = db.connect().cursor()

    sql = "SELECT * FROM progression_systems WHERE 1=1"
    params = []
    if book_name:
        sql += " AND book_name = ?"
        params.append(book_name)
    if system_type:
        sql += " AND system_type LIKE ?"
        params.append(f"%{system_type}%")
    sql += f" LIMIT {limit}"
    cursor.execute(sql, params)
    rows = cursor.fetchall()

    cols = ["id", "book_name", "system_type", "levels_json", "upgrade_conditions_json",
            "power_comparison_json", "milestones_json", "growth_pattern"]
    results = []
    for row in rows:
        item = dict(zip(cols, row))
        for json_field in ["levels_json", "upgrade_conditions_json", "power_comparison_json", "milestones_json"]:
            key = json_field.replace("_json", "")
            try:
                item[key] = json.loads(item.get(json_field, "[]"))
            except Exception:
                item[key] = []
        results.append(item)

    _log_search("default", "progression", query or system_type or "", len(results))
    return {"success": True, "data": results, "total": len(results)}


# ===================== 类型技法搜索 =====================

@router.get("/search/genre-technique")
def search_genre_technique(
    query: Optional[str] = Query(None, description="语义搜索关键词（如：金手指设计/萌属性塑造/权谋博弈）"),
    genre_tag: Optional[str] = Query(None, description="类型标签过滤（网文/轻小说/历史/严肃文学/悬疑/言情）"),
    technique_category: Optional[str] = Query(None, description="技法分类过滤（设定设计/角色塑造/情节编排/叙事手法/氛围营造）"),
    book_name: Optional[str] = Query(None, description="限定书名"),
    limit: int = Query(10, ge=1, le=50, description="返回数量"),
):
    """
    类型特定技法搜索

    搜索标杆作品中该类型特有的写作技法（如网文的金手指设计、轻小说的萌属性塑造等）。
    适用场景：创作特定类型作品时，参考标杆作品的类型特有技法。
    """
    db = get_db_manager()
    cursor = db.connect().cursor()

    sql = "SELECT * FROM genre_specific_techniques WHERE 1=1"
    params = []
    if book_name:
        sql += " AND book_name = ?"
        params.append(book_name)
    if genre_tag:
        sql += " AND genre_tag LIKE ?"
        params.append(f"%{genre_tag}%")
    if technique_category:
        sql += " AND technique_category LIKE ?"
        params.append(f"%{technique_category}%")
    sql += f" LIMIT {limit}"
    cursor.execute(sql, params)
    rows = cursor.fetchall()

    cols = ["id", "book_name", "genre_tag", "technique_name", "technique_category",
            "analysis", "original_example", "applicable_scenarios"]
    results = [dict(zip(cols, r)) for r in rows]

    _log_search("default", "genre_technique", query or genre_tag or "", len(results))
    return {"success": True, "data": results, "total": len(results)}


# ===================== 书籍列表 =====================

@router.get("/books")
def list_books(
    category: Optional[str] = Query(None, description="分类过滤"),
    genre_tag: Optional[str] = Query(None, description="类型标签过滤（模糊匹配）"),
    author: Optional[str] = Query(None, description="作者模糊匹配"),
    limit: int = Query(50, ge=1, le=200, description="返回数量"),
):
    """
    书籍列表

    返回知识库中所有书籍的元数据，支持按分类/标签/作者筛选。
    适用场景：了解知识库中有哪些标杆作品，按类型筛选参考书。
    """
    db = get_db_manager()
    cursor = db.connect().cursor()

    sql = "SELECT * FROM book_metadata WHERE 1=1"
    params = []
    if category:
        sql += " AND category LIKE ?"
        params.append(f"%{category}%")
    if genre_tag:
        sql += " AND genre_tags LIKE ?"
        params.append(f"%{genre_tag}%")
    if author:
        sql += " AND author LIKE ?"
        params.append(f"%{author}%")
    sql += f" ORDER BY added_at DESC LIMIT {limit}"
    cursor.execute(sql, params)
    rows = cursor.fetchall()

    cols = ["id", "book_name", "author", "category", "genre_tags",
            "total_chapters", "total_words", "description", "added_at"]
    results = [dict(zip(cols, r)) for r in rows]

    return {"success": True, "data": results, "total": len(results)}


# ===================== 跨书对比分析 =====================

@router.post("/compare")
def cross_book_compare(req: CompareRequest):
    """
    跨书对比分析

    对比多本标杆作品在指定维度上的处理方式，提炼共同模式、各自特色和最佳实践。
    适用场景：想了解"顶尖作者们在某个问题上是怎么处理的"。
    """
    from stages.stage_l import StageL
    stage = StageL()
    result = stage.run(
        comparison_dimension=req.dimension,
        book_names=req.book_names,
        category=req.category,
    )
    stage.insert(result)
    return {"success": True, "data": result}


# ===================== 常见错误模式查询 =====================

@router.get("/mistakes")
def search_mistakes(
    query: Optional[str] = Query(None, description="语义搜索关键词"),
    dimension: Optional[str] = Query(None, description="维度过滤（节奏/对话/描写/人物/情节）"),
    mistake_name: Optional[str] = Query(None, description="错误名称模糊匹配"),
    limit: int = Query(10, ge=1, le=50, description="返回数量"),
):
    """
    常见错误模式查询

    查询知识库中归纳的常见写作错误模式，包括典型表现、修正方向和标杆范文。
    适用场景：创作时检查是否犯了常见错误，或学习如何避免。
    """
    db = get_db_manager()
    cursor = db.connect().cursor()

    sql = "SELECT * FROM common_mistakes WHERE 1=1"
    params = []
    if dimension:
        sql += " AND dimension LIKE ?"
        params.append(f"%{dimension}%")
    if mistake_name:
        sql += " AND mistake_name LIKE ?"
        params.append(f"%{mistake_name}%")
    sql += f" ORDER BY frequency DESC LIMIT {limit}"
    cursor.execute(sql, params)
    rows = cursor.fetchall()

    cols = ["id", "dimension", "mistake_name", "typical_manifestation",
            "frequency", "correction_direction", "benchmark_example",
            "benchmark_book", "created_at"]
    results = [dict(zip(cols, r)) for r in rows]

    # 语义搜索
    semantic_results = []
    if query:
        chroma = get_chroma_manager()
        chroma_res = chroma.query(
            "common_mistakes_kb",
            query_texts=[query],
            n_results=limit,
        )
        if chroma_res and chroma_res.get("ids"):
            for i, doc_id in enumerate(chroma_res["ids"][0]):
                semantic_results.append({
                    "id": doc_id,
                    "text": chroma_res["documents"][0][i] if chroma_res.get("documents") else "",
                    "metadata": chroma_res["metadatas"][0][i] if chroma_res.get("metadatas") else {},
                })

    _log_search("default", "mistakes", query or dimension or "", len(results) + len(semantic_results))
    return {
        "success": True,
        "data": {
            "structured": results,
            "semantic": semantic_results,
        },
        "total": len(results) + len(semantic_results),
    }


# ===================== 上下文感知推荐 =====================


def _query_table_generic(
    cursor,
    table: str,
    columns: List[str],
    query_text: str = "",
    search_fields: List[str] = None,
    limit: int = 3,
) -> List[Dict]:
    """通用表查询辅助函数"""
    if not search_fields:
        search_fields = columns[1:3]  # 默认搜索第2、3个字段
    
    sql = f"SELECT {', '.join(columns)} FROM {table}"
    params = []
    
    if query_text:
        conditions = [f"{field} LIKE ?" for field in search_fields]
        sql += f" WHERE {' OR '.join(conditions)}"
        params = [f"%{query_text}%"] * len(search_fields)
    
    sql += f" LIMIT {limit}"
    cursor.execute(sql, params)
    
    rows = cursor.fetchall()
    return [dict(zip(columns, row)) for row in rows]


@router.post("/context-push")
def context_aware_push(req: ContextPushRequest):
    """
    上下文感知推荐

    根据当前创作上下文，自动识别场景类型并推送最相关的标杆知识。
    适用场景：创作时不知道“该参考什么”，让知识库主动推荐。
    """
    from core.context_analyzer import get_context_analyzer
    
    analyzer = get_context_analyzer()
    
    # 1. 分析上下文，识别场景
    analysis = analyzer.analyze_context(
        context_text=req.context_text,
        creation_stage=req.creation_stage,
        genre=req.genre,
    )
    
    # 2. 根据查询策略并行查询相关知识
    db = get_db_manager()
    cursor = db.connect().cursor()
    strategy = analysis.get("query_strategy", {})
    
    recommended = {
        "excerpts": [],
        "techniques": [],
        "structure_tips": [],
        "common_mistakes": [],
    }
    
    # 查询范文
    excerpt_table_configs = {
        "dialogue_samples": {
            "columns": ["book_name", "scene_type", "original_text", "subtext"],
            "search_fields": ["original_text", "subtext"],
            "transform": lambda row: {"type": "dialogue", "book_name": row["book_name"], "scene_type": row["scene_type"], "text": (row["original_text"] or "")[:200], "subtext": row["subtext"]},
        },
        "description_samples": {
            "columns": ["book_name", "description_type", "original_text", "technique_analysis"],
            "search_fields": ["original_text", "technique_analysis"],
            "transform": lambda row: {"type": "description", "book_name": row["book_name"], "desc_type": row["description_type"], "text": (row["original_text"] or "")[:200], "analysis": row["technique_analysis"]},
        },
        "action_scene_samples": {
            "columns": ["book_name", "action_type", "original_text", "technique_analysis"],
            "search_fields": ["original_text", "action_type"],
            "transform": lambda row: {"type": "action", "book_name": row["book_name"], "action_type": row["action_type"], "text": (row["original_text"] or "")[:200], "analysis": row["technique_analysis"]},
        },
        "climax_excerpts": {
            "columns": ["book_name", "excerpt_type", "original_text", "technique_analysis"],
            "search_fields": ["original_text"],
            "transform": lambda row: {"type": "climax", "book_name": row["book_name"], "excerpt_type": row["excerpt_type"], "text": (row["original_text"] or "")[:200], "analysis": row["technique_analysis"]},
        },
        "world_settings": {
            "columns": ["book_name", "module", "entity", "content"],
            "search_fields": ["content", "entity"],
            "transform": lambda row: {"type": "world", "book_name": row["book_name"], "module": row["module"], "entity": row["entity"], "content": (row["content"] or "")[:200]},
        },
        "romance_lines": {
            "columns": ["book_name", "couple_a", "couple_b", "development_stages_json"],
            "search_fields": ["couple_a", "couple_b"],
            "transform": lambda row: {"type": "romance", "book_name": row["book_name"], "couple": f"{row['couple_a']} & {row['couple_b']}", "stages_count": len(json.loads(row["development_stages_json"])) if row["development_stages_json"] else 0},
        },
        "mystery_clues": {
            "columns": ["book_name", "clue_name", "clue_type", "misdirection_method"],
            "search_fields": ["clue_name", "misdirection_method"],
            "transform": lambda row: {"type": "mystery", "book_name": row["book_name"], "clue_name": row["clue_name"], "clue_type": row["clue_type"], "method": row["misdirection_method"]},
        },
    }

    for query_spec in strategy.get("excerpts", []):
        table = query_spec.get("table", "")
        query_text = query_spec.get("query", "")
        config = excerpt_table_configs.get(table)
        if not config:
            continue
        try:
            # climax_excerpts 支持按 excerpt_type 过滤
            search_text = query_text
            if table == "climax_excerpts" and query_spec.get("type"):
                search_text = query_spec["type"]
                config = {**config, "search_fields": ["excerpt_type"]}
            
            rows = _query_table_generic(cursor, table, config["columns"], search_text, config["search_fields"])
            for row in rows:
                recommended["excerpts"].append(config["transform"](row))
        except Exception:
            continue
    
    # 查询技法
    technique_table_configs = {
        "skills": {
            "columns": ["book_name", "skill_name", "analysis"],
            "search_fields": ["skill_name", "analysis"],
            "transform": lambda row: {"type": "skill", "book_name": row["book_name"], "skill_name": row["skill_name"], "analysis": (row["analysis"] or "")[:150]},
        },
        "character_speech_style": {
            "columns": ["book_name", "character_name", "catchphrases", "vocabulary_preference"],
            "search_fields": ["character_name", "catchphrases"],
            "transform": lambda row: {"type": "speech_style", "book_name": row["book_name"], "character": row["character_name"], "catchphrases": row["catchphrases"], "vocabulary": row["vocabulary_preference"]},
        },
        "climax_buildup_chains": {
            "columns": ["book_name", "climax_name", "buildup_steps_json", "tension_escalation"],
            "search_fields": ["climax_name", "tension_escalation"],
            "transform": lambda row: {"type": "buildup_chain", "book_name": row["book_name"], "climax_name": row["climax_name"], "steps_count": len(json.loads(row["buildup_steps_json"])) if row["buildup_steps_json"] else 0, "escalation": row["tension_escalation"]},
        },
        "technique_combinations": {
            "columns": ["book_name", "combo_name", "technique_sequence_json", "applicable_scenarios"],
            "search_fields": ["combo_name", "applicable_scenarios"],
            "transform": lambda row: {"type": "combo", "book_name": row["book_name"], "combo_name": row["combo_name"], "sequence_count": len(json.loads(row["technique_sequence_json"])) if row["technique_sequence_json"] else 0, "scenarios": row["applicable_scenarios"]},
        },
        "sensory_mappings": {
            "columns": ["book_name", "emotion", "show_not_tell", "analysis"],
            "search_fields": ["emotion", "analysis"],
            "transform": lambda row: {"type": "sensory", "book_name": row["book_name"], "emotion": row["emotion"], "show_not_tell": row["show_not_tell"], "analysis": (row["analysis"] or "")[:100]},
        },
        "information_management": {
            "columns": ["book_name", "strategy_type", "target_info", "dramatic_purpose"],
            "search_fields": ["target_info", "strategy_type"],
            "transform": lambda row: {"type": "info_management", "book_name": row["book_name"], "strategy": row["strategy_type"], "target_info": row["target_info"], "purpose": row["dramatic_purpose"]},
        },
        "genre_specific_techniques": {
            "columns": ["book_name", "technique_name", "analysis"],
            "search_fields": ["technique_name", "analysis"],
            "transform": lambda row: {"type": "genre_technique", "book_name": row["book_name"], "technique_name": row["technique_name"], "analysis": (row["analysis"] or "")[:150]},
        },
    }

    for query_spec in strategy.get("techniques", []):
        table = query_spec.get("table", "")
        query_text = query_spec.get("query", "")
        category = query_spec.get("category", "")
        scene_type = query_spec.get("scene_type", "")
        config = technique_table_configs.get(table)
        if not config:
            continue
        try:
            # technique_combinations 支持按 scene_type 过滤
            search_text = query_text
            search_fields = config["search_fields"]
            if table == "technique_combinations" and scene_type:
                search_text = scene_type
                search_fields = ["scene_type"]
            # genre_specific_techniques 支持按 category 过滤
            elif table == "genre_specific_techniques" and category:
                search_text = category
                search_fields = ["technique_category"]

            rows = _query_table_generic(cursor, table, config["columns"], search_text, search_fields)
            for row in rows:
                recommended["techniques"].append(config["transform"](row))
        except Exception:
            continue

    # 查询结构建议
    structure_table_configs = {
        "book_structure": {
            "columns": ["book_name", "structure_type", "surface_theme", "deep_theme"],
            "search_fields": ["surface_theme", "deep_theme"],
            "transform": lambda row: {"type": "structure", "book_name": row["book_name"], "structure_type": row["structure_type"], "surface_theme": row["surface_theme"], "deep_theme": row["deep_theme"]},
        },
        "macro_outlines": {
            "columns": ["book_name", "volume_index", "theme", "conflict"],
            "search_fields": ["theme", "conflict"],
            "transform": lambda row: {"type": "outline", "book_name": row["book_name"], "volume": row["volume_index"], "theme": row["theme"], "conflict": row["conflict"]},
        },
        "cross_book_comparisons": {
            "columns": ["comparison_dimension", "books_analyzed", "common_patterns_json", "best_practices"],
            "search_fields": ["comparison_dimension"],
            "transform": lambda row: {"type": "comparison", "dimension": row["comparison_dimension"], "books_count": len(json.loads(row["books_analyzed"])) if row["books_analyzed"] else 0, "patterns_count": len(json.loads(row["common_patterns_json"])) if row["common_patterns_json"] else 0, "best_practices": (row["best_practices"] or "")[:200]},
        },
    }

    for query_spec in strategy.get("structure_tips", []):
        table = query_spec.get("table", "")
        query_text = query_spec.get("query", "")
        dimension = query_spec.get("dimension", "")
        config = structure_table_configs.get(table)
        if not config:
            continue
        try:
            search_text = query_text
            if table == "cross_book_comparisons" and dimension:
                search_text = dimension
            rows = _query_table_generic(cursor, table, config["columns"], search_text, config["search_fields"])
            for row in rows:
                recommended["structure_tips"].append(config["transform"](row))
        except Exception:
            continue

    # 查询常见错误
    for query_spec in strategy.get("common_mistakes", []):
        dimension = query_spec.get("dimension", "")
        try:
            sql = "SELECT mistake_name, typical_manifestation, correction_direction FROM common_mistakes"
            params = []
            if dimension:
                sql += " WHERE dimension LIKE ?"
                params.append(f"%{dimension}%")
            sql += " ORDER BY frequency DESC LIMIT 3"
            cursor.execute(sql, params)
            for row in cursor.fetchall():
                recommended["common_mistakes"].append({
                    "mistake_name": row[0],
                    "manifestation": row[1],
                    "correction": row[2],
                })
        except Exception:
            continue
    
    _log_search(
        req.project_name or "default",
        "context_push",
        analysis.get("detected_scene", ""),
        sum(len(v) for v in recommended.values()),
    )
    
    return {
        "success": True,
        "data": {
            "detected_scene": analysis.get("detected_scene", ""),
            "detected_sub_type": analysis.get("detected_sub_type", ""),
            "detected_mood": analysis.get("detected_mood", ""),
            "recommended": recommended,
        },
    }


# ===================== 技法组合模板查询 =====================

@router.get("/combos")
def search_combos(
    query: Optional[str] = Query(None, description="语义搜索关键词"),
    scene_type: Optional[str] = Query(None, description="场景类型过滤（打斗/对话/描写/高潮/转折/揭秘）"),
    combo_name: Optional[str] = Query(None, description="组合名称模糊匹配"),
    limit: int = Query(10, ge=1, le=50, description="返回数量"),
):
    """
    技法组合模板查询

    查询标杆作品的技法组合模式，包括技法序列、每个技法的作用、适用场景和变体建议。
    适用场景：学习"一组技法如何组合使用"，而非单个技法。
    """
    db = get_db_manager()
    cursor = db.connect().cursor()

    sql = "SELECT * FROM technique_combinations WHERE 1=1"
    params = []
    if scene_type:
        sql += " AND scene_type LIKE ?"
        params.append(f"%{scene_type}%")
    if combo_name:
        sql += " AND combo_name LIKE ?"
        params.append(f"%{combo_name}%")
    sql += f" LIMIT {limit}"
    cursor.execute(sql, params)
    rows = cursor.fetchall()

    cols = ["id", "scene_type", "combo_name", "technique_sequence_json",
            "technique_roles_json", "applicable_scenarios", "variations",
            "benchmark_book", "original_example", "created_at"]
    results = []
    for row in rows:
        item = dict(zip(cols, row))
        for json_field in ["technique_sequence_json", "technique_roles_json"]:
            key = json_field.replace("_json", "")
            try:
                item[key] = json.loads(item.get(json_field, "[]"))
            except Exception:
                item[key] = []
        results.append(item)

    # 语义搜索
    semantic_results = []
    if query:
        chroma = get_chroma_manager()
        chroma_res = chroma.query(
            "technique_combinations_kb",
            query_texts=[query],
            n_results=limit,
        )
        if chroma_res and chroma_res.get("ids"):
            for i, doc_id in enumerate(chroma_res["ids"][0]):
                semantic_results.append({
                    "id": doc_id,
                    "text": chroma_res["documents"][0][i] if chroma_res.get("documents") else "",
                    "metadata": chroma_res["metadatas"][0][i] if chroma_res.get("metadatas") else {},
                })

    _log_search("default", "combos", query or scene_type or "", len(results) + len(semantic_results))
    return {
        "success": True,
        "data": {
            "structured": results,
            "semantic": semantic_results,
        },
        "total": len(results) + len(semantic_results),
    }


# ===================== 高潮段落/名场面搜索 =====================

@router.get("/search/climax")
def search_climax(
    query: Optional[str] = Query(None, description="语义搜索关键词（如：决战/揭秘/情感爆发）"),
    excerpt_type: Optional[str] = Query(None, description="高潮类型过滤（决战/揭秘/情感爆发/逆转/生死抉择）"),
    book_name: Optional[str] = Query(None, description="限定书名"),
    limit: int = Query(10, ge=1, le=50, description="返回数量"),
):
    """
    高潮段落/名场面搜索

    搜索标杆作品的高潮段落原文，包括技法分析和情感冲击力。
    适用场景：创作高潮段落时，参考标杆作品的名场面写法。
    """
    db = get_db_manager()
    cursor = db.connect().cursor()

    sql = "SELECT * FROM climax_excerpts WHERE 1=1"
    params = []
    if book_name:
        sql += " AND book_name = ?"
        params.append(book_name)
    if excerpt_type:
        sql += " AND excerpt_type LIKE ?"
        params.append(f"%{excerpt_type}%")
    sql += f" LIMIT {limit}"
    cursor.execute(sql, params)
    rows = cursor.fetchall()

    cols = ["id", "book_name", "chapter_id", "excerpt_type",
            "original_text", "technique_analysis", "emotional_impact"]
    results = [dict(zip(cols, r)) for r in rows]

    # 语义搜索
    semantic_results = []
    if query:
        chroma = get_chroma_manager()
        where_filter = {"book_name": book_name} if book_name else None
        chroma_res = chroma.query(
            "climax_excerpts_kb",
            query_texts=[query],
            n_results=limit,
            where=where_filter,
        )
        if chroma_res and chroma_res.get("ids"):
            for i, doc_id in enumerate(chroma_res["ids"][0]):
                semantic_results.append({
                    "id": doc_id,
                    "text": chroma_res["documents"][0][i] if chroma_res.get("documents") else "",
                    "metadata": chroma_res["metadatas"][0][i] if chroma_res.get("metadatas") else {},
                })

    _log_search("default", "climax", query or excerpt_type or "", len(results) + len(semantic_results))
    return {
        "success": True,
        "data": {
            "structured": results,
            "semantic": semantic_results,
        },
        "total": len(results) + len(semantic_results),
    }


# ===================== 金句/名句搜索 =====================

@router.get("/search/quotes")
def search_quotes(
    query: Optional[str] = Query(None, description="语义搜索关键词（如：哲理句/经典台词/励志金句）"),
    quote_type: Optional[str] = Query(None, description="金句类型过滤（哲理句/经典台词/情感金句/励志金句/讽刺金句）"),
    book_name: Optional[str] = Query(None, description="限定书名"),
    limit: int = Query(10, ge=1, le=50, description="返回数量"),
):
    """
    金句/名句搜索

    搜索标杆作品的金句/名句，包括上下文和技法分析。
    适用场景：创作时参考标杆作品的经典台词和哲理句。
    """
    db = get_db_manager()
    cursor = db.connect().cursor()

    sql = "SELECT * FROM memorable_quotes WHERE 1=1"
    params = []
    if book_name:
        sql += " AND book_name = ?"
        params.append(book_name)
    if quote_type:
        sql += " AND quote_type LIKE ?"
        params.append(f"%{quote_type}%")
    sql += f" LIMIT {limit}"
    cursor.execute(sql, params)
    rows = cursor.fetchall()

    cols = ["id", "book_name", "chapter_id", "quote_text",
            "context", "technique_analysis", "quote_type"]
    results = [dict(zip(cols, r)) for r in rows]

    # 语义搜索
    semantic_results = []
    if query:
        chroma = get_chroma_manager()
        where_filter = {"book_name": book_name} if book_name else None
        chroma_res = chroma.query(
            "memorable_quotes_kb",
            query_texts=[query],
            n_results=limit,
            where=where_filter,
        )
        if chroma_res and chroma_res.get("ids"):
            for i, doc_id in enumerate(chroma_res["ids"][0]):
                semantic_results.append({
                    "id": doc_id,
                    "text": chroma_res["documents"][0][i] if chroma_res.get("documents") else "",
                    "metadata": chroma_res["metadatas"][0][i] if chroma_res.get("metadatas") else {},
                })

    _log_search("default", "quotes", query or quote_type or "", len(results) + len(semantic_results))
    return {
        "success": True,
        "data": {
            "structured": results,
            "semantic": semantic_results,
        },
        "total": len(results) + len(semantic_results),
    }


# ===================== 内部辅助函数 =====================

def _log_search(project_name: str, search_type: str, query_text: str, result_count: int):
    """记录搜索历史"""
    try:
        db = get_db_manager()
        cursor = db.connect().cursor()
        log_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO search_logs VALUES (?,?,?,?,?,?,?)",
            (log_id, project_name, search_type,
             query_text[:500] if query_text else "",
             "", result_count, now),
        )
        db.commit()
    except Exception:
        pass


# ===================== 知识图谱扩展查询 =====================

@router.get("/graph/character-relations")
def query_character_relations(
    book_name: str = Query(..., description="书名"),
    character_name: Optional[str] = Query(None, description="人物名称（可选，精确匹配）"),
):
    """
    知识图谱：人物关系查询

    查询指定书籍中人物之间的关系网络，包括关系类型、关系强度等。
    适用场景：分析人物关系网络、理解人物互动模式。
    """
    try:
        graph_mgr = get_graph_manager()
        graph = graph_mgr.load()
        
        # 查找人物节点
        char_nodes = []
        for node_id, attrs in graph.nodes(data=True):
            if attrs.get("node_type") == "character":
                book_list = attrs.get("book_list", "")
                if book_name in book_list:
                    if character_name:
                        if character_name in node_id:
                            char_nodes.append(node_id)
                    else:
                        char_nodes.append(node_id)
        
        # 提取人物之间的边
        relations = []
        for node in char_nodes:
            for neighbor in graph.successors(node):
                if neighbor in char_nodes:
                    edge_data = graph[node][neighbor]
                    relations.append({
                        "source": node,
                        "target": neighbor,
                        "relation_type": edge_data.get("relation_type", "unknown"),
                        "relation_strength": edge_data.get("strength", 0),
                        "attributes": dict(edge_data),
                    })
        
        return {
            "success": True,
            "data": {
                "book_name": book_name,
                "total_characters": len(char_nodes),
                "total_relations": len(relations),
                "characters": char_nodes,
                "relations": relations,
            },
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@router.get("/graph/plot-lines")
def query_plot_lines(
    book_name: str = Query(..., description="书名"),
    line_type: Optional[str] = Query(None, description="剧情线类型（main/subplot/romance/mystery）"),
):
    """
    知识图谱：剧情线查询

    查询指定书籍的剧情线节点，包括主线、支线、感情线等。
    适用场景：分析剧情结构、理解剧情线交织方式。
    """
    try:
        graph_mgr = get_graph_manager()
        graph = graph_mgr.load()
        
        # 查找剧情线节点
        plot_nodes = []
        for node_id, attrs in graph.nodes(data=True):
            if attrs.get("node_type") == "plot_line":
                book_list = attrs.get("book_list", "")
                if book_name in book_list:
                    node_line_type = attrs.get("line_type", "")
                    if line_type:
                        if line_type == node_line_type:
                            plot_nodes.append({
                                "node_id": node_id,
                                "attributes": dict(attrs),
                            })
                    else:
                        plot_nodes.append({
                            "node_id": node_id,
                            "attributes": dict(attrs),
                        })
        
        return {
            "success": True,
            "data": {
                "book_name": book_name,
                "total_plot_lines": len(plot_nodes),
                "plot_lines": plot_nodes,
            },
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


# ===================== 新增聚合接口 =====================


class EnhanceRequest(BaseModel):
    """通用增强请求"""
    document_type: str = Field(
        ...,
        description="文档类型: world_setting/character_profile/outline/chapter_outline/chapter_text/writing_style",
    )
    content: str = Field(..., description="AI 生成的文档内容")
    genre: str = Field(default="", description="小说类型/题材（如：玄幻/都市/悬疑/言情）")
    specific_needs: str = Field(
        default="",
        description="具体需求描述（如：世界观缺少力量体系/人物动机不够深刻/大纲节奏太平均）",
    )
    limit: int = Field(default=10, ge=1, le=30, description="返回的参考案例数量上限")


class ContextSearchRequest(BaseModel):
    """上下文感知搜索请求"""
    creation_stage: str = Field(
        ...,
        description="当前创作阶段: world_building/character_design/outlining/chapter_outlining/writing/polishing",
    )
    current_content: str = Field(default="", description="当前正在创作的内容片段")
    genre: str = Field(default="", description="小说类型/题材")
    focus_aspect: str = Field(
        default="",
        description="当前关注的具体方面（如：力量体系升级规则/反派动机/伏笔埋设）",
    )
    limit: int = Field(default=10, ge=1, le=30)


# 创作阶段 -> 应搜索的知识维度映射
STAGE_DIMENSION_MAP = {
    "world_building": {
        "tables": ["world_settings", "world_timeline", "faction_networks", "setting_evolutions"],
        "collections": ["world_settings_kb"],
        "description": "世界观构建",
    },
    "character_design": {
        "tables": ["character_profiles", "character_speech_style", "character_behavior_marks"],
        "collections": ["character_profiles_kb", "character_speech_style_kb"],
        "description": "人物设计",
    },
    "outlining": {
        "tables": ["book_structure", "macro_outlines", "plot_lines", "emotional_arc"],
        "collections": ["macro_outlines_kb"],
        "description": "大纲规划",
    },
    "chapter_outlining": {
        "tables": ["chapter_functions", "plot_foreshadowing", "climax_point_distribution",
                   "conflict_escalation", "climax_buildup_chains"],
        "collections": ["macro_outlines_kb"],
        "description": "章节细纲",
    },
    "writing": {
        "tables": ["skills", "dialogue_samples", "description_samples", "transition_samples",
                   "action_scene_samples", "climax_excerpts"],
        "collections": ["novel_skills", "dialogue_samples_kb", "description_samples_kb",
                        "transition_samples_kb", "action_scene_samples_kb", "climax_excerpts_kb"],
        "description": "正文写作",
    },
    "polishing": {
        "tables": ["author_fingerprints", "style_summaries", "show_tell_patterns",
                   "narrative_distance", "memorable_quotes"],
        "collections": ["sensory_details", "classic_excerpts", "memorable_quotes_kb"],
        "description": "润色打磨",
    },
}

# 文档类型 -> 应搜索的知识维度映射
DOC_TYPE_DIMENSION_MAP = {
    "world_setting": {
        "primary_tables": ["world_settings", "world_timeline", "faction_networks"],
        "primary_collections": ["world_settings_kb"],
        "secondary_tables": ["setting_evolutions", "progression_systems"],
    },
    "character_profile": {
        "primary_tables": ["character_profiles", "character_speech_style", "character_behavior_marks"],
        "primary_collections": ["character_profiles_kb", "character_speech_style_kb"],
        "secondary_tables": ["character_relationship_dynamics"],
    },
    "outline": {
        "primary_tables": ["book_structure", "macro_outlines", "plot_lines", "emotional_arc"],
        "primary_collections": ["macro_outlines_kb"],
        "secondary_tables": ["conflict_escalation", "climax_buildup_chains"],
    },
    "chapter_outline": {
        "primary_tables": ["chapter_functions", "plot_foreshadowing", "climax_point_distribution"],
        "primary_collections": ["macro_outlines_kb"],
        "secondary_tables": ["revelation_pacing", "information_management"],
    },
    "chapter_text": {
        "primary_tables": ["skills", "dialogue_samples", "description_samples", "action_scene_samples"],
        "primary_collections": ["novel_skills", "dialogue_samples_kb", "description_samples_kb",
                                "action_scene_samples_kb"],
        "secondary_tables": ["transition_samples", "climax_excerpts", "memorable_quotes"],
    },
    "writing_style": {
        "primary_tables": ["author_fingerprints", "style_summaries", "show_tell_patterns"],
        "primary_collections": ["sensory_details", "classic_excerpts"],
        "secondary_tables": ["narrative_distance", "sentence_rhythm"],
    },
}


@router.post("/enhance")
def enhance_document(req: EnhanceRequest):
    """
    通用增强接口

    接收 AI 生成的文档内容和类型，自动从知识库中召回最相关的标杆案例，
    返回结构化的增强建议。适用场景：
    - 生成世界观设定后，查询标杆作品的世界观构建模式
    - 生成人物档案后，查询经典人物设计的深度维度
    - 生成正文后，查询同类场景的标杆写法和技法组合
    """
    db = get_db_manager()
    cursor = db.connect().cursor()
    chroma = get_chroma_manager()

    doc_config = DOC_TYPE_DIMENSION_MAP.get(req.document_type)
    if not doc_config:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文档类型: {req.document_type}。"
                   f"支持的类型: {', '.join(DOC_TYPE_DIMENSION_MAP.keys())}",
        )

    results = {"primary": [], "secondary": [], "cross_book_patterns": []}

    # 1. 向量搜索主要维度
    query_text = req.content[:500] if len(req.content) > 500 else req.content
    if req.specific_needs:
        query_text = f"{req.specific_needs} {query_text}"

    for collection_name in doc_config["primary_collections"]:
        try:
            chroma_res = chroma.query(
                collection_name, query_texts=[query_text], n_results=req.limit,
            )
            if chroma_res and chroma_res.get("ids"):
                for i, doc_id in enumerate(chroma_res["ids"][0]):
                    item = {
                        "id": doc_id,
                        "text": chroma_res["documents"][0][i] if chroma_res.get("documents") else "",
                        "metadata": chroma_res["metadatas"][0][i] if chroma_res.get("metadatas") else {},
                        "source": "semantic",
                    }
                    results["primary"].append(item)
        except Exception:
            pass

    # 2. SQL 补充主要维度
    for table_name in doc_config["primary_tables"]:
        try:
            columns = [r[1] for r in cursor.execute(f"PRAGMA table_info({table_name})").fetchall()]
            if not columns:
                continue
            sql = f"SELECT * FROM {table_name} WHERE 1=1"
            params = []
            if req.genre and "category" in columns:
                sql += " AND category LIKE ?"
                params.append(f"%{req.genre}%")
            sql += f" LIMIT {req.limit}"
            cursor.execute(sql, params)
            for row in cursor.fetchall():
                row_dict = dict(zip(columns, row))
                row_id = row_dict.get("id", "")
                if row_id and not any(r.get("id") == row_id for r in results["primary"]):
                    row_dict["source"] = "structured"
                    results["primary"].append(row_dict)
        except Exception:
            pass

    # 3. 次要维度补充
    for table_name in doc_config.get("secondary_tables", []):
        try:
            columns = [r[1] for r in cursor.execute(f"PRAGMA table_info({table_name})").fetchall()]
            if not columns:
                continue
            sql = f"SELECT * FROM {table_name} WHERE 1=1"
            params = []
            if req.genre and "category" in columns:
                sql += " AND category LIKE ?"
                params.append(f"%{req.genre}%")
            sql += f" LIMIT {min(req.limit // 2, 5)}"
            cursor.execute(sql, params)
            for row in cursor.fetchall():
                row_dict = dict(zip(columns, row))
                row_dict["source"] = "secondary"
                results["secondary"].append(row_dict)
        except Exception:
            pass

    # 4. 跨书模式
    try:
        cursor.execute(
            "SELECT comparison_dimension, common_patterns_json, best_practices FROM cross_book_comparisons LIMIT 5"
        )
        for row in cursor.fetchall():
            results["cross_book_patterns"].append({
                "dimension": row[0], "common_patterns": row[1], "best_practices": row[2],
            })
    except Exception:
        pass

    results["primary"] = results["primary"][:req.limit]
    results["secondary"] = results["secondary"][:req.limit // 2]

    return {
        "success": True,
        "data": {
            "document_type": req.document_type,
            "genre": req.genre,
            "total_primary": len(results["primary"]),
            "total_secondary": len(results["secondary"]),
            "total_patterns": len(results["cross_book_patterns"]),
            "results": results,
        },
    }


@router.post("/search/context")
def context_aware_search(req: ContextSearchRequest):
    """
    上下文感知搜索

    根据当前创作阶段自动选择应搜索的知识维度，无需用户手动指定。
    支持阶段: world_building/character_design/outlining/chapter_outlining/writing/polishing
    """
    stage_config = STAGE_DIMENSION_MAP.get(req.creation_stage)
    if not stage_config:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的创作阶段: {req.creation_stage}。"
                   f"支持的阶段: {', '.join(STAGE_DIMENSION_MAP.keys())}",
        )

    db = get_db_manager()
    cursor = db.connect().cursor()
    chroma = get_chroma_manager()

    query_text = req.current_content[:300] if req.current_content else ""
    if req.focus_aspect:
        query_text = f"{req.focus_aspect} {query_text}"
    if not query_text.strip():
        raise HTTPException(status_code=400, detail="请提供 current_content 或 focus_aspect")

    results = {"vectors": [], "structured": [], "stage_description": stage_config["description"]}

    for collection_name in stage_config["collections"]:
        try:
            chroma_res = chroma.query(collection_name, query_texts=[query_text], n_results=req.limit)
            if chroma_res and chroma_res.get("ids"):
                for i, doc_id in enumerate(chroma_res["ids"][0]):
                    results["vectors"].append({
                        "id": doc_id,
                        "text": chroma_res["documents"][0][i] if chroma_res.get("documents") else "",
                        "metadata": chroma_res["metadatas"][0][i] if chroma_res.get("metadatas") else {},
                        "collection": collection_name,
                    })
        except Exception:
            pass

    for table_name in stage_config["tables"]:
        try:
            columns = [r[1] for r in cursor.execute(f"PRAGMA table_info({table_name})").fetchall()]
            if not columns:
                continue
            sql = f"SELECT * FROM {table_name} WHERE 1=1"
            params = []
            if req.genre and "category" in columns:
                sql += " AND category LIKE ?"
                params.append(f"%{req.genre}%")
            sql += f" LIMIT {min(req.limit // 2, 5)}"
            cursor.execute(sql, params)
            for row in cursor.fetchall():
                row_dict = dict(zip(columns, row))
                row_dict["source_table"] = table_name
                results["structured"].append(row_dict)
        except Exception:
            pass

    results["vectors"] = results["vectors"][:req.limit]
    results["structured"] = results["structured"][:req.limit]
    return {"success": True, "data": results}


@router.get("/patterns/{pattern_type}")
def query_patterns(
    pattern_type: str,
    genre: Optional[str] = Query(None, description="小说类型过滤"),
    limit: int = Query(10, ge=1, le=30),
):
    """
    模式库接口

    查询某类型的通用创作模式，数据来源于跨书对比分析（Stage L）。
    支持: world_building/character_design/plot_structure/conflict_escalation/dialogue_style/pacing/foreshadowing/romance/horror/action
    """
    db = get_db_manager()
    cursor = db.connect().cursor()
    results = []

    try:
        sql = """SELECT comparison_dimension, books_analyzed, common_patterns_json,
                        unique_features_json, best_practices, created_at
                 FROM cross_book_comparisons WHERE comparison_dimension LIKE ?"""
        params = [f"%{pattern_type}%"]
        if genre:
            sql += " AND books_analyzed LIKE ?"
            params.append(f"%{genre}%")
        sql += f" ORDER BY created_at DESC LIMIT {limit}"
        cursor.execute(sql, params)
        for row in cursor.fetchall():
            results.append({
                "dimension": row[0], "books_analyzed": row[1],
                "common_patterns": row[2], "unique_features": row[3],
                "best_practices": row[4], "created_at": row[5],
            })
    except Exception:
        pass

    try:
        sql2 = """SELECT scene_type, combo_name, technique_sequence_json,
                         applicable_scenarios, variations, benchmark_book, original_example
                  FROM technique_combinations
                  WHERE scene_type LIKE ? OR combo_name LIKE ?"""
        params2 = [f"%{pattern_type}%", f"%{pattern_type}%"]
        sql2 += f" LIMIT {limit}"
        cursor.execute(sql2, params2)
        for row in cursor.fetchall():
            results.append({
                "type": "technique_combination", "scene_type": row[0],
                "combo_name": row[1], "technique_sequence": row[2],
                "applicable_scenarios": row[3], "variations": row[4],
                "benchmark_book": row[5], "original_example": row[6],
            })
    except Exception:
        pass

    return {"success": True, "data": {"pattern_type": pattern_type, "total_results": len(results), "results": results[:limit]}}


@router.get("/benchmarks/{dimension}")
def query_benchmarks(
    dimension: str,
    genre: Optional[str] = Query(None, description="小说类型过滤"),
    book_name: Optional[str] = Query(None, description="指定书名"),
    limit: int = Query(10, ge=1, le=30),
):
    """
    标杆参考接口

    查询某维度的标杆案例。支持: world_setting/character/plot/style/dialogue/description/pacing/conflict/foreshadowing/climax
    """
    db = get_db_manager()
    cursor = db.connect().cursor()

    DIMENSION_TABLE_MAP = {
        "world_setting": ["world_settings", "world_timeline"],
        "character": ["character_profiles", "character_speech_style"],
        "plot": ["book_structure", "macro_outlines", "plot_lines"],
        "style": ["author_fingerprints", "style_summaries"],
        "dialogue": ["dialogue_samples", "character_speech_style"],
        "description": ["description_samples", "sensory_mappings"],
        "pacing": ["climax_point_distribution", "revelation_pacing"],
        "conflict": ["conflict_escalation", "climax_buildup_chains"],
        "foreshadowing": ["plot_foreshadowing", "information_management"],
        "climax": ["climax_buildup_chains", "climax_excerpts"],
    }

    tables = DIMENSION_TABLE_MAP.get(dimension)
    if not tables:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的维度: {dimension}。支持: {', '.join(DIMENSION_TABLE_MAP.keys())}",
        )

    results = []
    for table_name in tables:
        try:
            columns = [r[1] for r in cursor.execute(f"PRAGMA table_info({table_name})").fetchall()]
            if not columns:
                continue
            sql = f"SELECT * FROM {table_name} WHERE 1=1"
            params = []
            if genre and "category" in columns:
                sql += " AND category LIKE ?"
                params.append(f"%{genre}%")
            if book_name and "book_name" in columns:
                sql += " AND book_name = ?"
                params.append(book_name)
            sql += f" LIMIT {limit}"
            cursor.execute(sql, params)
            for row in cursor.fetchall():
                row_dict = dict(zip(columns, row))
                row_dict["source_table"] = table_name
                results.append(row_dict)
        except Exception:
            pass

    return {"success": True, "data": {"dimension": dimension, "total_results": len(results), "results": results[:limit]}}
