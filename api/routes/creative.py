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
               "content", "tags",                ]
    
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
        "personality", "relation_to_mc",
        "relations_to_others", "climax_or_fate", "background",
                "speech_samples", "behavior_samples",
        "relationship_evolution", "abilities", "internal_dilemma",
        "transformation_trigger", "contrast_design",
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


# ===================== 大纲/结构搜索 =====================

@router.get("/search/plot")


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


# ===================== 按书名检索全部知识 =====================

@router.get("/search/by-book")


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


# ===================== 线索/推理搜索 =====================

@router.get("/search/mystery")


# ===================== 升级体系搜索 =====================

@router.get("/search/progression")


# ===================== 类型技法搜索 =====================

@router.get("/search/genre-technique")


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

# 文档类型 -> 应搜索的知识维度映射


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


# ===================== 事件因果图谱接口 =====================


@router.get("/events/{book_name}")
def query_story_events(
    book_name: str,
    chapter_range: Optional[str] = Query(None, description="章节范围过滤（如：1-100）"),
    event_type: Optional[str] = Query(None, description="事件类型过滤（如：伏笔埋设/冲突爆发/高潮）"),
    significance: Optional[str] = Query(None, description="重要性过滤（high/medium/low）"),
    limit: int = Query(100, ge=1, le=500),
):
    """
    查询关键事件列表

    返回指定书籍的所有关键事件，支持按章节范围、事件类型、重要性过滤。
    """
    db = get_db_manager()
    cursor = db.connect().cursor()

    sql = "SELECT * FROM story_events WHERE book_name = ?"
    params: list = [book_name]

    if event_type:
        sql += " AND event_type = ?"
        params.append(event_type)
    if significance:
        sql += " AND significance = ?"
        params.append(significance)

    # 不使用 SQL LIMIT，在 Python 侧过滤后再截断（解决 chapter_range 过滤顺序问题）
    sql += " ORDER BY chapter_id"

    columns = ["id", "book_name", "chapter_id", "event_name", "event_summary", "event_type", "characters_involved", "significance"]

    try:
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        events = [dict(zip(columns, row)) for row in rows]

        # 解析 characters_involved JSON
        for event in events:
            try:
                event["characters_involved"] = json.loads(event["characters_involved"])
            except (json.JSONDecodeError, TypeError):
                event["characters_involved"] = []

        # 按章节范围过滤
        if chapter_range and "-" in chapter_range:
            try:
                start, end = chapter_range.split("-")
                start, end = int(start), int(end)
                import re
                filtered = []
                for event in events:
                    match = re.search(r"(\d+)", event.get("chapter_id", ""))
                    if match:
                        chap_num = int(match.group(1))
                        if start <= chap_num <= end:
                            filtered.append(event)
                events = filtered
            except ValueError:
                pass

        # 按章节编号数值排序（解决字符串排序 "第10章" < "第2章" 的问题）
        import re as _re
        def _chapter_num(e):
            m = _re.search(r"(\d+)", e.get("chapter_id", ""))
            return int(m.group(1)) if m else 99999
        events.sort(key=_chapter_num)

        # 过滤后再截断
        events = events[:limit]

        return {"success": True, "data": {"book_name": book_name, "total": len(events), "events": events}}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/causal-chain/{book_name}")
def query_causal_chain(
    book_name: str,
    event_name: Optional[str] = Query(None, description="起始事件名称（精确匹配）"),
    depth: int = Query(3, ge=1, le=10, description="遍历深度"),
    direction: Optional[str] = Query("downstream", description="方向：downstream/upstream/both"),
):
    """
    查询事件因果链

    从指定事件出发，沿因果关系图遍历，返回因果链。
    """
    db = get_db_manager()
    cursor = db.connect().cursor()

    # 加载该书所有事件
    cursor.execute("SELECT * FROM story_events WHERE book_name = ?", (book_name,))
    event_columns = ["id", "book_name", "chapter_id", "event_name", "event_summary", "event_type", "characters_involved", "significance"]
    all_events = {}
    for row in cursor.fetchall():
        event = dict(zip(event_columns, row))
        all_events[event["id"]] = event

    # 加载该书所有因果边
    cursor.execute("SELECT * FROM event_causal_edges WHERE book_name = ?", (book_name,))
    edge_columns = ["id", "book_name", "source_event_id", "target_event_id", "relation_type", "relation_detail"]
    edges = [dict(zip(edge_columns, row)) for row in cursor.fetchall()]

    # 构建邻接表
    downstream = {}  # event_id -> [(target_id, relation_type, detail)]
    upstream = {}    # event_id -> [(source_id, relation_type, detail)]
    for edge in edges:
        src = edge["source_event_id"]
        tgt = edge["target_event_id"]
        downstream.setdefault(src, []).append((tgt, edge["relation_type"], edge["relation_detail"]))
        upstream.setdefault(tgt, []).append((src, edge["relation_type"], edge["relation_detail"]))

    # 找到起始事件
    start_event_id = None
    if event_name:
        for eid, event in all_events.items():
            if event["event_name"] == event_name:
                start_event_id = eid
                break
        if not start_event_id:
            return {"success": False, "error": f"未找到事件: {event_name}"}
    else:
        # 没有指定起始事件，返回所有高重要性事件的因果链概览
        high_events = [e for e in all_events.values() if e.get("significance") == "high"]
        return {
            "success": True,
            "data": {
                "book_name": book_name,
                "total_events": len(all_events),
                "total_edges": len(edges),
                "high_significance_events": [
                    {"event_name": e["event_name"], "chapter_id": e["chapter_id"], "event_type": e["event_type"]}
                    for e in high_events[:50]
                ],
            },
        }

    # BFS 遍历因果链
    visited = set()
    chain = []

    def traverse(event_id: str, current_depth: int, direction_label: str):
        if current_depth > depth or event_id in visited:
            return
        visited.add(event_id)

        event = all_events.get(event_id, {})
        chain.append({
            "depth": current_depth,
            "direction": direction_label,
            "event_name": event.get("event_name", ""),
            "event_summary": event.get("event_summary", ""),
            "chapter_id": event.get("chapter_id", ""),
            "event_type": event.get("event_type", ""),
        })

        if direction in ("downstream", "both"):
            for tgt_id, rel_type, rel_detail in downstream.get(event_id, []):
                tgt_event = all_events.get(tgt_id, {})
                chain.append({
                    "depth": current_depth + 1,
                    "direction": "downstream",
                    "relation_type": rel_type,
                    "relation_detail": rel_detail,
                    "target_event_name": tgt_event.get("event_name", ""),
                    "target_chapter_id": tgt_event.get("chapter_id", ""),
                })
                traverse(tgt_id, current_depth + 1, "downstream")

        if direction in ("upstream", "both"):
            for src_id, rel_type, rel_detail in upstream.get(event_id, []):
                src_event = all_events.get(src_id, {})
                chain.append({
                    "depth": current_depth + 1,
                    "direction": "upstream",
                    "relation_type": rel_type,
                    "relation_detail": rel_detail,
                    "target_event_name": src_event.get("event_name", ""),
                    "target_chapter_id": src_event.get("chapter_id", ""),
                })
                traverse(src_id, current_depth + 1, "upstream")

    traverse(start_event_id, 0, "start")

    return {
        "success": True,
        "data": {
            "book_name": book_name,
            "start_event": all_events.get(start_event_id, {}).get("event_name", ""),
            "direction": direction,
            "depth": depth,
            "chain_length": len(chain),
            "chain": chain,
        },
    }


# ===================== 去AI味判据库 =====================

_anti_ai_cache = None


@router.get("/anti-ai-patterns")
def get_anti_ai_patterns():
    """
    获取去AI味判据库

    返回：
    - banned_words: AI高频词汇 + 替换建议
    - banned_patterns: AI味句式模板 + 替代表达
    - style_guidelines: 标杆文风参考方向
    """
    global _anti_ai_cache
    if _anti_ai_cache is None:
        import os
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "anti_ai_patterns.json",
        )
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                _anti_ai_cache = json.load(f)
        except FileNotFoundError:
            return {"success": False, "error": "anti_ai_patterns.json 未找到"}

    return {"success": True, "data": _anti_ai_cache}


# ===================== 题材裁决规则 =====================


@router.get("/genre-rules/{genre}")
def get_genre_rules(
    genre: str = Query(..., description="题材标签"),
):
    """查询该题材的裁决规则（genre_rules表已废弃）"""
    return {"success": True, "data": [], "total": 0}

