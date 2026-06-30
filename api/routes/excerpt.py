"""
样本库语义搜索接口
"""
from typing import Optional
from fastapi import APIRouter, Query
from pydantic import BaseModel
from core.chroma_client import get_chroma_manager
from core.search_utils import hybrid_search

router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    book_name: Optional[str] = None
    category: Optional[str] = None
    scene_type: Optional[str] = None
    limit: int = 5


@router.post("/search")
def search_excerpts(request: SearchRequest):
    """
    语义搜索样本库
    支持搜索对话样本、描写样本、经典摘录
    """
    chroma = get_chroma_manager()

    # 构建过滤条件
    where_filter = {}
    if request.book_name:
        where_filter["book_name"] = request.book_name
    if request.category:
        where_filter["category"] = request.category
    if request.scene_type:
        where_filter["scene_type"] = request.scene_type

    where = where_filter if where_filter else None

    # 搜索三个集合
    results = {"dialogue": [], "description": [], "excerpts": []}

    # 搜索对话样本
    try:
        dialogue_res = chroma.query(
            "dialogue_samples_kb",
            query_texts=[request.query],
            n_results=request.limit,
            where=where,
        )
        if dialogue_res and dialogue_res.get("ids"):
            for i, doc_id in enumerate(dialogue_res["ids"][0]):
                results["dialogue"].append({
                    "id": doc_id,
                    "text": dialogue_res["documents"][0][i],
                    "metadata": dialogue_res["metadatas"][0][i] if dialogue_res.get("metadatas") else {},
                })
    except Exception:
        pass

    # 搜索描写样本
    try:
        desc_res = chroma.query(
            "description_samples_kb",
            query_texts=[request.query],
            n_results=request.limit,
            where=where,
        )
        if desc_res and desc_res.get("ids"):
            for i, doc_id in enumerate(desc_res["ids"][0]):
                results["description"].append({
                    "id": doc_id,
                    "text": desc_res["documents"][0][i],
                    "metadata": desc_res["metadatas"][0][i] if desc_res.get("metadatas") else {},
                })
    except Exception:
        pass

    # 搜索经典摘录
    try:
        excerpt_res = chroma.query(
            "classic_excerpts",
            query_texts=[request.query],
            n_results=request.limit,
            where=where,
        )
        if excerpt_res and excerpt_res.get("ids"):
            for i, doc_id in enumerate(excerpt_res["ids"][0]):
                results["excerpts"].append({
                    "id": doc_id,
                    "text": excerpt_res["documents"][0][i],
                    "metadata": excerpt_res["metadatas"][0][i] if excerpt_res.get("metadatas") else {},
                })
    except Exception:
        pass

    total = sum(len(v) for v in results.values())
    return {"success": True, "data": results, "total": total}


@router.get("/dialogue")
def search_dialogue(
    query: str = Query(..., description="搜索关键词"),
    book_name: Optional[str] = Query(None, description="书名"),
    scene_type: Optional[str] = Query(None, description="场景类型"),
    limit: int = Query(5, ge=1, le=20, description="返回数量"),
):
    """语义搜索对话样本 - 混合检索"""
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


@router.get("/description")
def search_description(
    query: str = Query(..., description="搜索关键词"),
    book_name: Optional[str] = Query(None, description="书名"),
    description_type: Optional[str] = Query(None, description="描写类型"),
    limit: int = Query(5, ge=1, le=20, description="返回数量"),
):
    """语义搜索描写样本 - 混合检索"""
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


@router.get("/classic")
def search_classic_excerpts(
    query: str = Query(..., description="搜索关键词"),
    book_name: Optional[str] = Query(None, description="书名"),
    style_tag: Optional[str] = Query(None, description="风格标签"),
    limit: int = Query(5, ge=1, le=20, description="返回数量"),
):
    """语义搜索经典摘录"""
    chroma = get_chroma_manager()

    where_filter = {}
    if book_name:
        where_filter["book_name"] = book_name
    if style_tag:
        where_filter["style_tag"] = style_tag

    where = where_filter if where_filter else None

    res = chroma.query(
        "classic_excerpts",
        query_texts=[query],
        n_results=limit,
        where=where,
    )

    items = []
    if res and res.get("ids"):
        for i, doc_id in enumerate(res["ids"][0]):
            items.append({
                "id": doc_id,
                "text": res["documents"][0][i],
                "metadata": res["metadatas"][0][i] if res.get("metadatas") else {},
            })

    return {"success": True, "data": items, "total": len(items)}
