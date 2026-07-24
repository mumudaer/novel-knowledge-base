"""
ChromaDB 批量写入工具模块
提供通用的 ChromaDB 批量 upsert 函数，减少重复代码
"""
import json
import logging
from typing import List, Dict, Any
from core.chroma_client import get_chroma_manager
from core.utils import generate_id

logger = logging.getLogger(__name__)


def bulk_upsert_to_chroma(
    collection: str,
    items: List[Dict[str, Any]],
    id_fields: List[str],
    text_field: str,
    metadata_fields: List[str],
) -> int:
    """
    批量写入 ChromaDB

    Args:
        collection: 集合名称
        items: 数据项列表
        id_fields: 用于生成 ID 的字段列表
        text_field: 文本内容字段
        metadata_fields: 元数据字段列表

    Returns:
        成功写入的数量
    """
    if not items:
        return 0

    try:
        chroma = get_chroma_manager()
    except Exception as e:
        logger.warning(f"⚠️ ChromaDB 连接失败: {e}")
        return 0
    ids = []
    documents = []
    metadatas = []

    for item in items:
        doc_id = generate_id(*[str(item.get(f, "")) for f in id_fields])
        text = str(item.get(text_field, ""))
        metadata = {}
        for f in metadata_fields:
            val = item.get(f, "")
            # ChromaDB metadata 只支持 str/int/float/bool
            if isinstance(val, (list, dict)):
                val = json.dumps(val, ensure_ascii=False) if val else ""
            elif not isinstance(val, (str, int, float, bool)):
                val = str(val)
            metadata[f] = val

        ids.append(doc_id)
        documents.append(text)
        metadatas.append(metadata)

    # 去重：同一批次中相同 ID 只保留第一条（ChromaDB upsert 不接受重复 ID）
    if ids:
        seen = set()
        unique_ids, unique_docs, unique_metas = [], [], []
        for i, doc_id in enumerate(ids):
            if doc_id not in seen:
                seen.add(doc_id)
                unique_ids.append(doc_id)
                unique_docs.append(documents[i])
                unique_metas.append(metadatas[i])
        ids, documents, metadatas = unique_ids, unique_docs, unique_metas

    if ids:
        try:
            chroma.upsert_batch(collection, ids, documents, metadatas)
        except Exception as e:
            logger.warning(f"⚠️ ChromaDB 批量写入 {collection} 失败: {e}")
            return 0

    return len(ids)
