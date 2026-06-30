"""
ChromaDB 向量数据库管理模块
封装 ChromaDB 的初始化、集合管理、向量操作
使用 bge-m3 中文优化 embedding 模型替代默认英文模型
"""
import chromadb
import logging
import threading
from typing import Optional, List, Dict, Any
from config.settings import (
    CHROMA_PATH,
    CHROMA_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
)

logger = logging.getLogger(__name__)


class ChromaManager:
    """ChromaDB 向量数据库管理器"""

    # 集合定义
    COLLECTIONS = {
        # Stage B: 写作技法
        "novel_skills": "写作技法向量库",

        # Stage C: 感官映射与经典摘录
        "sensory_details": "感官映射向量库",
        "classic_excerpts": "经典文风段落向量库",

        # Stage D: 世界观与人物
        "world_settings_kb": "世界观矩阵向量库",
        "character_profiles_kb": "人物静态底色向量库",

        # Stage E: 宏观大纲
        "macro_outlines_kb": "宏观卷大纲向量库",

        # Stage F: 样本库
        "dialogue_samples_kb": "对话样本向量库",
        "description_samples_kb": "描写样本向量库",
        "transition_samples_kb": "转场样本向量库",
        "action_scene_samples_kb": "动作场景样本向量库",
        "climax_excerpts_kb": "高潮段落向量库",
        "memorable_quotes_kb": "金句名句向量库",

        # Stage G: 人物深度特征
        "character_speech_style_kb": "人物语言风格向量库",

        # Stage N: 技法组合
        "technique_combinations_kb": "技法组合模板向量库",
    }

    def __init__(self, chroma_path: Optional[str] = None):
        """初始化 ChromaDB 客户端"""
        self.chroma_path = chroma_path or CHROMA_PATH
        self.client: Optional[chromadb.PersistentClient] = None
        self.collections: Dict[str, Any] = {}
        self._embedding_fn = None  # 延迟加载 embedding 模型

    def _get_embedding_function(self):
        """
        延迟加载 bge-m3 embedding 函数
        使用延迟加载策略：只在首次调用时加载模型到显存，
        配合 VRAMManager 实现与 LLM 推理的分时复用
        """
        if self._embedding_fn is None:
            try:
                from chromadb.utils.embedding_functions import (
                    SentenceTransformerEmbeddingFunction,
                )

                logger.info(
                    f"正在加载 embedding 模型: {EMBEDDING_MODEL_NAME} (device={EMBEDDING_DEVICE})"
                )
                self._embedding_fn = SentenceTransformerEmbeddingFunction(
                    model_name=EMBEDDING_MODEL_NAME,
                    device=EMBEDDING_DEVICE,
                )
                logger.info(f"Embedding 模型加载成功: {EMBEDDING_MODEL_NAME}")
            except Exception as e:
                logger.warning(
                    f"bge-m3 加载失败 ({e})，降级使用 ChromaDB 默认 embedding（英文，中文检索质量会下降）"
                )
                self._embedding_fn = None  # 降级为 None，让 ChromaDB 用默认
        return self._embedding_fn

    def connect(self) -> chromadb.PersistentClient:
        """建立 ChromaDB 连接"""
        if self.client is None:
            self.client = chromadb.PersistentClient(path=self.chroma_path)
        return self.client

    def init_collections(self, reset: bool = False):
        """
        初始化所有集合

        Args:
            reset: 是否清空并重建所有集合（当切换 embedding 模型时必须设为 True，
                   因为旧向量数据与新模型不兼容）
        """
        client = self.connect()
        embedding_fn = self._get_embedding_function()

        if reset:
            print("WARNING: 正在清空并重建所有 ChromaDB 集合（embedding 模型已更换，旧向量数据不兼容）")

        print("正在初始化 ChromaDB 向量库...")

        for collection_name, description in self.COLLECTIONS.items():
            try:
                if reset:
                    # 删除旧集合并重建
                    client.delete_collection(name=collection_name)

                if embedding_fn is not None:
                    self.collections[collection_name] = client.get_or_create_collection(
                        name=collection_name,
                        embedding_function=embedding_fn,
                    )
                else:
                    # 降级：不传 embedding_function，使用 ChromaDB 默认
                    self.collections[collection_name] = client.get_or_create_collection(
                        name=collection_name,
                    )

                logger.debug(f"集合 [{collection_name}] 初始化成功: {description}")
            except Exception as e:
                logger.error(f"集合 [{collection_name}] 初始化失败: {e}")
                # 降级：尝试不带 embedding_function 初始化
                try:
                    if reset:
                        client.delete_collection(name=collection_name)
                    self.collections[collection_name] = client.get_or_create_collection(
                        name=collection_name,
                    )
                    logger.warning(
                        f"集合 [{collection_name}] 已降级为默认 embedding 初始化"
                    )
                except Exception as fallback_err:
                    logger.error(f"集合 [{collection_name}] 降级初始化也失败: {fallback_err}")

        print(f"ChromaDB 初始化完毕，共 {len(self.collections)} 个集合。")

    def get_collection(self, name: str):
        """获取指定集合"""
        if name not in self.collections:
            raise ValueError(f"集合 [{name}] 不存在，请先调用 init_collections()")
        return self.collections[name]

    def upsert_batch(
        self,
        collection_name: str,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ):
        """批量写入向量数据"""
        collection = self.get_collection(collection_name)

        # 分批写入，避免内存峰值
        for i in range(0, len(ids), CHROMA_BATCH_SIZE):
            batch_ids = ids[i : i + CHROMA_BATCH_SIZE]
            batch_docs = documents[i : i + CHROMA_BATCH_SIZE]
            batch_metas = metadatas[i : i + CHROMA_BATCH_SIZE]

            try:
                collection.upsert(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metas,
                )
            except Exception as e:
                logger.error(f"ChromaDB 批量写入失败: {e}")
                raise

    def query(
        self,
        collection_name: str,
        query_texts: List[str],
        n_results: int = 5,
        where: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """查询向量数据"""
        collection = self.get_collection(collection_name)

        try:
            results = collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where,
            )
            return results
        except Exception as e:
            logger.error(f"ChromaDB 查询失败: {e}")
            return {"ids": [], "documents": [], "metadatas": []}

    def count(self, collection_name: str) -> int:
        """统计集合中的文档数量"""
        collection = self.get_collection(collection_name)
        return collection.count()

    def unload_embedding_model(self):
        """
        卸载 embedding 模型释放显存
        配合 VRAMManager 实现分时复用：
        - 批量写入完成后调用此方法释放显存
        - 下次查询时会重新加载
        """
        if self._embedding_fn is not None:
            try:
                # 尝试清理 sentence-transformers 模型的 GPU 缓存
                model = getattr(self._embedding_fn, "_model", None)
                if model is not None:
                    del model
                self._embedding_fn = None
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                logger.info("Embedding 模型已卸载，显存已释放")
            except Exception as e:
                logger.warning(f"卸载 embedding 模型时出错: {e}")
                self._embedding_fn = None


# 全局 ChromaDB 管理器实例
_global_chroma_manager: Optional[ChromaManager] = None
_chroma_manager_lock = threading.Lock()


def get_chroma_manager() -> ChromaManager:
    """获取全局 ChromaDB 管理器实例（线程安全）"""
    global _global_chroma_manager
    if _global_chroma_manager is None:
        with _chroma_manager_lock:
            # 双重检查锁定
            if _global_chroma_manager is None:
                _global_chroma_manager = ChromaManager()
    return _global_chroma_manager
