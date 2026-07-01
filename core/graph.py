"""
知识图谱管理模块
封装 NetworkX 图谱的初始化、节点/边操作、持久化
"""
import os
import re
import threading
import networkx as nx
import logging
from typing import Optional
from config.settings import BASE_DIR

logger = logging.getLogger(__name__)


class GraphManager:
    """知识图谱管理器"""

    # 预编译正则，避免每次调用都重新编译
    _ILLEGAL_XML_RE = re.compile(
        "[\x00-\x08\x0b\x0c\x0e-\x1F\uD800-\uDFFF\uFFFE\uFFFF]"
    )

    def __init__(self, graph_path: Optional[str] = None):
        """初始化图谱管理器"""
        self.graph_path = graph_path or os.path.join(BASE_DIR, "knowledge_graph.graphml")
        self.graph: Optional[nx.DiGraph] = None

    def load(self) -> nx.DiGraph:
        """加载或创建知识图谱"""
        if self.graph is not None:
            return self.graph

        if os.path.exists(self.graph_path):
            try:
                self.graph = nx.read_graphml(self.graph_path)
                logger.info(f"✅ 知识图谱加载成功: {self.graph_path}")
            except Exception as e:
                logger.warning(f"⚠️ 知识图谱加载失败: {e}，将创建新图谱")
                try:
                    os.remove(self.graph_path)
                except Exception:
                    pass
                self.graph = nx.DiGraph()
        else:
            self.graph = nx.DiGraph()
            logger.info("📊 创建新知识图谱")

        return self.graph

    def save(self):
        """保存知识图谱到文件"""
        if self.graph is None:
            logger.warning("⚠️ 知识图谱未初始化，跳过保存")
            return

        try:
            # 清理图谱中的非法字符
            sanitized = self._sanitize_graph()
            nx.write_graphml(sanitized, self.graph_path)
            logger.info(f"✅ 知识图谱保存成功: {self.graph_path}")
        except Exception as e:
            logger.error(f"❌ 知识图谱保存失败: {e}")

    def _sanitize_graph(self) -> nx.DiGraph:
        """清理图谱中的非法字符（GraphML 格式要求）"""
        import json as _json
        sanitized = nx.DiGraph()

        def clean_val(v):
            if v is None:
                return ""
            if isinstance(v, (dict, list)):
                return _json.dumps(v, ensure_ascii=False)  # dict/list 转为合法 JSON 字符串
            if isinstance(v, str):
                return self._clean_xml_chars(v)  # 清理 XML 非法字符
            return str(v)

        for node, data in self.graph.nodes(data=True):
            clean_data = {k: clean_val(v) for k, v in data.items()}
            sanitized.add_node(node, **clean_data)

        for u, v, data in self.graph.edges(data=True):
            clean_data = {k: clean_val(val) for k, val in data.items()}
            sanitized.add_edge(u, v, **clean_data)

        return sanitized

    @staticmethod
    def _clean_xml_chars(text: str) -> str:
        """清理 XML 非法字符"""
        return GraphManager._ILLEGAL_XML_RE.sub("", text)

    def add_node(self, node_id: str, **attributes):
        """添加节点"""
        graph = self.load()
        graph.add_node(node_id, **attributes)

    def add_edge(self, source: str, target: str, **attributes):
        """添加边"""
        graph = self.load()
        graph.add_edge(source, target, **attributes)

    def safe_append_edge_attr(
        self, source: str, target: str, attr_name: str, attr_value: str
    ):
        """安全地追加边的属性值（按逗号分割去重，避免子串误判）"""
        if not attr_value:
            return
        graph = self.load()

        if not graph.has_edge(source, target):
            graph.add_edge(source, target, **{attr_name: attr_value})
        else:
            old_val = str(graph[source][target].get(attr_name, ""))
            old_list = [x.strip() for x in old_val.split(",") if x.strip()] if old_val else []
            if attr_value not in old_list:
                old_list.append(attr_value)
            graph[source][target][attr_name] = ",".join(old_list)


# 全局图谱管理器实例
_global_graph_manager: Optional[GraphManager] = None
_graph_manager_lock = threading.Lock()


def get_graph_manager() -> GraphManager:
    """获取全局图谱管理器实例（线程安全）"""
    global _global_graph_manager
    if _global_graph_manager is None:
        with _graph_manager_lock:
            if _global_graph_manager is None:
                _global_graph_manager = GraphManager()
    return _global_graph_manager
