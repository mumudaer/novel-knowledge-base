import os
import sys
import re
import sqlite3
import chromadb
import networkx as nx
import json
import logging

# ===================== 日志配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("BookBanisher")

# ===================== 目标书籍与路径配置 =====================
TARGET_BOOKS = ["干掉万人迷的一百种方法", "穿进万人迷文的我人设崩了"]

# 自动探测 BASE_DIR (与主程序逻辑完全一致)
if os.environ.get("NOVEL_KB_DATA_DIR"):
    BASE_DIR = os.environ["NOVEL_KB_DATA_DIR"]
else:
    app_root = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.join(app_root, "novel_kb")

SQLITE_PATH = os.path.join(BASE_DIR, "knowledge.db")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
GRAPH_PATH = os.path.join(BASE_DIR, "knowledge_graph.graphml")
MANIFEST_FILE = os.path.join(BASE_DIR, "process_manifest.json")


def wipe_sqlite(book_name: str, cursor: sqlite3.Cursor):
    """SQLite 物理级删行，彻底清除脏数据"""
    tables = ["skills", "plot_arcs", "author_fingerprints", "sensory_mappings"]
    total_deleted = 0
    for table in tables:
        try:
            cursor.execute(
                f"SELECT COUNT(*) FROM {table} WHERE book_name = ?", (book_name,)
            )
            count = cursor.fetchone()[0]
            if count > 0:
                cursor.execute(f"DELETE FROM {table} WHERE book_name = ?", (book_name,))
                logger.info(f"  🗑️ [{table}] 物理删除 {count} 条记录")
                total_deleted += count
        except sqlite3.OperationalError:
            pass
    return total_deleted


def wipe_chroma(book_name: str, client: chromadb.ClientAPI):
    """清理 ChromaDB 向量数据"""
    collections = ["novel_skills", "sensory_details", "classic_excerpts"]
    total_deleted = 0
    for col_name in collections:
        try:
            collection = client.get_collection(name=col_name)
            results = collection.get(where={"book_name": book_name})
            if results and results["ids"]:
                ids_to_delete = results["ids"]
                collection.delete(ids=ids_to_delete)
                logger.info(f"  🗑️ [{col_name}] 删除 {len(ids_to_delete)} 个向量")
                total_deleted += len(ids_to_delete)
        except Exception:
            pass
    return total_deleted


def wipe_graph(book_name: str, graph: nx.DiGraph):
    """清理 NetworkX 知识图谱中的相关节点"""
    nodes_to_remove = []

    for node, attr in list(graph.nodes(data=True)):
        if (
            node.startswith("char:") or node.startswith("skill:")
        ) and book_name in attr.get("book_list", ""):
            books = [
                b.strip() for b in attr.get("book_list", "").split(",") if b.strip()
            ]
            if book_name in books:
                books.remove(book_name)

            if not books:
                nodes_to_remove.append(node)
            else:
                graph.nodes[node]["book_list"] = ",".join(books)

    if nodes_to_remove:
        graph.remove_nodes_from(nodes_to_remove)
        logger.info(f"  🗑️ [Graph] 移除 {len(nodes_to_remove)} 个专属节点")
    return len(nodes_to_remove)


def wipe_state_files(book_name: str):
    """清理断点续传的状态 JSON 文件与临时合并文件"""
    safe_name = re.sub(r'[\\/*?:"<>|]', "", book_name)
    patterns = [
        f"state_A_{safe_name}.json",
        f"state_B_{safe_name}.json",
        f"state_C_{safe_name}.json",
        f"state_A_window_{safe_name}.json",
        f"temp_{book_name}.txt",
    ]
    deleted = 0
    for p in patterns:
        path = os.path.join(BASE_DIR, p)
        if os.path.exists(path):
            os.remove(path)
            deleted += 1
    if deleted > 0:
        logger.info(f"  🗑️ [Files] 清理了 {deleted} 个断点/临时文件")
    return deleted


def banish_in_manifest(book_name: str):
    """【核心修改】强制将书籍加入完工清单（黑名单），让主程序永久跳过"""
    if not os.path.exists(MANIFEST_FILE):
        # 如果 manifest 不存在，创建一个
        data = {"completed_books": []}
    else:
        try:
            with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {"completed_books": []}

    completed_list = data.get("completed_books", [])
    if book_name not in completed_list:
        completed_list.append(book_name)
        data["completed_books"] = completed_list
        with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"  🚫 [Manifest] 已加入完工清单（黑名单），主程序将永久跳过此书！")
    else:
        logger.info(f"  🚫 [Manifest] 已在完工清单中，无需重复添加。")


if __name__ == "__main__":
    logger.info(f"🚀 启动永久剔除程序，目标：{TARGET_BOOKS}")

    # 1. 连接数据库
    db_conn = sqlite3.connect(SQLITE_PATH) if os.path.exists(SQLITE_PATH) else None
    chroma_client = (
        chromadb.PersistentClient(path=CHROMA_PATH)
        if os.path.exists(CHROMA_PATH)
        else None
    )

    # 2. 加载图谱
    graph = nx.DiGraph()
    if os.path.exists(GRAPH_PATH):
        try:
            graph = nx.read_graphml(GRAPH_PATH)
        except Exception:
            pass

    for book in TARGET_BOOKS:
        logger.info(f"\n🔥 正在永久剔除：《{book}》")

        if db_conn:
            wipe_sqlite(book, db_conn.cursor())
            db_conn.commit()

        if chroma_client:
            wipe_chroma(book, chroma_client)

        wipe_graph(book, graph)
        wipe_state_files(book)
        banish_in_manifest(book)  # 加入黑名单

    # 3. 保存清理后的图谱
    if os.path.exists(GRAPH_PATH) and graph.number_of_nodes() > 0:
        try:
            # 清理空值，防止 GraphML 报错
            for _, attr in graph.nodes(data=True):
                for k, v in list(attr.items()):
                    if v is None:
                        attr[k] = ""
            for _, _, attr in graph.edges(data=True):
                for k, v in list(attr.items()):
                    if v is None:
                        attr[k] = ""

            nx.write_graphml(graph, GRAPH_PATH)
            logger.info("\n✅ 知识图谱已更新保存。")
        except Exception as e:
            logger.error(f"⚠️ 图谱保存失败: {e}")

    if db_conn:
        db_conn.close()

    logger.info("\n🎉 剔除完毕！这两本书的数据已从物理层面抹除。")
    logger.info(
        "💡 提示：它们已被加入 Manifest 黑名单。下次运行主程序时，系统会自动跳过它们。"
    )
