import sqlite3
import chromadb
import json
import os
import random

# ⚠️ 请将这里的路径替换为您项目中实际的路径变量
SQLITE_PATH = "./novel_kb/knowledge.db"  # ✅ 不是 chroma.sqlite3
CHROMA_PATH = "./novel_kb/chroma_db"  # ✅ 是目录名，不是文件
GRAPH_PATH = "./novel_kb/knowledge_graph.graphml"  # ✅ 文件名


def peek_sqlite(cursor, table_name):
    """抽样查看 SQLite 表"""
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    print(f"  📊 表 [{table_name}]: 共 {count} 条记录")

    if count > 0:
        cursor.execute(f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT 1")
        cols = [desc[0] for desc in cursor.description]
        row = cursor.fetchone()
        sample = dict(zip(cols, row))

        # 格式化打印，避免长文本糊脸
        print("    📝 随机抽样 1 条:")
        for k, v in sample.items():
            val_str = str(v)
            if len(val_str) > 100:
                val_str = val_str[:100] + "..."
            print(f"       - {k}: {val_str}")
    print("-" * 40)


def peek_chroma(collection_name, client):
    """抽样查看 ChromaDB 集合"""
    try:
        collection = client.get_collection(name=collection_name)
        count = collection.count()
        print(f"  🧠 集合 [{collection_name}]: 共 {count} 个向量")

        if count > 0:
            # 随机取一条数据
            sample = collection.peek(limit=1)
            print("    📝 随机抽样 1 条:")
            print(f"       - ID: {sample['ids'][0]}")

            doc = sample["documents"][0]
            if len(doc) > 150:
                doc = doc[:150] + "..."
            print(f"       - 文本内容: {doc}")

            meta = sample["metadatas"][0]
            print(f"       - 元数据: {json.dumps(meta, ensure_ascii=False)}")
        print("-" * 40)
    except Exception as e:
        print(f"  ❌ 集合 [{collection_name}] 读取失败: {e}")
        print("-" * 40)


def main():
    print("=" * 50)
    print("🔍 开始进行知识库全面体检...")
    print("=" * 50)

    # 1. 检查 SQLite
    print("\n【1. SQLite 关系型数据库】")
    if not os.path.exists(SQLITE_PATH):
        print(f"❌ 找不到 SQLite 文件: {SQLITE_PATH}")
    else:
        conn = sqlite3.connect(SQLITE_PATH)
        cursor = conn.cursor()
        for table in ["plot_arcs", "skills", "author_fingerprints", "sensory_mappings"]:
            peek_sqlite(cursor, table)
        conn.close()

    # 2. 检查 ChromaDB
    print("\n【2. ChromaDB 向量数据库】")
    if not os.path.exists(CHROMA_PATH):
        print(f"❌ 找不到 ChromaDB 目录: {CHROMA_PATH}")
    else:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        for col_name in ["novel_skills", "sensory_details", "classic_excerpts"]:
            peek_chroma(col_name, client)

    # 3. 检查 知识图谱
    print("\n【3. NetworkX 知识图谱】")
    if os.path.exists(GRAPH_PATH):
        size_kb = os.path.getsize(GRAPH_PATH) / 1024
        print(f"  🕸️ 图谱文件存在: {GRAPH_PATH} (大小: {size_kb:.2f} KB)")
        # 简单读取看看节点数
        try:
            import networkx as nx

            G = nx.read_graphml(GRAPH_PATH)
            print(
                f"  📊 包含节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}"
            )
        except Exception as e:
            print(f"  ⚠️ 图谱文件可能损坏，无法读取: {e}")
    else:
        print(f"  ❌ 找不到图谱文件: {GRAPH_PATH}")

    print("\n" + "=" * 50)
    print("✅ 体检完成！请核对上述抽样数据是否符合预期。")
    print("=" * 50)


if __name__ == "__main__":
    main()
