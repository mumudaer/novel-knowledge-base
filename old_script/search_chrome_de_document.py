import chromadb
from pprint import pprint

# 连接到您的 ChromaDB
client = chromadb.PersistentClient(path="./novel_kb/chroma_db")
collection = client.get_collection("classic_excerpts")  # 这是您存储典例的集合

# 方法一：查看前 5 个典例
results = collection.peek(limit=100)
for i in range(len(results["ids"])):
    print(f"\n{'='*50}")
    print(f"【典例 #{i+1}】ID: {results['ids'][i]}")
    print(f"元数据: {results['metadatas'][i]}")
    print(f"正文内容 ({len(results['documents'][i])} 字):")
    print(results["documents"][i])
    print(f"{'='*50}")

# 方法二：按关键词搜索（例如找"雨夜"场景）
# query_results = collection.query(
#     query_texts=["雨夜"], n_results=3, include=["documents", "metadatas"]
# )
# for i, doc in enumerate(query_results["documents"][0]):
#     print(f"\n【搜索结果 #{i+1}】")
#     print(f"匹配度: {query_results['distances'][0][i]:.4f}")
#     print(f"元数据: {query_results['metadatas'][0][i]}")
#     print(f"正文内容 ({len(doc)} 字):")
#     print(doc)
