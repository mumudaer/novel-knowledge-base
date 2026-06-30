import os
import sys
import re
import sqlite3
import chromadb
import logging

# ===================== 日志配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("GarbageCleaner")

# ===================== 路径自动探测 =====================
# 完美复刻你主代码中的 BASE_DIR 探测逻辑
if os.environ.get("NOVEL_KB_DATA_DIR"):
    BASE_DIR = os.environ["NOVEL_KB_DATA_DIR"]
else:
    app_root = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.join(app_root, "novel_kb")

SQLITE_PATH = os.path.join(BASE_DIR, "knowledge.db")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

# ===================== 核心检测引擎 =====================
# 匹配 GBK 误读 UTF-8 产生的典型乱码特征（如连续的非 CJK 异常字节）
GARBLED_PATTERN = re.compile(r"(?:[\x80-\xff]{3,}|[À-ÿ]{2,})")


def is_garbled(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False
    return bool(GARBLED_PATTERN.search(text))


# ===================== 1. ChromaDB 全集合清洗 =====================
def clean_chroma_db():
    if not os.path.exists(CHROMA_PATH):
        logger.warning(f"⚠️ 未找到 ChromaDB 目录: {CHROMA_PATH}，跳过向量库清洗。")
        return

    logger.info(f"🔍 正在连接 ChromaDB: {CHROMA_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # 自动获取你代码中创建的三个集合
    target_collections = ["novel_skills", "sensory_details", "classic_excerpts"]
    total_deleted = 0

    for col_name in target_collections:
        try:
            collection = client.get_collection(name=col_name)
        except Exception:
            logger.warning(f"⚠️ 集合 '{col_name}' 不存在，跳过。")
            continue

        all_data = collection.get(include=["documents", "metadatas"])
        ids = all_data["ids"]
        documents = all_data["documents"]
        metadatas = all_data["metadatas"]

        garbage_ids = []
        for i in range(len(ids)):
            doc_text = documents[i] or ""
            # 同时检查 metadata 中的文本字段（如 analysis, emotion 等）
            meta_text = " ".join(
                str(v) for v in (metadatas[i] or {}).values() if isinstance(v, str)
            )

            if is_garbled(doc_text) or is_garbled(meta_text):
                garbage_ids.append(ids[i])

        if garbage_ids:
            collection.delete(ids=garbage_ids)
            # 【已修复】将 logger.success 改为 logger.info
            logger.info(
                f"✅ [{col_name}] 删除 {len(garbage_ids)} 条乱码向量，剩余 {len(ids) - len(garbage_ids)} 条。"
            )
            total_deleted += len(garbage_ids)
        else:
            logger.info(f"✨ [{col_name}] 非常干净，无需清理。")

    logger.info(f"🎉 ChromaDB 清洗完毕，共清理 {total_deleted} 条脏数据。")


# ===================== 2. SQLite 多表安全清洗 =====================
def clean_sqlite_db():
    if not os.path.exists(SQLITE_PATH):
        logger.warning(f"⚠️ 未找到 SQLite 数据库: {SQLITE_PATH}，跳过关系库清洗。")
        return

    logger.info(f"🔍 正在连接 SQLite: {SQLITE_PATH}")
    conn = sqlite3.connect(SQLITE_PATH)
    cursor = conn.cursor()

    # 表名 -> 需要检测的文本字段列表
    table_configs = {
        "skills": {
            "fields": ["skill_name", "analysis", "original_example"],
            "action": "DELETE",
            "pk": "id",
        },
        "sensory_mappings": {
            "fields": ["emotion", "show_not_tell", "analysis"],
            "action": "DELETE",
            "pk": "id",
        },
        "plot_arcs": {
            "fields": ["summary", "character_state_json"],
            "action": "UPDATE_NULL",
            "pk": "chapter_id",
        },
        "author_fingerprints": {
            "fields": ["verbs", "adjectives", "imagery", "transitions"],
            "action": "UPDATE_NULL",
            "pk": "id",
        },
    }

    total_affected = 0
    for table, config in table_configs.items():
        # 检查表是否存在
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'"
        )
        if not cursor.fetchone():
            continue

        fields = config["fields"]
        pk = config["pk"]

        # 构建查询，拉取所有相关文本
        select_cols = [pk] + fields
        cursor.execute(f"SELECT {', '.join(select_cols)} FROM {table}")
        rows = cursor.fetchall()

        dirty_pks = []
        dirty_field_map = {}

        for row in rows:
            row_pk = row[0]
            has_garbage = False
            dirty_fields = []

            for idx, field in enumerate(fields):
                val = row[idx + 1]
                if is_garbled(str(val) if val else ""):
                    has_garbage = True
                    dirty_fields.append(field)

            if has_garbage:
                dirty_pks.append(row_pk)
                dirty_field_map[row_pk] = dirty_fields

        if not dirty_pks:
            logger.info(f"✨ [{table}] 非常干净，无需清理。")
            continue

        if config["action"] == "DELETE":
            placeholders = ",".join("?" * len(dirty_pks))
            cursor.execute(
                f"DELETE FROM {table} WHERE {pk} IN ({placeholders})", dirty_pks
            )
            logger.warning(f"🗑️ [{table}] 删除 {len(dirty_pks)} 条包含乱码的附属记录。")

        elif config["action"] == "UPDATE_NULL":
            update_count = 0
            for row_pk, d_fields in dirty_field_map.items():
                set_clause = ", ".join([f"{f} = ''" for f in d_fields])
                cursor.execute(
                    f"UPDATE {table} SET {set_clause} WHERE {pk} = ?", (row_pk,)
                )
                update_count += 1
            logger.warning(
                f"🧽 [{table}] 保留主键，清空了 {update_count} 条记录中的乱码字段。"
            )

        total_affected += len(dirty_pks)

    conn.commit()
    conn.close()
    logger.info(f"🎉 SQLite 清洗完毕，共处理 {total_affected} 条脏数据。")


if __name__ == "__main__":
    logger.info("🚀 启动全库乱码清洗引擎...")
    clean_chroma_db()
    clean_sqlite_db()
    logger.info("🏁 全库清洗任务圆满完成！现在你的知识库已经纯净无暇。")
