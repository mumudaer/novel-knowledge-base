"""
全局配置模块
集中管理所有硬件配置、模型配置、路径配置等
"""
import os
import sys

# ===================== 全局硬件专属配置 =====================
# 硬件环境: 16G 显存 + 16G 内存
# 可用模型: qwen2.5:3b, qwen2.5:7b, qwen3:14b

# 模型分配策略（根据任务复杂度自动选择）
# 注意：Stage A 从 3b 升级为 7b，摘要质量是全链路地基，3b 质量不足会污染下游所有 Stage
STAGE_A_MODEL = "qwen2.5:7b"      # 中复杂度：摘要、状态追踪（升级为 7b 保证质量）
STAGE_B_MODEL = "qwen2.5:7b"      # 中复杂度：技法提取
STAGE_C_MODEL = "qwen2.5:7b"      # 中复杂度：文风指纹
STAGE_D_MODEL = "qwen3:14b"       # 高复杂度：世界观、人物深度提取
STAGE_E_MODEL = "qwen2.5:7b"      # 中复杂度：卷大纲聚合
STAGE_F_MODEL = "qwen3:14b"       # 高复杂度：样本提取与鉴赏
STAGE_G_MODEL = "qwen3:14b"       # 高复杂度：人物深度分析
STAGE_H_MODEL = "qwen3:14b"       # 高复杂度：全书宏观分析
STAGE_J_MODEL = "qwen3:14b"       # 高复杂度：正文质量评审（服务期按需调用，不在构建流水线中）
STAGE_K_MODEL = "qwen2.5:7b"      # 中复杂度：知识库引用推荐（服务期按需调用，不在构建流水线中）
STAGE_L_MODEL = "qwen3:14b"       # 高复杂度：跨书对比分析
STAGE_M_MODEL = "qwen3:14b"       # 高复杂度：错误模式归纳
STAGE_N_MODEL = "qwen3:14b"       # 高复杂度：技法组合分析
STAGE_CTX_MODEL = "qwen2.5:7b"    # 中复杂度：上下文场景识别

# Ollama API 配置
OLLAMA_API_URL = "http://localhost:11434/api/chat"
OLLAMA_BASE_URL = "http://localhost:11434"

# 并发配置（根据硬件限制）
# RTX 5060 Ti 16G + 16G 内存：7b 双并发约 15GB，14b 双并发接近上限
STAGE_A_WORKERS = 2  # 7b 模型（原 3b 升级），双并发
STAGE_B_WORKERS = 2  # 7b 模型，双并发
STAGE_C_WORKERS = 2  # 7b 模型，双并发
STAGE_D_WORKERS = 2  # 14b 模型，双并发（用户实测可行）
STAGE_E_WORKERS = 1  # 7b 模型，单并发（需要聚合上下文）
STAGE_F_WORKERS = 2  # 14b 模型，双并发
STAGE_G_WORKERS = 2  # 14b 模型，双并发
STAGE_H_WORKERS = 2  # 14b 模型，双并发
STAGE_J_WORKERS = 2  # 14b 模型，双并发
STAGE_K_WORKERS = 1  # 7b 模型，单并发（推荐以保证质量）
STAGE_L_WORKERS = 1  # 14b 模型，单并发（分析任务较重）
STAGE_M_WORKERS = 1  # 14b 模型，单并发（归纳任务较重）
STAGE_N_WORKERS = 1  # 14b 模型，单并发（分析任务较重）
STAGE_CTX_WORKERS = 2  # 7b 模型，双并发（轻量级场景识别）

# 上下文长度配置（根据模型能力和显存限制）
# Stage A 从 3b 升级为 7b，上下文保持 16384（双并发时显存约 15GB，在安全范围内）
OLLAMA_NUM_CTX_7B = 16384   # 7b 模型 16K 上下文（双并发时约 15GB）
OLLAMA_NUM_CTX_14B = 8192   # 14b 模型 8K 上下文（双并发时接近上限）
OLLAMA_NUM_PREDICT = 2048   # 最大生成长度
OLLAMA_TIMEOUT = 600        # 超时时间（秒）

# Embedding 模型配置（中文优化）
# bge-m3 是 BAAI 发布的多语言 embedding 模型，中文语义检索质量远优于 ChromaDB 默认的英文模型
EMBEDDING_MODEL = "bge-m3"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"  # HuggingFace 模型名（sentence-transformers 使用）
EMBEDDING_DIMENSION = 1024
EMBEDDING_DEVICE = "cuda"  # cuda 或 cpu，构建期 GPU 加速，服务期按需

# 文本切分配置
SPLIT_THRESHOLD = 5000      # 章节切分阈值（字符数），提升以更好利用 14b 模型 8192 上下文
SPLIT_OVERLAP = 200         # 二次切分时相邻块的重叠字符数，避免上下文断裂
MATCH_THRESHOLD = 85        # 模糊匹配阈值

# 数据库配置
CHROMA_BATCH_SIZE = 50      # ChromaDB 批量写入大小（降低内存峰值）
SQL_COMMIT_CHUNK = 5000     # SQLite 批量提交大小

# HTTP 连接池配置
HTTP_POOL_CONNECTIONS = 10
HTTP_POOL_MAXSIZE = 10
HTTP_MAX_RETRIES = 3

# ===================== 路径配置 =====================
if os.environ.get("NOVEL_KB_DATA_DIR"):
    BASE_DIR = os.environ["NOVEL_KB_DATA_DIR"]
else:
    if getattr(sys, "frozen", False):
        app_root = os.path.dirname(sys.executable)
    else:
        app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BASE_DIR = os.path.join(app_root, "novel_kb")

# 确保目录存在
os.makedirs(BASE_DIR, exist_ok=True)

# 数据库文件路径
SQLITE_PATH = os.path.join(BASE_DIR, "knowledge.db")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
MANIFEST_FILE = os.path.join(BASE_DIR, "process_manifest.json")
UNMATCHED_LOG = os.path.join(BASE_DIR, "unmatched_quotes.jsonl")

# ===================== 环境变量配置 =====================
# 强制设置 Ollama 环境变量
if os.environ.get("OLLAMA_NUM_PARALLEL") is None:
    os.environ["OLLAMA_NUM_PARALLEL"] = "2"
    print("\n" + "=" * 50)
    print("🚀 已开启 OLLAMA_NUM_PARALLEL = 2，压榨 16G 显存双并发性能！")
    print("=" * 50 + "\n")

# 限制 PyTorch/Ollama 抢占过多 CPU 线程
os.environ["OMP_NUM_THREADS"] = "8"

# ===================== 模型配置映射 =====================
MODEL_CONFIG = {
    "A": {
        "model": STAGE_A_MODEL,
        "workers": STAGE_A_WORKERS,
        "num_ctx": OLLAMA_NUM_CTX_7B,  # 从 3b 升级为 7b，使用 7b 上下文配置
        "temperature": 0.1,
    },
    "B": {
        "model": STAGE_B_MODEL,
        "workers": STAGE_B_WORKERS,
        "num_ctx": OLLAMA_NUM_CTX_7B,
        "temperature": 0.2,
    },
    "C": {
        "model": STAGE_C_MODEL,
        "workers": STAGE_C_WORKERS,
        "num_ctx": OLLAMA_NUM_CTX_7B,
        "temperature": 0.3,
    },
    "D": {
        "model": STAGE_D_MODEL,
        "workers": STAGE_D_WORKERS,
        "num_ctx": OLLAMA_NUM_CTX_14B,
        "temperature": 0.1,
    },
    "E": {
        "model": STAGE_E_MODEL,
        "workers": STAGE_E_WORKERS,
        "num_ctx": OLLAMA_NUM_CTX_7B,
        "temperature": 0.2,
    },
    "F": {
        "model": STAGE_F_MODEL,
        "workers": STAGE_F_WORKERS,
        "num_ctx": OLLAMA_NUM_CTX_14B,
        "temperature": 0.2,
    },
    "G": {
        "model": STAGE_G_MODEL,
        "workers": STAGE_G_WORKERS,
        "num_ctx": OLLAMA_NUM_CTX_14B,
        "temperature": 0.2,
    },
    "H": {
        "model": STAGE_H_MODEL,
        "workers": STAGE_H_WORKERS,
        "num_ctx": OLLAMA_NUM_CTX_14B,
        "temperature": 0.2,
    },
    "J": {
        "model": STAGE_J_MODEL,
        "workers": STAGE_J_WORKERS,
        "num_ctx": OLLAMA_NUM_CTX_14B,
        "temperature": 0.2,
    },
    "K": {
        "model": STAGE_K_MODEL,
        "workers": STAGE_K_WORKERS,
        "num_ctx": OLLAMA_NUM_CTX_7B,
        "temperature": 0.2,
    },
    "L": {
        "model": STAGE_L_MODEL,
        "workers": STAGE_L_WORKERS,
        "num_ctx": OLLAMA_NUM_CTX_14B,
        "temperature": 0.3,
    },
    "M": {
        "model": STAGE_M_MODEL,
        "workers": STAGE_M_WORKERS,
        "num_ctx": OLLAMA_NUM_CTX_14B,
        "temperature": 0.3,
    },
    "N": {
        "model": STAGE_N_MODEL,
        "workers": STAGE_N_WORKERS,
        "num_ctx": OLLAMA_NUM_CTX_14B,
        "temperature": 0.3,
    },
    "CTX": {
        "model": STAGE_CTX_MODEL,
        "workers": STAGE_CTX_WORKERS,
        "num_ctx": OLLAMA_NUM_CTX_7B,
        "temperature": 0.2,
    },
}


def get_model_config(stage: str) -> dict:
    """获取指定 Stage 的模型配置"""
    return MODEL_CONFIG.get(stage, MODEL_CONFIG["A"])
