"""
全局配置模块
集中管理所有硬件配置、模型配置、路径配置等
"""

import os
import sys

# ===================== 全局硬件专属配置 =====================
# 硬件环境: 16G 显存 + 16G 内存
# 可用模型: qwen2.5:3b, qwen2.5:7b, qwen14b:latest

# 模型分配策略（根据任务复杂度自动选择）
# 注意：Stage A 从 3b 升级为 7b，摘要质量是全链路地基，3b 质量不足会污染下游所有 Stage
STAGE_A_MODEL = "qwen2.5:7b"  # 9b thinking API 不兼容，回退7b
STAGE_B_MODEL = "qwen2.5:7b"  # 中复杂度：技法提取
STAGE_C_MODEL = "qwen2.5:7b"  # 中复杂度：文风指纹
STAGE_D_MODEL = "qwen14b:latest"  # 高复杂度：世界观、人物深度提取
STAGE_E_MODEL = "qwen2.5:7b"  # 中复杂度：卷大纲聚合
STAGE_F_MODEL = "qwen14b:latest"  # 高复杂度：样本提取与鉴赏
STAGE_G_MODEL = "qwen14b:latest"  # 高复杂度：人物深度分析
STAGE_H_MODEL = "qwen14b:latest"  # 高复杂度：全书宏观分析
STAGE_J_MODEL = (
    "qwen14b:latest"  # 高复杂度：正文质量评审（服务期按需调用，不在构建流水线中）
)
STAGE_K_MODEL = (
    "qwen2.5:7b"  # 中复杂度：知识库引用推荐（服务期按需调用，不在构建流水线中）
)
STAGE_L_MODEL = "qwen14b:latest"  # 高复杂度：跨书对比分析
STAGE_M_MODEL = "qwen14b:latest"  # 高复杂度：错误模式归纳
STAGE_N_MODEL = "qwen14b:latest"  # 高复杂度：技法组合分析
STAGE_O_MODEL = "qwen14b:latest"  # 高复杂度：事件因果图谱分析
STAGE_CTX_MODEL = "qwen2.5:7b"  # 中复杂度：上下文场景识别

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
STAGE_O_WORKERS = 1  # 14b 模型，单并发（全书级因果分析）
STAGE_CTX_WORKERS = 2  # 7b 模型，双并发（轻量级场景识别）

# 上下文长度配置（根据模型能力和显存限制）
# Stage A 从 3b 升级为 7b，上下文保持 16384（双并发时显存约 15GB，在安全范围内）
# 7b KV cache: 56 KB/token → 12288 tokens ≈ 0.69 GB, model ~4.5 GB → 总计 ~5.2 GB
# 3500字章节×1.5+模板3k+pred2k+buf500=10798 token, 12288 安全
OLLAMA_NUM_CTX_7B = 12288  # 7b/9b 通用上下文（足够容纳 3500 字章节 + 模板）
# 14b KV cache: 192 KB/token → 14336 tokens ≈ 2.8 GB, model ~9 GB → 总计 ~12 GB
OLLAMA_NUM_CTX_14B = (
    14336  # 配合 SPLIT_THRESHOLD=3500, 每块≤3500字完整喂给 LLM (D-char budget=4909)
)
OLLAMA_NUM_PREDICT = 2048  # 最大生成长度
OLLAMA_TIMEOUT = 300  # 超时时间（秒）（14b正常60-120s/次，300s已含2.5-5x余量）

# Embedding 模型配置（中文优化）
# bge-m3 是 BAAI 发布的多语言 embedding 模型，中文语义检索质量远优于 ChromaDB 默认的英文模型
# 注意：EMBEDDING_DEVICE 必须设为 "cpu"，因为 embedding 和 LLM 推理会同时加载，
# GPU 显存只有 16GB，bge-m3(cuda ~2GB) + qwen3:14b(~10GB) 会 OOM。
# embedding 在 CPU 上运行足够快（每批 ~50 条约 1-2 秒），不影响整体性能。
EMBEDDING_MODEL = "bge-m3"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"  # HuggingFace 模型名（sentence-transformers 使用）
EMBEDDING_DIMENSION = 1024
EMBEDDING_DEVICE = "cpu"  # 强制 CPU，避免与 LLM 推理抢占 GPU 显存

# 采样公式: sample_count = STAGE_SAMPLE_BASE + STAGE_SAMPLE_MULTIPLIER * sqrt(total / STAGE_SAMPLE_DENOMINATOR)
STAGE_SAMPLE_BASE = 10
STAGE_SAMPLE_MULTIPLIER = 5
STAGE_SAMPLE_DENOMINATOR = 100

# 文本切分配置
SPLIT_THRESHOLD = 5000  # 章节切分阈值（字符数），Stage D 模板精简后 14b@14336 可容纳
SPLIT_OVERLAP = 200  # 二次切分时相邻块的重叠字符数，避免上下文断裂
MATCH_THRESHOLD = 85  # 模糊匹配阈值

# 数据库配置
CHROMA_BATCH_SIZE = 50  # ChromaDB 批量写入大小（降低内存峰值）
SQL_COMMIT_CHUNK = 5000  # SQLite 批量提交大小

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
        "num_predict": 2048,  # 单章摘要，2048 足够
    },
    "B": {
        "model": STAGE_B_MODEL,
        "workers": STAGE_B_WORKERS,
        "num_ctx": OLLAMA_NUM_CTX_7B,
        "temperature": 0.2,
        "num_predict": 2048,  # 技法提取，中等输出
    },
    "C": {
        "model": STAGE_C_MODEL,
        "workers": STAGE_C_WORKERS,
        "num_ctx": OLLAMA_NUM_CTX_7B,
        "temperature": 0.3,
        "num_predict": 2048,  # 文风指纹，中等输出
    },
    "D": {
        "model": STAGE_D_MODEL,
        "workers": STAGE_D_WORKERS,
        "num_ctx": OLLAMA_NUM_CTX_14B,
        "temperature": 0.1,
        "num_predict": 4096,  # 33维度人物档案 + 世界观7维，小chunk(≤3500字)下够用
    },
    "E": {
        "model": STAGE_E_MODEL,
        "workers": STAGE_E_WORKERS,
        "num_ctx": OLLAMA_NUM_CTX_7B,
        "temperature": 0.2,
        "num_predict": 4096,  # 卷大纲+伏笔+状态变更，重输出
    },
    "F": {
        "model": STAGE_F_MODEL,
        "workers": STAGE_F_WORKERS,
        "num_ctx": OLLAMA_NUM_CTX_14B,
        "temperature": 0.2,
        "num_predict": 4096,  # 11种样本结构(对话/描写/转场/叙事/高潮/金句等)，重输出
    },
    "G": {
        "model": STAGE_G_MODEL,
        "workers": STAGE_G_WORKERS,
        "num_ctx": OLLAMA_NUM_CTX_14B,
        "temperature": 0.2,
        "num_predict": 3072,  # 人物深度分析(语言风格+行为标志+关系)，偏重输出
    },
    "H": {
        "model": STAGE_H_MODEL,
        "workers": STAGE_H_WORKERS,
        "num_ctx": 16384,  # 覆盖 14b 默认 14336，配合 6144 输出需更大输入窗口
        "temperature": 0.2,
        "num_predict": 6144,  # 3组×5-6种宏观结构，嵌套JSON输出重
    },
    "J": {
        "model": STAGE_J_MODEL,
        "workers": STAGE_J_WORKERS,
        "num_ctx": OLLAMA_NUM_CTX_14B,
        "temperature": 0.2,
        "num_predict": 2048,
    },
    "K": {
        "model": STAGE_K_MODEL,
        "workers": STAGE_K_WORKERS,
        "num_ctx": OLLAMA_NUM_CTX_7B,
        "temperature": 0.2,
        "num_predict": 2048,
    },
    "L": {
        "model": STAGE_L_MODEL,
        "workers": STAGE_L_WORKERS,
        "num_ctx": OLLAMA_NUM_CTX_14B,
        "temperature": 0.3,
        "num_predict": 4096,  # 跨书对比分析，重输出
    },
    "M": {
        "model": STAGE_M_MODEL,
        "workers": STAGE_M_WORKERS,
        "num_ctx": OLLAMA_NUM_CTX_14B,
        "temperature": 0.3,
        "num_predict": 4096,  # 错误模式提取，重输出
    },
    "N": {
        "model": STAGE_N_MODEL,
        "workers": STAGE_N_WORKERS,
        "num_ctx": OLLAMA_NUM_CTX_14B,
        "temperature": 0.3,
        "num_predict": 4096,  # 技法组合模板，重输出
    },
    "O": {
        "model": STAGE_O_MODEL,
        "workers": STAGE_O_WORKERS,
        "num_ctx": OLLAMA_NUM_CTX_14B,
        "temperature": 0.2,
        "num_predict": 4096,  # 多事件因果图谱，重输出
    },
    "CTX": {
        "model": STAGE_CTX_MODEL,
        "workers": STAGE_CTX_WORKERS,
        "num_ctx": OLLAMA_NUM_CTX_7B,
        "temperature": 0.2,
        "num_predict": 2048,
    },
}


def get_model_config(stage: str) -> dict:
    """获取指定 Stage 的模型配置"""
    return MODEL_CONFIG.get(stage, MODEL_CONFIG["A"])
