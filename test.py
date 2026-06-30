import os

# 🌟 核心修复：强制开启离线模式，禁止模型偷偷联网检查更新
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer
import time

print("正在将 bge-m3 模型加载到 RTX 5060 Ti 中 (离线模式)...")
# 加载模型
model = SentenceTransformer("BAAI/bge-m3", device="cuda")

print(f"✅ 模型已成功运行在: {model.device}")

sentences = [
    "萧炎看着手中的玄重尺，眼中闪过一丝坚毅。",
    "三十年河东，三十年河西，莫欺少年穷！",
    "Knowledge Management System Test",
    "这是一只可爱的猫",
]

print("\n正在使用 GPU 生成文本向量...")
start_time = time.time()

# 生成向量
embeddings = model.encode(sentences)

end_time = time.time()

print(f"✅ 向量生成成功！")
print(f"📊 向量矩阵维度: {embeddings.shape} (4句话，每句1024维)")
print(f"⏱️ 耗时: {end_time - start_time:.4f} 秒")
