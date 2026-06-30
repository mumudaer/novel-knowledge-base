import sys
from chromadb.cli.cli import app

# 模拟在命令行输入: chroma run --path "..." --host localhost --port 8000
sys.argv = [
    "chroma", 
    "run", 
    "--path", r"D:\WorkFish\Novel-Knowledge-Base\novel_kb\chroma_db",
    "--host", "localhost",
    "--port", "8000"
]

# 启动服务
app()