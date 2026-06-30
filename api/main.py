"""
FastAPI 主应用
提供小说知识库的 REST API 接口
"""
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import worldbuilding, character, plot, style, excerpt, creative

app = FastAPI(
    title="小说知识库 API",
    description="为小说创作全流程提供知识库查询服务",
    version="2.0.0",
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(worldbuilding.router, prefix="/api/worldbuilding", tags=["世界观"])
app.include_router(character.router, prefix="/api/character", tags=["人物"])
app.include_router(plot.router, prefix="/api/plot", tags=["大纲/细纲"])
app.include_router(style.router, prefix="/api/style", tags=["写作风格"])
app.include_router(excerpt.router, prefix="/api/excerpt", tags=["样本库"])
app.include_router(creative.router, prefix="/api/kb", tags=["知识库搜索"])


@app.get("/")
def root():
    return {
        "message": "小说知识库 API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}
