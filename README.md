# 小说知识库构建系统

基于本地大模型（Ollama）的小说知识库自动构建系统，从标杆小说作品中提取世界观、人物档案、大纲结构、写作风格、对话/描写样本等多维度知识，为 AI 辅助小说创作提供 **RAG 知识外挂** 支持。

---

## 真实需求定位（重要，请勿偏移）

> **本知识库是"纯知识外挂"，不是创作管理平台。**
> **本知识库是"全类型通用"，不是"偏网文向"。**

### 核心设计原则

1. **全类型通用**：支持网文、传统文学、科幻、悬疑、历史、言情、武侠、奇幻等所有小说类型
   - Prompt 不使用网文特有术语（如"打脸/升级/爽点"），改用通用术语（如"高潮/张力点/情感爆发"）
   - 场景类型、技法分类根据 `category` 字段自适应
   - 类型特定元素（如感情线/线索推理/升级体系）按需提取，没有则返回空数组

2. **纯知识外挂**：只存储标杆作品的结构化知识，不存储用户创作数据
   - 输入：标杆小说作品（如《斗破苍穹》《诡秘之主》《三体》《白夜行》等顶尖作品）
   - 输出：多维度结构化知识 + 向量语义索引
   - 服务对象：Reasonix 的创作 Skill（通过 FastAPI 接口调用）

3. **高质量提取**：Prompt 设计确保提取的数据有用、详细、能作为创作参考
   - 所有原文摘录必须原封不动复制，禁止改写
   - 关键维度（如人物弧光、冲突升级、信息管理策略）必须深度分析
   - 提供原文示例和技法分析，便于 Skill 对标学习

### 核心工作流

```
用户在 Reasonix 中创作
        │
        ▼
Reasonix 创作 Skill 调用 DeepSeek API 生成初稿
（世界观/人物档案/写作风格指南/大纲/细纲/正文）
        │
        ▼
Reasonix 创作 Skill 调用本知识库 FastAPI 搜索标杆知识
（按创作维度/写作技法/人物类型/语义搜索）
        │
        ▼
Skill 将标杆知识与初稿对比，完善/打磨/优化创作内容
```

### 本知识库的角色

- **输入**：标杆小说作品（如《斗破苍穹》《诡秘之主》等顶尖作品）
- **输出**：从标杆作品中提取的多维度结构化知识 + 向量语义索引
- **服务对象**：Reasonix 的创作 Skill（通过 FastAPI 接口调用）
- **价值**：让 Skill 在创作时能随时查阅"顶尖小说家是怎么写的"，对标优化

### 本知识库 **不做** 的事

- ❌ 不存储用户的创作项目数据（创作数据在 Reasonix 管理）
- ❌ 不管理用户小说的版本迭代
- ❌ 不替代创作写作过程（那是 Reasonix + DeepSeek 的职责）
- ❌ 不存储用户的章节细纲、风格指南等创作产物

---

## 知识库能解决什么

### 1. 世界观创作辅助
> 场景：Skill 生成了一份力量体系设定初稿，想看看标杆作品是怎么设计力量体系的

- `GET /api/kb/search/world?module=力量体系` → 返回标杆作品的力量体系设计
- `GET /api/kb/search/world?module=社会阶层&book_name=诡秘之主` → 返回特定作品的社会结构

### 2. 人物档案创作辅助
> 场景：Skill 设计了一个反派角色，想参考标杆作品的反派是怎么塑造的

- `GET /api/kb/search/character?role_type=反派` → 返回标杆作品的反派角色档案
- `GET /api/kb/search/character?query=复杂动机的高智商反派` → 语义搜索最相似的角色

### 3. 写作风格/技法参考
> 场景：Skill 写了一段对话，想看看标杆作品怎么处理对话中的潜台词

- `GET /api/kb/search/style?technique_type=dialogue` → 返回对话样本+潜台词分析
- `GET /api/kb/search/style?query=如何写好打脸爽点` → 语义搜索相关技法

### 4. 大纲/结构设计参考
> 场景：Skill 在设计三幕结构，想参考标杆作品的结构编排

- `GET /api/kb/search/plot?structure_type=三幕结构` → 返回标杆作品的结构分析
- `GET /api/kb/search/plot?query=冲突升级方式` → 语义搜索冲突设计

### 5. 正文范文参考
> 场景：Skill 要写一段打斗场景，想看看标杆作品的打斗是怎么写的

- `GET /api/kb/search/excerpt?scene_type=打斗` → 返回打斗场景的范文段落
- `GET /api/kb/search/excerpt?query=紧张氛围的环境描写` → 语义搜索相关描写

### 6. 综合语义搜索（最强大）
> 场景：Skill 生成了一段世界观设定初稿，想让知识库返回最相似的标杆设定供参考

- `POST /api/kb/search/comprehensive` + 发送初稿文本 → 按多维度返回最相关的标杆知识

### 7. 正文质量评审（对标标杆）
> 场景：Skill 写完了一章正文，想和标杆作品对比，找出差距

- `POST /api/kb/review` + 发送章节正文 → 多维度打分 + 问题标记 + 修改建议 + 改写示范

### 8. 知识库引用推荐
> 场景：Skill 开始一个新项目，想知道哪些标杆作品最值得参考

- `POST /api/kb/recommend` + 发送题材/类型 → 按维度推荐最相关的标杆作品

### 9. 跨书对比分析（高级功能）
> 场景：Skill 想了解"顶尖作者们在感情线设计上是怎么处理的，有什么共同规律"

- `POST /api/kb/compare` + 发送对比维度（如"感情线设计"、"高潮铺垫方式"） → 返回多本书的共同模式、各自特色和最佳实践建议

### 10. 常见错误模式查询（高级功能）
> 场景：Skill 写完一段对话，想检查是否犯了常见错误

- `GET /api/kb/mistakes?dimension=对话` → 返回对话维度的常见错误模式、典型表现、修正方向和标杆范文

### 11. 上下文感知推荐（高级功能）
> 场景：Skill 正在写一段打斗，不知道"该参考什么"，让知识库主动推荐

- `POST /api/kb/context-push` + 发送当前创作内容片段 → 自动识别场景类型，推送最相关的范文、技法、结构建议和错误警告

### 12. 技法组合模板查询（高级功能）
> 场景：Skill 想学习"一组技法如何组合使用"，而非单个技法

- `GET /api/kb/combos?scene_type=打斗` → 返回打斗场景的技法组合模板（如：铺垫轻视→主角沉默→实力展示→旁观者震惊），包括技法序列、每个技法的作用、适用场景和变体建议

### 13. 高潮段落/名场面搜索
> 场景：Skill 创作高潮段落时，想参考标杆作品的名场面写法

- `GET /api/kb/search/climax?excerpt_type=决战` → 返回决战类型的高潮段落原文+技法分析
- `GET /api/kb/search/climax?query=情感爆发` → 语义搜索最相关的高潮段落

### 14. 金句/名句搜索
> 场景：Skill 创作时想参考标杆作品的经典台词和哲理句

- `GET /api/kb/search/quotes?quote_type=哲理句` → 返回哲理类金句+上下文+技法分析
- `GET /api/kb/search/quotes?query=经典台词` → 语义搜索最相关的名句

### 15. 书籍统计指标查询
> 场景：Skill 想了解标杆作品的量化写作特征（字数、对话占比、节奏模式等）

- `GET /api/style/statistics?book_name=诡秘之主` → 返回该书的总字数、平均章节字数、对话占比、段落长度分布、节奏模式等量化指标

## 硬件要求

- **显存**: 16G（如 RTX 4080/4090）
- **内存**: 16G+
- **磁盘**: 根据小说库规模，建议 50G+

## 模型要求

通过 Ollama 安装以下模型：

```bash
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b
ollama pull qwen14b:latest
```

## 模型分配策略

| Stage D | 世界观/人物/编年史 | qwen14b:latest | 2 |
| Stage E | 宏观大纲/卷节拍 | qwen14b:latest | 2 |
| Stage F | 多类型样本库 | qwen14b:latest | 2 |

## 进入你的项目目录并创建虚拟环境
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```


## 安装

```bash
pip install -r requirements.txt
```

## 使用

### 构建知识库

1. 修改 `novel_analyzer.py` 中的 `NOVELS_ROOT_DIR` 为你的小说库目录
2. 确保 Ollama 服务已启动
3. 运行：

```bash
python novel_analyzer.py
```

### 启动 API 服务

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

API 文档访问：`http://localhost:8000/docs`

## 项目结构

```
novel-knowledge-base/
├── novel_analyzer.py              # 主入口
├── run_advanced_stages.py         # 高级功能独立执行脚本（Stage L/M/N）
├── config/
│   └── settings.py                # 全局配置
├── core/
│   ├── ollama_client.py           # Ollama API 封装
│   ├── db.py                      # SQLite 数据库管理
│   ├── chroma_client.py           # ChromaDB 向量库管理
│   ├── graph.py                   # 知识图谱管理
│   ├── context_analyzer.py        # 上下文分析模块（场景识别+查询策略映射）
│   └── utils.py                   # 通用工具函数
├── stages/
│   ├── base.py                    # Stage 基类
│   ├── stage_a.py                 # 剧情摘要与人物状态（含关键事件/场景切换）
│   ├── stage_b.py                 # 技法与高潮/张力点（全类型通用术语）
│   ├── stage_c.py                 # 文风指纹与感官映射
│   ├── stage_d.py                 # 世界观与人物深度提取（自适应题材）
│   ├── stage_e.py                 # 宏观大纲与章节功能（按卷批量处理）
│   ├── stage_f.py                 # 对话/描写/动作样本库（代码截取开头/结尾）
│   ├── stage_g.py                 # 人物深度特征
│   ├── stage_h.py                 # 全书宏观分析（分3组独立调用）
│   ├── stage_i.py                 # 纯统计模块（字数/对话占比/段落分布/节奏模式）
│   ├── stage_j.py                 # 正文质量评审（对标知识库标杆）
│   ├── stage_k.py                 # 知识库引用推荐
│   ├── stage_l.py                 # 跨书对比分析（多书共同模式+最佳实践）
│   ├── stage_m.py                 # 常见错误模式提取（从评审历史归纳）
│   └── stage_n.py                 # 技法组合模板提取（技法序列+作用+变体）
├── api/
│   ├── main.py                    # FastAPI 主应用
│   ├── schemas.py                 # Pydantic 数据模型
│   └── routes/
│       ├── worldbuilding.py       # 世界观查询
│       ├── character.py           # 人物查询
│       ├── plot.py                # 大纲/细纲查询
│       ├── style.py               # 写作风格查询
│       ├── excerpt.py             # 样本库语义搜索
│       └── creative.py            # 知识库搜索引擎（面向 Reasonix Skill）
└── requirements.txt
```

## 知识库覆盖维度

### 世界观设定（7 维度）
力量体系/科技树、日常生活体系、禁忌与边界、冲突根源图谱、地理空间拓扑、经济与资源体系、语言与文化符号

### 人物档案（8 维度）
基础信息、性格画像、欲望vs需求、秘密、恐惧、社交面具、成长代价、语言风格样本、行为标志样本、关系动态演变

### 大纲结构
三幕/多幕结构、主线/支线分离、情感曲线、高潮/张力点分布图、象征体系、伏笔追踪、章节功能分类、章末钩子模式、信息管理策略、高潮构建链、冲突升级阶梯

### 写作风格
文风指纹（动词/形容词/意象/视角/节奏/禁忌词）、感官映射、对话样本库（5种场景）、描写样本库（5种类型）

## API 接口概览

### 知识库搜索引擎（面向 Reasonix 创作 Skill）

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/kb/search/world` | GET | 世界观知识搜索（按模块/语义） |
| `/api/kb/search/character` | GET | 人物档案搜索（按角色类型/语义） |
| `/api/kb/search/style` | GET | 写作风格搜索（按技法类型/语义） |
| `/api/kb/search/plot` | GET | 大纲/结构搜索（按结构类型/语义） |
| `/api/kb/search/excerpt` | GET | 正文样本搜索（按场景/描写类型/语义） |
| `/api/kb/search/comprehensive` | POST | 综合语义搜索（发送文本，返回多维度标杆知识） |
| `/api/kb/search/by-book` | GET | 按书名检索全部知识概览 |
| `/api/kb/review` | POST | 正文质量评审（对标知识库标杆） |
| `/api/kb/review/{project}/{chapter}` | GET | 评审结果查询 |
| `/api/kb/recommend` | POST | 知识库引用推荐（按题材匹配） |
| `/api/kb/compare` | POST | 跨书对比分析（多书共同模式+最佳实践） |
| `/api/kb/mistakes` | GET | 常见错误模式查询（典型表现+修正方向） |
| `/api/kb/context-push` | POST | 上下文感知推荐（自动识别场景+推送相关知识） |
| `/api/kb/combos` | GET | 技法组合模板查询（技法序列+作用+变体） |
| `/api/kb/search/climax` | GET | 高潮段落/名场面搜索（按类型/语义） |
| `/api/kb/search/quotes` | GET | 金句/名句搜索（按类型/语义） |
| `/api/kb/search-history` | GET | 搜索历史查询 |

### 知识库维度查询（按维度精查）

| 分类 | 接口 | 说明 |
|------|------|------|
| 世界观 | `GET /api/worldbuilding/settings` | 查询世界观设定 |
| 世界观 | `GET /api/worldbuilding/timeline` | 查询编年史 |
| 世界观 | `GET /api/worldbuilding/search` | 语义搜索世界观 |
| 世界观 | `GET /api/worldbuilding/factions` | 查询势力关系网络 |
| 世界观 | `GET /api/worldbuilding/setting-evolutions` | 查询设定演变追踪 |
| 人物 | `GET /api/character/profile` | 查询人物档案 |
| 人物 | `GET /api/character/speech-style` | 查询语言风格 |
| 人物 | `GET /api/character/behavior` | 查询行为标志 |
| 人物 | `GET /api/character/relationship` | 查询关系动态 |
| 大纲 | `GET /api/plot/structure` | 查询全书结构 |
| 大纲 | `GET /api/plot/main-line` | 查询主线剧情 |
| 大纲 | `GET /api/plot/subplots` | 查询支线剧情 |
| 大纲 | `GET /api/plot/emotional-arc` | 查询情感曲线 |
| 大纲 | `GET /api/plot/cool-points` | 查询爽点分布 |
| 大纲 | `GET /api/plot/foreshadowing` | 查询伏笔追踪 |
| 大纲 | `GET /api/plot/symbols` | 查询象征体系 |
| 大纲 | `GET /api/plot/chapter-functions` | 查询章节功能 |
| 大纲 | `GET /api/plot/volume-outlines` | 查询卷大纲 |
| 大纲 | `GET /api/plot/information-management` | 查询信息管理策略 |
| 大纲 | `GET /api/plot/climax-buildup` | 查询高潮构建链 |
| 大纲 | `GET /api/plot/conflict-escalation` | 查询冲突升级阶梯 |
| 风格 | `GET /api/style/fingerprint` | 查询文风指纹 |
| 风格 | `GET /api/style/sensory` | 查询感官映射 |
| 风格 | `GET /api/style/dialogue-samples` | 查询对话样本 |
| 风格 | `GET /api/style/description-samples` | 查询描写样本 |
| 风格 | `GET /api/style/skills` | 查询叙事技法 |
| 风格 | `GET /api/style/narrative-distance` | 查询叙事距离控制 |
| 风格 | `GET /api/style/show-tell` | 查询 Show vs Tell 策略 |
| 样本 | `POST /api/excerpt/search` | 语义搜索全部样本 |
| 样本 | `GET /api/excerpt/dialogue` | 搜索对话样本 |
| 样本 | `GET /api/excerpt/description` | 搜索描写样本 |
| 样本 | `GET /api/excerpt/classic` | 搜索经典摘录 |
