"""
Stage O: 事件因果图谱
使用 qwen14b:latest 模型，从 Stage A 的逐章摘要中提取关键事件，
分析事件之间的因果关系，构建有向因果图。
"""
import json
import logging
from typing import List, Dict, Any
from tqdm import tqdm
from stages.base import BaseStage
from core.ollama_client import ollama_chat, safe_parse_json
from core.utils import generate_id

logger = logging.getLogger(__name__)


class StageO(BaseStage):
    """Stage O: 事件因果图谱"""

    # 事件提取批次大小（每批处理的章节数）
    EVENT_EXTRACTION_BATCH = 10
    # 因果分析批次大小（每批分析的事件数）
    CAUSAL_ANALYSIS_BATCH = 50

    def __init__(self, book_name: str, category: str):
        super().__init__("O", book_name, category)

    def run(self, stage_a_res: List[Dict], **kwargs) -> Dict[str, List[Dict]]:
        """
        执行 Stage O（支持断点续跑）

        Args:
            stage_a_res: Stage A 的章节摘要结果（每章含 summary 和 key_events）

        Returns:
            {"story_events": [...], "event_causal_edges": [...]}
        """
        logger.info(f"=== 阶段O：事件因果图谱构建 ({self.book_name}) ===")

        # 断点恢复
        cache = self.load_cache()
        story_events = []
        event_causal_edges = []
        events_done = False
        causal_done = False

        if cache:
            story_events = cache.get("story_events", [])
            event_causal_edges = cache.get("event_causal_edges", [])
            events_done = cache.get("events_done", False)
            causal_done = cache.get("causal_done", False)
            if events_done:
                logger.info(f"✅ [阶段O] 恢复断点：事件提取已完成 ({len(story_events)} 个事件)")
            if causal_done:
                logger.info(f"✅ [阶段O] 恢复断点：因果分析已完成 ({len(event_causal_edges)} 条因果边)")

        # Step 1: 事件提取（如果未完成）
        if not events_done:
            story_events = self._extract_events(stage_a_res)
            logger.info(f"[阶段O] 提取到 {len(story_events)} 个关键事件")
            # 保存事件提取断点
            self.save_cache({
                "story_events": story_events,
                "event_causal_edges": [],
                "events_done": True,
                "causal_done": False,
            })
        else:
            logger.info(f"[阶段O] 事件提取已从断点恢复 ({len(story_events)} 个事件)")

        if len(story_events) < 3:
            logger.warning("[阶段O] 事件数量过少，跳过因果分析")
            return {"story_events": story_events, "event_causal_edges": []}

        # Step 2: 因果分析（如果未完成）
        if not causal_done:
            event_causal_edges = self._analyze_causal_relations(story_events)
            logger.info(f"[阶段O] 分析出 {len(event_causal_edges)} 条因果关系")
            # 保存最终断点
            self.save_cache({
                "story_events": story_events,
                "event_causal_edges": event_causal_edges,
                "events_done": True,
                "causal_done": True,
            })
        else:
            logger.info(f"[阶段O] 因果分析已从断点恢复 ({len(event_causal_edges)} 条因果边)")

        return {
            "story_events": story_events,
            "event_causal_edges": event_causal_edges,
        }

    def _extract_events(self, stage_a_res: List[Dict]) -> List[Dict]:
        """
        Step 1: 从章节摘要中批量提取关键事件
        """
        all_events = []

        # 分批处理章节摘要
        batches = [
            stage_a_res[i : i + self.EVENT_EXTRACTION_BATCH]
            for i in range(0, len(stage_a_res), self.EVENT_EXTRACTION_BATCH)
        ]

        for batch in tqdm(batches, desc="[阶段O] 事件提取"):
            # 构建批次摘要文本
            batch_text = ""
            for chap in batch:
                chap_id = chap.get("id", "未知章节")
                summary = chap.get("summary", "")
                key_events = chap.get("key_events", [])

                if not summary and not key_events:
                    continue

                batch_text += f"\n【{chap_id}】\n摘要：{summary}\n"
                if key_events:
                    events_str = "; ".join(
                        [
                            f"{e.get('event', '')}(影响:{e.get('impact', '')})"
                            for e in key_events
                            if isinstance(e, dict)
                        ]
                    )
                    batch_text += f"关键事件：{events_str}\n"

            if not batch_text.strip():
                continue

            # 截断以避免超出上下文
            if len(batch_text) > 6000:
                batch_text = batch_text[:6000] + "\n...(截断)"

            prompt = f"""你是专业的叙事分析师。从以下章节摘要中提取所有关键事件。

书名：{self.book_name}  分类：{self.category}

{batch_text}

请输出纯JSON格式：
{{
  "events": [
    {{
      "chapter_id": "章节ID",
      "event_name": "事件名称（10字以内，简洁概括）",
      "event_summary": "事件详细描述（50字以内）",
      "event_type": "事件类型（伏笔埋设/伏笔回收/冲突爆发/角色转变/世界规则揭示/高潮/转折/日常推进）",
      "characters_involved": ["涉及的角色名"],
      "significance": "重要性（high/medium/low）"
    }}
  ]
}}

要求：
1. 每章至少提取1个事件，重要章节可提取2-3个
2. event_type 必须从给定选项中选择
3. 只提取对剧情有实质影响的事件，忽略纯过渡性内容
4. significance=high 的事件是对整个故事有深远影响的关键节点
5. 禁止使用反引号"""

            try:
                resp = ollama_chat(prompt, 0.2, "O")
                data = safe_parse_json(resp)
                if not data:
                    continue

                for event in data.get("events", []):
                    if not event.get("event_name"):
                        continue
                    event["book_name"] = self.book_name
                    event["id"] = generate_id(
                        self.book_name,
                        event.get("chapter_id", ""),
                        event["event_name"],
                    )
                    all_events.append(event)

            except Exception as e:
                logger.warning(f"[阶段O] 事件提取批次失败: {e}")

        return all_events

    def _analyze_causal_relations(
        self, story_events: List[Dict]
    ) -> List[Dict]:
        """
        Step 2: 分析事件之间的因果关系
        """
        all_edges = []

        # 构建事件索引（按章节排序）
        sorted_events = sorted(
            story_events,
            key=lambda e: self._chapter_sort_key(e.get("chapter_id", "")),
        )

        # 分批进行因果分析（使用滑动窗口，每批与前一批重叠10个事件以捕捉跨批因果）
        overlap = 10
        step = self.CAUSAL_ANALYSIS_BATCH - overlap
        batches = [
            sorted_events[i : i + self.CAUSAL_ANALYSIS_BATCH]
            for i in range(0, max(len(sorted_events), 1), step)
        ]

        for batch in tqdm(batches, desc="[阶段O] 因果分析"):
            if len(batch) < 2:
                continue

            # 构建事件列表文本（用编号作为唯一标识，避免同名事件歧义）
            events_text = ""
            for i, event in enumerate(batch):
                events_text += (
                    f"#{i+1} [{event.get('chapter_id', '')}] "
                    f"{event['event_name']}: {event.get('event_summary', '')} "
                    f"(类型:{event.get('event_type', '')}, "
                    f"角色:{','.join(event.get('characters_involved', []))})\n"
                )

            # 动态截断：滑动窗口批次可达60事件，按事件数调整字符预算
            max_causal_chars = min(8000, len(batch) * 200 + 1000)
            if len(events_text) > max_causal_chars:
                events_text = events_text[:max_causal_chars] + "\n...(截断)"

            prompt = f"""你是专业的叙事因果分析师。分析以下事件之间的因果关系。

书名：{self.book_name}  分类：{self.category}

事件列表：
{events_text}

请输出纯JSON格式：
{{
  "causal_edges": [
    {{
      "source_index": 1,
      "target_index": 3,
      "relation_type": "关系类型（直接导致/铺垫/触发/回应/并行/对比）",
      "relation_detail": "因果关系的具体描述（30字以内）"
    }}
  ]
}}

要求：
1. 只分析列表中事件之间的关系，不要引入列表外的事件
2. source_index 和 target_index 是事件列表中的编号（#后面的数字）
3. relation_type 说明：
   - 直接导致：A 直接引发 B（如“获得硬币” → “硬币激活”）
   - 铺垫：A 为 B 创造条件（如“建立信任” → “信任被考验”）
   - 触发：A 是 B 的导火索（如“发现秘密” → “愤怒质问”）
   - 回应：B 是对 A 的反应（如“背叛” → “复仇”）
   - 并行：A 和 B 同时发生且互相影响
   - 对比：A 和 B 形成叙事对比
4. 不要强行建立因果关系，只分析确实存在的因果链
5. 禁止使用反引号"""

            try:
                resp = ollama_chat(prompt, 0.2, "O")
                data = safe_parse_json(resp)
                if not data:
                    continue

                # 构建事件索引映射（编号 -> 事件）
                index_to_event = {i + 1: e for i, e in enumerate(batch)}

                for edge in data.get("causal_edges", []):
                    source_idx = edge.get("source_index")
                    target_idx = edge.get("target_index")

                    # 强制转为整数，兼容 LLM 返回字符串的情况
                    try:
                        source_idx = int(source_idx)
                        target_idx = int(target_idx)
                    except (TypeError, ValueError):
                        continue

                    # 用编号精确匹配事件，避免同名事件歧义
                    source_event = index_to_event.get(source_idx)
                    target_event = index_to_event.get(target_idx)

                    if not source_event or not target_event:
                        continue  # 跳过无法匹配的事件

                    source_id = source_event["id"]
                    target_id = target_event["id"]

                    edge_id = generate_id(
                        self.book_name, source_id, target_id
                    )
                    all_edges.append(
                        {
                            "id": edge_id,
                            "book_name": self.book_name,
                            "source_event_id": source_id,
                            "target_event_id": target_id,
                            "relation_type": edge.get(
                                "relation_type", "未知"
                            ),
                            "relation_detail": edge.get(
                                "relation_detail", ""
                            ),
                        }
                    )

            except Exception as e:
                logger.warning(f"[阶段O] 因果分析批次失败: {e}")

        return all_edges

    @staticmethod
    def _chapter_sort_key(chapter_id: str) -> float:
        """将章节 ID 转为可排序的数值键"""
        import re

        match = re.search(r"(\d+)", chapter_id)
        if match:
            return float(match.group(1))
        return 9999.0

    def insert(self, results: Dict[str, List[Dict]]) -> Dict[str, int]:
        """将事件因果图谱写入数据库"""
        cursor = self.db.connect().cursor()
        stats = {"story_events": 0, "event_causal_edges": 0}

        # 1. 写入 story_events 表
        for event in results.get("story_events", []):
            cursor.execute(
                "INSERT OR REPLACE INTO story_events VALUES (?,?,?,?,?,?,?,?)",
                (
                    event["id"],
                    event["book_name"],
                    event.get("chapter_id", ""),
                    event["event_name"],
                    event.get("event_summary", ""),
                    event.get("event_type", ""),
                    json.dumps(
                        event.get("characters_involved", []),
                        ensure_ascii=False,
                    ),
                    event.get("significance", ""),
                ),
            )
            stats["story_events"] += 1

        # 2. 写入 event_causal_edges 表
        for edge in results.get("event_causal_edges", []):
            cursor.execute(
                "INSERT OR REPLACE INTO event_causal_edges VALUES (?,?,?,?,?,?)",
                (
                    edge["id"],
                    edge["book_name"],
                    edge["source_event_id"],
                    edge["target_event_id"],
                    edge.get("relation_type", ""),
                    edge.get("relation_detail", ""),
                ),
            )
            stats["event_causal_edges"] += 1

        self.db.commit()

        # 3. 写入 ChromaDB（事件向量库）
        self._upsert_events_to_chroma(results.get("story_events", []))
        # 4. 写入 ChromaDB（因果关系向量库）
        self._upsert_edges_to_chroma(
            results.get("event_causal_edges", []),
            results.get("story_events", []),
        )
        # 5. 写入 NetworkX 图谱
        self._add_to_graph(
            results.get("story_events", []),
            results.get("event_causal_edges", []),
        )

        logger.info(
            f"   [阶段O战报] 事件: {stats['story_events']} 个 | "
            f"因果边: {stats['event_causal_edges']} 条"
        )
        return stats

    def _upsert_events_to_chroma(self, events: List[Dict]):
        """将事件写入 ChromaDB 向量库"""
        if not events:
            return

        ids, docs, metas = [], [], []
        for event in events:
            ids.append(event["id"])
            docs.append(
                f"事件:{event['event_name']}\n"
                f"摘要:{event.get('event_summary', '')}\n"
                f"类型:{event.get('event_type', '')}\n"
                f"角色:{','.join(event.get('characters_involved', []))}\n"
                f"重要性:{event.get('significance', '')}"
            )
            metas.append(
                {
                    "book_name": event["book_name"],
                    "chapter_id": event.get("chapter_id", ""),
                    "event_type": event.get("event_type", ""),
                    "significance": event.get("significance", ""),
                }
            )

        if ids:
            self.chroma.upsert_batch(
                "story_events_kb", ids, docs, metas
            )

    def _upsert_edges_to_chroma(
        self, edges: List[Dict], events: List[Dict]
    ):
        """将因果关系写入 ChromaDB 向量库"""
        if not edges:
            return

        # 构建事件 ID 到名称的映射
        id_to_name = {e["id"]: e["event_name"] for e in events}

        ids, docs, metas = [], [], []
        for edge in edges:
            source_name = id_to_name.get(
                edge["source_event_id"], "未知事件"
            )
            target_name = id_to_name.get(
                edge["target_event_id"], "未知事件"
            )

            ids.append(edge["id"])
            docs.append(
                f"因:{source_name}\n"
                f"果:{target_name}\n"
                f"关系:{edge.get('relation_type', '')}\n"
                f"详情:{edge.get('relation_detail', '')}"
            )
            metas.append(
                {
                    "book_name": edge["book_name"],
                    "relation_type": edge.get("relation_type", ""),
                    "source_event": source_name,
                    "target_event": target_name,
                }
            )

        if ids:
            self.chroma.upsert_batch(
                "event_causal_edges_kb", ids, docs, metas
            )

    def _add_to_graph(
        self, events: List[Dict], edges: List[Dict]
    ):
        """将事件和因果关系添加到 NetworkX 图谱"""
        if not events:
            return

        # 添加事件节点
        for event in events:
            node_id = f"event:{event['event_name']}"
            self.graph.add_node(
                node_id,
                node_type="story_event",
                book_list=self.book_name,
                chapter_id=event.get("chapter_id", ""),
                event_type=event.get("event_type", ""),
                significance=event.get("significance", ""),
            )

        # 添加因果关系边
        id_to_name = {e["id"]: e["event_name"] for e in events}
        for edge in edges:
            source_name = id_to_name.get(
                edge["source_event_id"], ""
            )
            target_name = id_to_name.get(
                edge["target_event_id"], ""
            )
            if source_name and target_name:
                self.graph.add_edge(
                    f"event:{source_name}",
                    f"event:{target_name}",
                    relation_type=edge.get("relation_type", ""),
                    book=self.book_name,
                )
