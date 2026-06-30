"""
Stage A: 剧情摘要与人物状态追踪
使用 qwen2.5:3b 模型，逐章生成剧情摘要和人物状态
"""
import os
import json
import logging
from typing import List, Dict, Any, Tuple, Set
from tqdm import tqdm

from stages.base import BaseStage
from core.ollama_client import ollama_chat, safe_parse_json
from core.utils import (
    get_window_file,
    save_state_atomic,
    compress_character_state,
    compress_state_to_text,
    flatten_character_state,
)

logger = logging.getLogger(__name__)


class StageA(BaseStage):
    """Stage A: 剧情摘要与人物状态追踪"""

    def __init__(self, book_name: str, category: str):
        super().__init__("A", book_name, category)
        self.window_file = get_window_file(book_name)

    def run(
        self, chapters: List[Dict], **kwargs
    ) -> Tuple[List[Dict], str, Set[str]]:
        """
        执行 Stage A

        Args:
            chapters: 章节列表

        Returns:
            (处理后的章节列表, 推断的分类, 主角名集合)
        """
        print("=== 阶段一：生成剧情上下文与智能推断分类 ===")

        processed_chaps = []
        last_char_state = {}
        finish_count = 0
        inferred_category = self.category
        protagonist_names = set()
        recent_texts = []

        # 尝试恢复全量断点
        cache = self.load_cache()
        if cache and cache.get("stage") == "A":
            cached_data = cache.get("data", [])
            if len(cached_data) <= len(chapters) and all(
                cached_data[i].get("id") == chapters[i].get("id")
                for i in range(len(cached_data))
            ):
                for i, item in enumerate(cached_data):
                    chapters[i]["summary"] = item.get("summary", "")
                    chapters[i]["character_state"] = item.get("character_state", {})
                    processed_chaps.append(chapters[i])
                finish_count = len(cached_data)
                last_char_state = processed_chaps[-1]["character_state"]
                inferred_category = cache.get("inferred_category", self.category)
                protagonist_names = set(cache.get("protagonist_names", []))
                recent_texts = [c["text"] for c in processed_chaps[-3:]]
                print(f"✅ [阶段A] 恢复全量断点：从第 {finish_count + 1} 章继续")

        # 尝试恢复窗口断点
        if not processed_chaps and os.path.exists(self.window_file):
            try:
                with open(self.window_file, "r", encoding="utf-8") as f:
                    win = json.load(f)
                if win.get("stage") == "A_window":
                    offset = win.get("offset", 0)
                    w_data = win["data"]
                    if offset + len(w_data) <= len(chapters) and all(
                        chapters[offset + i]["id"] == w_data[i]["id"]
                        for i in range(len(w_data))
                    ):
                        for i in range(offset):
                            chapters[i].setdefault("summary", "【前文摘要丢失，请仅根据本章内容推断】")
                            chapters[i].setdefault("character_state", {})
                            processed_chaps.append(chapters[i])
                        for i, item in enumerate(w_data):
                            chapters[offset + i]["summary"] = item["summary"]
                            chapters[offset + i]["character_state"] = item["character_state"]
                            processed_chaps.append(chapters[offset + i])
                        finish_count = offset + len(w_data)
                        last_char_state = processed_chaps[-1]["character_state"]
                        recent_texts = [c["text"] for c in processed_chaps[-3:]]
                        inferred_category = win.get("inferred_category", self.category)
                        protagonist_names = set(win.get("protagonist_names", []))
                        print(f"✅ [阶段A] 窗口抢救成功！从第 {finish_count + 1} 章续跑")
            except Exception:
                pass

        # 逐章处理
        remaining_chaps = chapters[finish_count:]
        pbar = tqdm(remaining_chaps, desc="阶段A进度")
        consecutive_fails = 0

        for idx, chap in enumerate(pbar):
            chap_text = chap["text"]

            if consecutive_fails >= 3:
                fallback_state = {
                    name: last_char_state[name]
                    for name in protagonist_names
                    if name in last_char_state
                }
                fallback_state["旁白"] = "前文状态部分丢失，尝试从本章重新推断"
                last_char_state = fallback_state
                consecutive_fails = 0

            compressed_state = compress_character_state(
                last_char_state, recent_texts, protagonist_names
            )
            safe_state_str = compress_state_to_text(compressed_state)

            category_prompt = ""
            if finish_count + idx == 0:
                category_prompt = '\n  "inferred_category": "推断题材(玄幻/都市/悬疑等，限2-4字)",\n  "protagonist_names": ["主角名1", "主角名2"],'

            prompt = f"""你是专业的小说剧情摘要助手。结合前文笔记生成本章摘要与人物状态。仅输出JSON。
【前文人物笔记】{safe_state_str}
【本章正文】{chap_text}
输出JSON：{{
  "chapter_summary": "300-500字剧情摘要，包含主要事件和转折",
  "key_events": [
    {{"event": "关键事件1(50字内)", "impact": "对剧情的影响(30字内)"}},
    {{"event": "关键事件2(50字内)", "impact": "对剧情的影响(30字内)"}}
  ],
  "character_state": {{"角色名": "位置/状态/情绪/目标(如:京城/受伤/愤怒/寻找解药)"}},
  "scene_transitions": {{
    "count": 场景切换次数(数字),
    "methods": ["切换方式1(如:时间跳跃/空间切换/视角转换)", "切换方式2"]
  }},{category_prompt}
}}"""
            try:
                resp = ollama_chat(prompt, 0.1, "A")
                data = safe_parse_json(resp)
                if not data:
                    raise ValueError("解析失败")

                chap["character_state"] = flatten_character_state(
                    data.get("character_state", {})
                )
                chap["summary"] = data.get("chapter_summary", "")
                chap["key_events"] = data.get("key_events", [])
                chap["scene_transitions"] = data.get("scene_transitions", {})
                consecutive_fails = 0

                if finish_count + idx == 0:
                    if data.get("inferred_category"):
                        inferred_category = data["inferred_category"].strip()
                    if isinstance(data.get("protagonist_names"), list):
                        protagonist_names = set(data["protagonist_names"])
            except Exception:
                consecutive_fails += 1
                chap["character_state"] = flatten_character_state({"旁白": "断层"})
                chap["summary"] = "处理失败"
                chap["key_events"] = []
                chap["scene_transitions"] = {}

            last_char_state = chap["character_state"]
            processed_chaps.append(chap)
            recent_texts.append(chap["text"])
            if len(recent_texts) > 3:
                recent_texts.pop(0)

            # 定期保存窗口断点
            if len(processed_chaps) % 10 == 0:
                save_state_atomic(
                    self.window_file,
                    {
                        "stage": "A_window",
                        "offset": len(processed_chaps) - min(50, len(processed_chaps)),
                        "inferred_category": inferred_category,
                        "protagonist_names": list(protagonist_names),
                        "data": [
                            {
                                "id": c["id"],
                                "summary": c["summary"],
                                "character_state": c["character_state"],
                                "key_events": c.get("key_events", []),
                                "scene_transitions": c.get("scene_transitions", {}),
                            }
                            for c in processed_chaps[-50:]
                        ],
                    },
                )

            # 定期保存全量断点
            if len(processed_chaps) % 200 == 0:
                self.save_cache({
                    "stage": "A",
                    "inferred_category": inferred_category,
                    "protagonist_names": list(protagonist_names),
                    "data": [
                        {
                            "id": c["id"],
                            "summary": c["summary"],
                            "character_state": c["character_state"],
                            "key_events": c.get("key_events", []),
                            "scene_transitions": c.get("scene_transitions", {}),
                        }
                        for c in processed_chaps
                    ],
                })

        # 最终保存
        self.save_cache({
            "stage": "A",
            "inferred_category": inferred_category,
            "protagonist_names": list(protagonist_names),
            "data": [
                {
                    "id": c["id"],
                    "summary": c["summary"],
                    "character_state": c["character_state"],
                    "key_events": c.get("key_events", []),
                    "scene_transitions": c.get("scene_transitions", {}),
                }
                for c in processed_chaps
            ],
        })

        return processed_chaps, inferred_category, protagonist_names

    def insert(self, results: Tuple[List[Dict], str, Set[str]]) -> Dict[str, int]:
        """将 Stage A 结果写入数据库"""
        processed_chaps, _, _ = results
        cursor = self.db.connect().cursor()

        existing_ids = self.db.get_existing_ids("plot_arcs", self.book_name, "chapter_id")
        stats = {"plot_arcs": 0, "graph_nodes": 0}

        for chap in tqdm(processed_chaps, desc="入库剧情"):
            if chap["id"] not in existing_ids:
                # 将 key_events 和 scene_transitions 合并到 character_state_json 中
                extended_state = {
                    "character_state": chap.get("character_state", {}),
                    "key_events": chap.get("key_events", []),
                    "scene_transitions": chap.get("scene_transitions", {}),
                }
                cursor.execute(
                    "INSERT OR REPLACE INTO plot_arcs VALUES (?,?,?,?,?)",
                    (
                        chap["id"],
                        self.book_name,
                        self.category,
                        chap.get("summary", ""),
                        json.dumps(extended_state, ensure_ascii=False),
                    ),
                )
                stats["plot_arcs"] += 1

            # 图谱人物节点
            for char_name, char_state in chap.get("character_state", {}).items():
                if char_name in ("_raw", "旁白"):
                    continue
                char_node = f"char:{char_name}"
                if not self.graph.load().has_node(char_node):
                    stats["graph_nodes"] += 1
                self.graph.add_node(char_node, node_type="character", book_list=self.book_name)
                self.graph.safe_append_edge_attr(
                    char_node, f"chap:{chap['id']}", "action", str(char_state)[:50]
                )

        self.db.commit()
        logger.info(f"   ✅ [阶段A战报] plot_arcs 新增: {stats['plot_arcs']} 条 | 图谱人物节点: {stats['graph_nodes']} 个")
        return stats
