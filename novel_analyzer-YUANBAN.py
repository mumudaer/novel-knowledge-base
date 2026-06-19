import os
import sys
import re
import json
import time
import copy
import shutil
import hashlib
import sqlite3
import chromadb
import networkx as nx
import requests
import json_repair
from requests.adapters import HTTPAdapter
from thefuzz import fuzz
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================== 全局硬件专属配置 Win11 16G显存 =====================
STAGE_A_MODEL = "qwen2.5:7b"
STAGE_B_MODEL = "qwen14b:latest"
OLLAMA_API_URL = "http://localhost:11434/api/chat"
CONCURRENCY_LIMIT = 2
MATCH_THRESHOLD = 85
CHROMA_BATCH_SIZE = 100
SPLIT_THRESHOLD = 3500
OLLAMA_NUM_CTX = 4096
OLLAMA_NUM_PREDICT = 2048
OLLAMA_TIMEOUT = 600
SQL_COMMIT_CHUNK = 5000

if os.environ.get("NOVEL_KB_DATA_DIR"):
    BASE_DIR = os.environ["NOVEL_KB_DATA_DIR"]
else:
    if getattr(sys, "frozen", False):
        app_root = os.path.dirname(sys.executable)
    else:
        app_root = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.join(app_root, "novel_kb")

os.makedirs(BASE_DIR, exist_ok=True)
SQLITE_PATH = os.path.join(BASE_DIR, "knowledge.db")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
STATE_FILE = os.path.join(BASE_DIR, "process_state.json")
STATE_WINDOW_FILE = os.path.join(BASE_DIR, "process_state.window.json")
STATE_WINDOW_FALLBACK = STATE_WINDOW_FILE + ".fallback"
STAGE_B_CACHE_FILE = os.path.join(
    BASE_DIR, "stage_b_cache.json"
)  # 【新增】阶段二独立断点文件
UNMATCHED_LOG = os.path.join(BASE_DIR, "unmatched_quotes.jsonl")
STAGE2_FAIL_LOG = os.path.join(BASE_DIR, "stage_b_failures.jsonl")


def safe_str(val, default="未知") -> str:
    if val is None:
        return default
    s = str(val).strip()
    return s if s else default


def flatten_character_state(state: Any) -> Dict[str, str]:
    flat_result = {}
    if not isinstance(state, dict):
        flat_result["_raw"] = str(state)
        return flat_result
    for key, val in state.items():
        k = str(key).strip()
        if isinstance(val, dict):
            flat_result[k] = json.dumps(val, ensure_ascii=False)
        else:
            flat_result[k] = str(val)
    return flat_result


def safe_append_edge_attr(graph, u, v, attr_name: str, attr_value: str):
    if graph.has_edge(u, v):
        old_val = safe_str(graph[u][v].get(attr_name, ""))
        old_list = old_val.split(",") if old_val else []
        if attr_value not in old_list:
            graph[u][v][attr_name] = (
                f"{old_val},{attr_value}" if old_val else attr_value
            )
    else:
        graph.add_edge(u, v, **{attr_name: attr_value})


def sanitize_graph_for_graphml(graph: nx.DiGraph):
    for _, node_attr in graph.nodes(data=True):
        key_list = list(node_attr.keys())
        for k in key_list:
            val = node_attr[k]
            if isinstance(val, (dict, list)):
                node_attr[k] = json.dumps(val, ensure_ascii=False)
    for _, _, edge_attr in graph.edges(data=True):
        key_list = list(edge_attr.keys())
        for k in key_list:
            val = edge_attr[k]
            if isinstance(val, (dict, list)):
                edge_attr[k] = json.dumps(val, ensure_ascii=False)


def find_quote_position_fast(text_scope: str, quote: str) -> int:
    if not quote or not text_scope:
        return -1
    pos = text_scope.find(quote)
    if pos != -1:
        return pos
    split_marks = ("。", "！", "？", "\n", ".", "!", "?", "；")
    positions = [0]
    for idx, char in enumerate(text_scope):
        if char in split_marks:
            positions.append(idx + 1)
    if positions[-1] != len(text_scope):
        positions.append(len(text_scope))
    window_size = 3
    for i in range(len(positions) - 1):
        start = positions[i]
        end_idx = min(i + window_size, len(positions) - 1)
        end = positions[end_idx]
        combined = text_scope[start:end]
        if not combined.strip():
            continue
        if fuzz.WRatio(quote, combined) >= MATCH_THRESHOLD:
            return start
    return -1


JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*\})\s*```", re.DOTALL)


def extract_raw_json(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return text

    brace_count = 0
    in_string = False
    escape = False
    last_brace_idx = -1

    for idx, char in enumerate(text[start:]):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
        if not in_string:
            if char == "{":
                brace_count += 1
                last_brace_idx = idx
            elif char == "}":
                brace_count -= 1
                last_brace_idx = idx
                if brace_count == 0:
                    return text[start : start + idx + 1]
    if last_brace_idx != -1:
        truncated_slice = text[start : start + last_brace_idx + 1]
        fill_brackets = "}" * brace_count
        return truncated_slice + fill_brackets
    return text[start:]


def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    match = JSON_BLOCK_RE.search(text)
    if match:
        json_text = match.group(1)
    else:
        json_text = extract_raw_json(text)
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        pass
    try:
        return json_repair.repair_json(json_text, return_objects=True)
    except Exception:
        return None


def save_state_atomic(filepath: str, data: Dict[str, Any]) -> None:
    dir_abs = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(dir_abs, exist_ok=True)
    temp_path = os.path.join(dir_abs, os.path.basename(filepath) + ".tmp")

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

    max_retry = 20
    for i in range(max_retry):
        try:
            os.replace(temp_path, filepath)
            return
        except PermissionError:
            time.sleep(0.1 * (2**i))
        except OSError:
            shutil.move(temp_path, filepath)
            return

    fallback_path = filepath + ".fallback"
    try:
        shutil.move(temp_path, fallback_path)
        print(
            f"⚠️ 主状态文件被系统锁定，断点数据已安全写入固定备用文件：{fallback_path}"
        )
    except Exception as e:
        print(f"❌ 断点数据写入彻底失败：{str(e)}")


def ollama_chat(prompt: str, temperature: float = 0.2, stage: str = "A") -> str:
    if stage == "A":
        use_model = STAGE_A_MODEL
    else:
        use_model = STAGE_B_MODEL

    payload = {
        "model": use_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": OLLAMA_NUM_PREDICT,
            "temperature": temperature,
        },
    }
    max_retry = 3
    timeout_conn_read = (10, OLLAMA_TIMEOUT)
    for retry_idx in range(max_retry):
        with requests.Session() as local_session:
            adapter = HTTPAdapter(pool_connections=1, pool_maxsize=1)
            local_session.mount("http://", adapter)
            try:
                resp = local_session.post(
                    OLLAMA_API_URL, json=payload, timeout=timeout_conn_read, stream=True
                )
                try:
                    resp.raise_for_status()
                except requests.exceptions.HTTPError as http_err:
                    print("======== Ollama 400 错误调试信息 ========")
                    print(f"服务返回错误内容：{resp.text}")
                    print(f"本次prompt总长度：{len(prompt)} 字符")
                    print("========================================")
                    raise http_err
                raw_bytes = resp.content
                resp_text = raw_bytes.decode("utf-8")
                resp_data = safe_parse_json(resp_text)
                if not resp_data or "message" not in resp_data:
                    raise Exception(f"Ollama响应格式异常，原始返回：{resp_text[:600]}")
                return resp_data["message"]["content"]
            except Exception as err:
                wait_sec = 2**retry_idx
                if retry_idx == max_retry - 1:
                    raise RuntimeError(f"Ollama请求最终失败：{str(err)}") from err
                print(f"请求失败，第{retry_idx+1}次重试，等待{wait_sec}秒")
                time.sleep(wait_sec)


def load_chapters_from_txt(
    txt_path: str, book_name: str, category: str, split_threshold=3500
) -> List[Dict]:
    full_text = ""
    try:
        with open(txt_path, "r", encoding="gbk") as f:
            full_text = f.read()
    except UnicodeDecodeError:
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                full_text = f.read()
        except UnicodeDecodeError:
            with open(txt_path, "rb") as f:
                raw_bytes = f.read()
            full_text = raw_bytes.decode("gbk", errors="ignore")

    chapter_reg = re.compile(
        r"(^(?:"
        r"第[一二三四五六七八九十百千\d]+[章节卷节回]|"
        r"[Cc]hapter\s*\d+|"
        r"\d{1,3}[.．、]?\s*"
        r")[^\n]*\n)",
        re.MULTILINE,
    )
    split_parts = chapter_reg.split(full_text)
    del full_text

    chapter_list = []
    clean_parts = [p for p in split_parts if p.strip() != ""]

    if len(clean_parts) >= 1 and not re.match(
        r"(第[一二三四五六七八九十百千\d]+[章节卷节回]|[Cc]hapter|\d{1,3}[.．])",
        clean_parts[0],
    ):
        intro_text = clean_parts.pop(0).strip()
        if intro_text:
            chapter_list.append(
                {"id": "开篇引子", "text": intro_text, "slice_tag": "full"}
            )

    for i in range(0, len(clean_parts), 2):
        if i + 1 >= len(clean_parts):
            break
        chap_title = clean_parts[i].strip()
        chap_raw_text = clean_parts[i + 1].strip()
        char_len = len(chap_raw_text)
        if char_len <= split_threshold:
            chapter_list.append(
                {"id": chap_title, "text": chap_raw_text, "slice_tag": "full"}
            )
            continue
        print(f"⚠️ 超长章节自动均衡分片：{chap_title} 总字符数 {char_len}")
        mid_pos = len(chap_raw_text) // 2
        search_range_start = max(0, mid_pos - 200)
        search_range_end = min(len(chap_raw_text), mid_pos + 200)
        window_str = chap_raw_text[search_range_start:search_range_end]
        match_all = list(re.finditer(r"[。！？\n]", window_str))
        if match_all:
            best_split_match = min(
                match_all, key=lambda m: abs((search_range_start + m.end()) - mid_pos)
            )
            real_split_index = search_range_start + best_split_match.end()
        else:
            real_split_index = mid_pos
        part_up = chap_raw_text[:real_split_index]
        part_down = chap_raw_text[real_split_index:]
        chapter_list.append(
            {"id": f"{chap_title}_上半段", "text": part_up, "slice_tag": "split"}
        )
        chapter_list.append(
            {"id": f"{chap_title}_下半段", "text": part_down, "slice_tag": "split"}
        )

    del clean_parts
    return chapter_list


def run_stage_a(chapters: List[Dict], book_name: str, category: str) -> List[Dict]:
    print("=== 阶段一：生成章节剧情上下文（支持断点续跑） ===")
    processed_chaps = []
    last_char_state = "{}"
    cache_valid = False
    finish_count = 0

    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                state_cache = json.load(f)
            if state_cache.get("stage") == "A":
                cached_data = state_cache.get("data", [])
                cache_valid = False
                if len(cached_data) <= len(chapters):
                    id_match_all = True
                    for i, cache_item in enumerate(cached_data):
                        item_id = cache_item.get("id")
                        chapter_id = chapters[i].get("id")
                        if (
                            item_id != chapter_id
                            or "summary" not in cache_item
                            or "character_state" not in cache_item
                        ):
                            id_match_all = False
                            break
                    if id_match_all:
                        cache_valid = True
                if cache_valid:
                    for i, cache_item in enumerate(cached_data):
                        chapters[i]["summary"] = cache_item.get(
                            "summary", "摘要缓存缺失"
                        )
                        chapters[i]["character_state"] = cache_item.get(
                            "character_state", {}
                        )
                    finish_count = len(cached_data)
                    processed_chaps = chapters[:finish_count]
                    if finish_count > 0:
                        temp_state = processed_chaps[-1]["character_state"]
                        flat_state = flatten_character_state(temp_state)
                        last_char_state = json.dumps(flat_state, ensure_ascii=False)
                        print(
                            f"✅ 检测到有效轻量断点缓存，从第 {finish_count + 1} 章恢复处理"
                        )
        except Exception:
            print("⚠️ 主断点文件损坏，尝试从滑动窗口备份抢救算力进度...")
            window_path = STATE_WINDOW_FILE
            if not os.path.exists(window_path):
                window_path = STATE_WINDOW_FALLBACK
            if os.path.exists(window_path):
                try:
                    with open(window_path, "r", encoding="utf-8") as fw:
                        window_cache = json.load(fw)
                    if window_cache.get("stage") == "A_window" and window_cache.get(
                        "data"
                    ):
                        offset = window_cache.get("offset", 0)
                        w_data = window_cache["data"]
                        window_len = len(w_data)
                        if offset + window_len <= len(chapters):
                            match_ok = True
                            for idx_in_win, win_item in enumerate(w_data):
                                chap_idx = offset + idx_in_win
                                if chapters[chap_idx]["id"] != win_item["id"]:
                                    match_ok = False
                                    break
                            if match_ok:
                                for i in range(offset):
                                    chapters[i].setdefault(
                                        "summary",
                                        "【断点抢救】前文摘要丢失，基于本章正文独立解析",
                                    )
                                    chapters[i].setdefault(
                                        "character_state",
                                        {"旁白": "前文人物状态已丢失"},
                                    )
                                for idx_in_win, win_item in enumerate(w_data):
                                    chap_idx = offset + idx_in_win
                                    chapters[chap_idx]["summary"] = win_item["summary"]
                                    chapters[chap_idx]["character_state"] = win_item[
                                        "character_state"
                                    ]
                                finish_count = offset + window_len
                                processed_chaps = chapters[:finish_count]
                                temp_state = processed_chaps[-1]["character_state"]
                                flat_state = flatten_character_state(temp_state)
                                last_char_state = json.dumps(
                                    flat_state, ensure_ascii=False
                                )
                                print(
                                    f"✅ 窗口备份抢救成功！从第 {finish_count + 1} 章续跑，挽回 {window_len} 章算力"
                                )
                                cache_valid = True
                except Exception as win_err:
                    print(
                        f"❌ 滑动窗口备份同样损坏：{str(win_err)}，只能从头执行阶段一"
                    )

    if not cache_valid:
        print("⚠️ 无可用完整断点缓存，将从头执行阶段一")

    remaining_chaps = chapters[len(processed_chaps) :]
    total_all = len(chapters)
    done_before = len(processed_chaps)
    pbar = tqdm(remaining_chaps, desc="阶段A剧情摘要生成进度")

    for idx, chap in enumerate(pbar):
        global_finished = done_before + idx + 1
        pbar.set_postfix({"全局进度": f"{global_finished}/{total_all}"}, refresh=False)

        chap_text = chap["text"]
        if len(chap_text) > 1600:
            chap_text = chap_text[:1600] + "\n【章节过长，中间内容截断】"

        safe_last_state = last_char_state
        if len(safe_last_state) > 1200:
            safe_last_state = (
                "...【前文早期边缘人物状态已省略】...\n" + safe_last_state[-1200:]
            )

        prompt_a = f"""
你是网文剧情摘要助手，结合前文人物状态生成本章摘要与更新人物信息，仅输出标准JSON，禁止多余文字。
【前文人物状态】
{safe_last_state}
【本章正文】
{chap_text}
输出JSON格式：
{{
  "chapter_summary": "200字内本章剧情摘要",
  "character_state": {{
    "角色名": "当前身份、状态、人际关系"
  }}
}}
注意：直接输出最外层JSON对象，不要使用markdown代码块，不要嵌套json字段，不要包含任何解释性文字。
"""
        try:
            resp_raw = ollama_chat(prompt_a, temperature=0.1, stage="A")
            parse_data = safe_parse_json(resp_raw)

            if not parse_data:
                raise ValueError("JSON解析为空或格式严重错误")

            raw_char_state = parse_data.get("character_state", {})
            chap["character_state"] = flatten_character_state(raw_char_state)
            chap["summary"] = parse_data.get("chapter_summary", "摘要提取失败")

        except Exception as e:
            print(
                f"\n⚠️ 第 {global_finished} 章({chap['id']}) 请求或解析异常，触发熔断保护: {str(e)}"
            )
            fallback_state = {}
            try:
                fallback_state = json.loads(last_char_state)
            except json.JSONDecodeError:
                pass

            fallback_state["__系统警告__"] = (
                "本章因API超时或格式错误跳过，上下文存在断层"
            )
            chap["character_state"] = flatten_character_state(fallback_state)
            chap["summary"] = (
                f"【系统异常】本章处理失败(原文长度:{len(chap['text'])})，错误信息:{str(e)[:50]}"
            )

        last_char_state = json.dumps(chap["character_state"], ensure_ascii=False)
        processed_chaps.append(chap)

        if len(processed_chaps) % 10 == 0:
            window_slice = [
                {
                    "id": c["id"],
                    "summary": c["summary"],
                    "character_state": c["character_state"],
                }
                for c in processed_chaps[-50:]
            ]
            save_state_atomic(
                STATE_WINDOW_FILE,
                {
                    "stage": "A_window",
                    "offset": len(processed_chaps) - len(window_slice),
                    "data": window_slice,
                },
            )

        if len(processed_chaps) % 100 == 0:
            cache_lite = [
                {
                    "id": c["id"],
                    "summary": c["summary"],
                    "character_state": c["character_state"],
                }
                for c in processed_chaps
            ]
            save_state_atomic(
                STATE_FILE,
                {"stage": "A", "processed": len(processed_chaps), "data": cache_lite},
            )

    pbar.close()

    cache_lite = [
        {
            "id": c["id"],
            "summary": c["summary"],
            "character_state": c["character_state"],
        }
        for c in processed_chaps
    ]
    save_state_atomic(
        STATE_FILE,
        {"stage": "A", "processed": len(processed_chaps), "data": cache_lite},
    )

    window_slice = [
        {
            "id": c["id"],
            "summary": c["summary"],
            "character_state": c["character_state"],
        }
        for c in processed_chaps[-50:]
    ]
    save_state_atomic(
        STATE_WINDOW_FILE,
        {
            "stage": "A_window",
            "offset": len(processed_chaps) - len(window_slice),
            "data": window_slice,
        },
    )

    print(f"阶段一全部完成，共处理 {len(processed_chaps)} 章")
    return processed_chaps


def process_single_chapter(chap: Dict, book_name: str, category: str) -> Dict[str, Any]:
    max_chapter_text = 2500
    raw_text = chap["text"]
    if len(raw_text) > max_chapter_text:
        half = max_chapter_text // 2
        head_part = raw_text[:half]
        tail_part = raw_text[-half:]
        safe_chap_text = f"{head_part}\n……【本章内容过长，中间段落省略】……\n{tail_part}"
    else:
        safe_chap_text = raw_text

    prompt_b = f"""
你是网文专业写作技法分析师，严格基于原文提取可复用写作模板，输出纯JSON，禁止编造不存在的原文内容。
【书籍信息】书名：{book_name}，章节：{chap["id"]}，题材分类：{category}
【前文剧情摘要】{chap["summary"]}
【当前人物状态】{json.dumps(chap["character_state"], ensure_ascii=False)}
【本章完整正文】{safe_chap_text}
固定输出JSON结构严格遵守：
{{
  "scene_type": "场景分类（冲突打脸/升级突破/日常过渡/悬疑铺垫等）",
  "core_function": "本章核心叙事作用",
  "narrative_skills": [
    {{
      "skill_name": "技法名称",
      "original_example": "原文精准对应句子",
      "analysis": "技法底层逻辑",
      "reuse_scenario": "适配写作场景"
    }}
  ],
  "cool_point": {{
    "has_cool_point": true/false,
    "type": "爽点细分类型",
    "quote": "爽点核心原句"
  }},
  "end_hook": "章节结尾钩子，无则填写无",
  "toxic_points": [],
  "style_feature": {{
    "sentence_feature": "句式特点",
    "tone": "整体文风调性"
  }}
}}
强制输出规则：
1. 仅输出外层JSON，不要```json、注释、额外说明文字；
2. toxic_points规则：本章无逻辑硬伤、降智、圣母等负面毒点时，必须输出空数组 []，严禁凭空编造毒点；
3. narrative_skills无对应写法则为空数组；
4. cool_point无爽点has_cool_point设为false，quote填空字符串。
"""
    resp_raw = ollama_chat(prompt_b, temperature=0.2, stage="B")
    result = safe_parse_json(resp_raw)
    if not result:
        raise Exception("章节技法JSON解析完全失败")
    result.setdefault("narrative_skills", [])
    result.setdefault("scene_type", "未知场景")
    result.setdefault("cool_point", {"has_cool_point": False, "type": "", "quote": ""})
    result.setdefault("toxic_points", [])
    result.setdefault("style_feature", {"sentence_feature": "无", "tone": "无"})
    cool_data = result.get("cool_point", {})
    unmatched_info = None
    if cool_data.get("has_cool_point"):
        quote_text = cool_data.get("quote", "")
        pos = find_quote_position_fast(chap["text"], quote_text)
        result["cool_point"]["char_pos"] = pos
        if pos == -1 and quote_text.strip():
            unmatched_info = {"chapter": chap["id"], "unmatched_quote": quote_text}
    result["_unmatched_log"] = unmatched_info
    result["chapter_id"] = chap["id"]
    result["book_name"] = book_name
    result["category"] = category
    return result


def run_stage_b(
    context_chapters: List[Dict], book_name: str, category: str
) -> List[Dict]:
    print("=== 阶段二：多线程并发提取写作技法 ===")
    success_list = []
    fail_list = []
    log_buffer = []

    # 【新增】阶段二断点缓存加载逻辑
    cached_chapter_ids = set()
    if os.path.exists(STAGE_B_CACHE_FILE):
        try:
            with open(STAGE_B_CACHE_FILE, "r", encoding="utf-8") as f:
                b_cache = json.load(f)
            cached_data = b_cache.get("data", [])
            if cached_data:
                success_list.extend(cached_data)
                cached_chapter_ids = {item["chapter_id"] for item in cached_data}
                print(f"✅ 检测到阶段二断点缓存，已恢复 {len(cached_data)} 章算力成果")
        except Exception as e:
            print(f"⚠️ 阶段二缓存读取失败，将重新计算：{str(e)}")

    # 过滤掉已经缓存的章节，只提交未处理的章节给线程池
    pending_chapters = [
        c for c in context_chapters if c["id"] not in cached_chapter_ids
    ]

    if not pending_chapters and success_list:
        print(f"✅ 阶段二已全部命中缓存，跳过 LLM 请求，直接进入入库流程")
        return success_list

    # ✅ 正确做法：根据硬件设定合理并发，避免OOM和无效排队
    # 8G显存/16G内存建议设为 2-3；24G显存建议设为 4-6
    STAGE_B_WORKERS = int(os.getenv("STAGE_B_WORKERS", 3))

    with ThreadPoolExecutor(max_workers=STAGE_B_WORKERS) as executor:
        futures = []
        for chap in pending_chapters:
            fut = pool.submit(process_single_chapter, chap, book_name, category)
            fut._chap_id = chap["id"]
            futures.append(fut)

        for task in tqdm(
            as_completed(futures), total=len(futures), desc="阶段二提取进度"
        ):
            try:
                task_res = task.result()
                success_list.append(task_res)
                if task_res.get("_unmatched_log"):
                    log_buffer.append(task_res["_unmatched_log"])

                # 【新增】每成功处理10章，实时持久化阶段二结果
                if len(success_list) % 10 == 0:
                    save_state_atomic(STAGE_B_CACHE_FILE, {"data": success_list})

            except Exception as err:
                err_chap = getattr(task, "_chap_id", "未知章节")
                fail_list.append((err_chap, str(err)))
                print(f"❌ 章节 {err_chap} 处理失败：{str(err)}")

    # 【新增】循环结束后最终保存一次完整缓存
    if success_list:
        save_state_atomic(STAGE_B_CACHE_FILE, {"data": success_list})

    if log_buffer:
        try:
            with open(UNMATCHED_LOG, "a", encoding="utf-8") as f:
                for item in log_buffer:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        except PermissionError:
            fallback_path = os.path.join(
                BASE_DIR, f"unmatched_quotes_{int(time.time())}.jsonl"
            )
            with open(fallback_path, "a", encoding="utf-8") as f:
                for item in log_buffer:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"⚠️ 主爽点日志被占用，已写入备用日志：{fallback_path}")

    if fail_list:
        try:
            with open(STAGE2_FAIL_LOG, "w", encoding="utf-8") as f:
                for chap_id, err_msg in fail_list:
                    record = {
                        "chapter_id": chap_id,
                        "error_detail": err_msg,
                        "timestamp": int(time.time()),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"⚠️ 共{len(fail_list)}章节处理失败，错误清单已保存至：{STAGE2_FAIL_LOG}，可单独补跑"
            )
        except Exception as log_err:
            print(
                f"⚠️ 失败日志写入失败，内存失败清单：{fail_list}，错误：{str(log_err)}"
            )

    print(f"阶段二执行完毕：成功 {len(success_list)} 章，失败 {len(fail_list)}")
    return success_list


def init_database_resource():
    conn = sqlite3.connect(SQLITE_PATH, timeout=30.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA cache_size = -64000;")
    conn.execute("PRAGMA synchronous = NORMAL;")

    conn.execute("""
    CREATE TABLE IF NOT EXISTS skills (
        id TEXT PRIMARY KEY,
        book_name TEXT,
        chapter_id TEXT,
        category TEXT,
        scene_type TEXT,
        skill_name TEXT,
        analysis TEXT,
        original_example TEXT,
        tags TEXT
    )
    """)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_tags ON skills(tags);")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_book_chapter ON skills(book_name, chapter_id);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_scene_skill ON skills(scene_type, skill_name);"
    )

    conn.commit()
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    skill_collection = chroma_client.get_or_create_collection(name="novel_skills")
    graph_save_path = os.path.join(BASE_DIR, "knowledge_graph.graphml")
    graph = nx.DiGraph()
    if os.path.exists(graph_save_path):
        try:
            graph = nx.read_graphml(graph_save_path)
        except Exception as xml_err:
            bak_path = graph_save_path + f"_corrupted_{int(time.time())}.bak"
            try:
                shutil.move(graph_save_path, bak_path)
                print(
                    f"⚠️ 图谱文件损坏({str(xml_err)})，已备份至 {bak_path}，自动重建空图谱继续启动"
                )
            except Exception:
                os.remove(graph_save_path)
                print(f"⚠️ 图谱文件损坏且备份失败，强制删除损坏文件并重建空图谱")
    return conn, skill_collection, graph, graph_save_path


def insert_knowledge(extract_results: List[Dict]):
    if not extract_results:
        print(
            "⚠️ 阶段二技法提取结果为空（Ollama服务宕机/模型全部OOM），触发数据熔断保护，跳过入库与图谱写入，防止历史图谱被清空！"
        )
        return

    conn, skill_collection, graph, graph_path = init_database_resource()
    db_cursor = conn.cursor()
    chapter_merge_map = defaultdict(list)
    for item in extract_results:
        pure_chap_id = item["chapter_id"].replace("_上半段", "").replace("_下半段", "")
        item["chapter_id"] = pure_chap_id
        chapter_merge_map[pure_chap_id].append(item)
    unified_results = []
    for chap_name, slice_list in chapter_merge_map.items():
        base_data = copy.deepcopy(slice_list[0])
        all_skills = []
        all_scene_tags = set()
        for slice_data in slice_list:
            if slice_data.get("scene_type"):
                all_scene_tags.add(slice_data["scene_type"])
            skill_list = slice_data.get("narrative_skills", [])
            all_skills.extend(skill_list)
        base_data["narrative_skills"] = all_skills
        base_data["scene_type"] = (
            "/".join(list(all_scene_tags)) if all_scene_tags else "未知场景"
        )
        unified_results.append(base_data)

    batch_ids = []
    batch_docs = []
    batch_metas = []
    db_exec_count = 0
    commit_chunk = SQL_COMMIT_CHUNK

    try:
        for item in tqdm(unified_results, desc="全库入库进度"):
            book = item["book_name"]
            chap_id = item["chapter_id"]
            category = item["category"]
            scene_type = item["scene_type"]
            skill_list = item["narrative_skills"]
            for skill in skill_list:
                raw_id_str = f"{book}|{chap_id}|{skill['skill_name']}|{skill['original_example']}"
                unique_sid = hashlib.md5(raw_id_str.encode("utf-8")).hexdigest()
                tag_text = (
                    f"|场景:{scene_type}|题材:{category}|技法:{skill['skill_name']}|"
                )
                db_cursor.execute(
                    """
                INSERT OR IGNORE INTO skills 
                (id, book_name, chapter_id, category, scene_type, skill_name, analysis, original_example, tags)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                    (
                        unique_sid,
                        book,
                        chap_id,
                        category,
                        scene_type,
                        skill["skill_name"],
                        skill["analysis"],
                        skill["original_example"],
                        tag_text,
                    ),
                )
                db_exec_count += 1

                if db_exec_count >= commit_chunk:
                    conn.commit()
                    try:
                        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                    except sqlite3.OperationalError:
                        conn.execute("PRAGMA wal_checkpoint(PASSIVE);")
                    db_exec_count = 0
                    try:
                        sanitize_graph_for_graphml(graph)
                        nx.write_graphml(graph, graph_path)
                    except Exception as graph_err:
                        print(
                            f"⚠️ 图谱增量快照落盘失败，入库收尾重试：{str(graph_err)}"
                        )

                vec_meta_info = {
                    "book_name": safe_str(book),
                    "chapter": safe_str(chap_id),
                    "category": safe_str(category),
                    "scene_type": safe_str(scene_type),
                    "skill_name": safe_str(skill.get("skill_name")),
                }
                vec_doc = f"""
技法名称：{skill['skill_name']}
解析逻辑：{skill['analysis']}
原文示例：{skill['original_example']}
复用场景：{skill['reuse_scenario']}
                """.strip()
                batch_ids.append(unique_sid)
                batch_docs.append(vec_doc)
                batch_metas.append(vec_meta_info)

                scene_node = f"scene:{safe_str(scene_type)}"
                skill_node = f"skill:{safe_str(skill['skill_name'])}:{safe_str(category)}:{safe_str(book)}"
                cate_node = f"category:{safe_str(category)}"

                graph.add_node(
                    scene_node, node_type="scene", raw_name=safe_str(scene_type)
                )

                # NetworkX 3.x AtlasView 兼容处理
                if graph.has_node(skill_node):
                    old_cate = safe_str(
                        graph.nodes[skill_node].get("category_list", "")
                    )
                    if category not in old_cate:
                        graph.nodes[skill_node]["category_list"] = (
                            f"{old_cate},{category}" if old_cate else category
                        )
                    old_books = safe_str(graph.nodes[skill_node].get("book_list", ""))
                    if book not in old_books:
                        graph.nodes[skill_node]["book_list"] = (
                            f"{old_books},{book}".strip(",")
                        )
                else:
                    graph.add_node(
                        skill_node,
                        node_type="skill",
                        raw_name=safe_str(skill["skill_name"]),
                        category_list=category,
                        book_list=book,
                    )

                graph.add_node(
                    cate_node, node_type="category", raw_name=safe_str(category)
                )
                safe_append_edge_attr(
                    graph, scene_node, skill_node, "relation", "包含写作技法"
                )
                safe_append_edge_attr(
                    graph, skill_node, cate_node, "relation", "归属题材分类"
                )

                if len(batch_ids) >= CHROMA_BATCH_SIZE:
                    seen = set()
                    uniq_ids, uniq_docs, uniq_metas = [], [], []
                    for idx, sid in enumerate(batch_ids):
                        if sid not in seen:
                            seen.add(sid)
                            uniq_ids.append(sid)
                            uniq_docs.append(batch_docs[idx])
                            uniq_metas.append(batch_metas[idx])
                    try:
                        skill_collection.upsert(
                            ids=uniq_ids, documents=uniq_docs, metadatas=uniq_metas
                        )
                    except Exception as batch_err:
                        print(
                            f"⚠️ Chroma批量写入失败，降级单条隔离脏数据: {str(batch_err)}"
                        )
                        for i in range(len(uniq_ids)):
                            try:
                                skill_collection.upsert(
                                    ids=[uniq_ids[i]],
                                    documents=[uniq_docs[i]],
                                    metadatas=[uniq_metas[i]],
                                )
                            except Exception as single_err:
                                print(
                                    f"❌ 脏数据跳过 ID:{uniq_ids[i]} 错误:{str(single_err)}"
                                )
                    batch_ids, batch_docs, batch_metas = [], [], []

        if db_exec_count > 0:
            conn.commit()
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            except sqlite3.OperationalError:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE);")
            try:
                sanitize_graph_for_graphml(graph)
                nx.write_graphml(graph, graph_path)
            except Exception as graph_err:
                print(f"⚠️ 入库收尾图谱落盘失败：{str(graph_err)}")

        if batch_ids:
            seen = set()
            uniq_ids, uniq_docs, uniq_metas = [], [], []
            for idx, sid in enumerate(batch_ids):
                if sid not in seen:
                    seen.add(sid)
                    uniq_ids.append(sid)
                    uniq_docs.append(batch_docs[idx])
                    uniq_metas.append(batch_metas[idx])
            try:
                skill_collection.upsert(
                    ids=uniq_ids, documents=uniq_docs, metadatas=uniq_metas
                )
            except Exception as batch_err:
                print(f"⚠️ Chroma尾批写入失败，降级单条：{str(batch_err)}")
                for i in range(len(uniq_ids)):
                    try:
                        skill_collection.upsert(
                            ids=[uniq_ids[i]],
                            documents=[uniq_docs[i]],
                            metadatas=[uniq_metas[i]],
                        )
                    except Exception as single_err:
                        print(f"❌ 脏数据跳过 ID:{uniq_ids[i]} 错误:{str(single_err)}")

        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")

    except Exception as err:
        conn.rollback()
        raise err
    finally:
        if "db_cursor" in locals() and db_cursor:
            try:
                db_cursor.close()
            except Exception:
                pass
        if "conn" in locals() and conn:
            try:
                conn.rollback()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass

    sanitize_graph_for_graphml(graph)
    nx.write_graphml(graph, graph_path)
    print(
        "✅ 入库完成：空数据熔断保护、SQLite分批TRUNCATE WAL、图谱损坏自动备份重建、移除危险cursor.reset规避底层段错误、向量批次自动去重、图谱增量快照、数据库资源安全释放、磁盘日志持续受控不膨胀、极端Ollama宕机不会清空存量图谱"
    )


def main():
    # ========== 修改小说配置 ==========
    NOVEL_TEXT_PATH = "./novels/jingdiansuojiao.txt"
    NOVEL_BOOK_NAME = "靓女生猛"
    NOVEL_CATEGORY = "言情"
    # =================================
    chapter_list = load_chapters_from_txt(
        NOVEL_TEXT_PATH, NOVEL_BOOK_NAME, NOVEL_CATEGORY, SPLIT_THRESHOLD
    )
    print(f"小说加载完成，总章节（含分片）：{len(chapter_list)}")
    stage_a_result = run_stage_a(chapter_list, NOVEL_BOOK_NAME, NOVEL_CATEGORY)
    stage_b_result = run_stage_b(stage_a_result, NOVEL_BOOK_NAME, NOVEL_CATEGORY)
    insert_knowledge(stage_b_result)
    print("🎉 小说全流程拆解、知识库构建全部执行完成！")


if __name__ == "__main__":
    main()
