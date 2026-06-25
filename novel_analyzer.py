import os
import sys
import re
import json
import time
import copy
import shutil
import hashlib
import sqlite3
import glob
import gc
import random
import threading
import traceback
import chromadb
import networkx as nx
import requests
import json_repair
import logging
from requests.adapters import HTTPAdapter
from thefuzz import fuzz
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================== 全局硬件专属配置 Win11 16G显存 + 16G内存 =====================
STAGE_A_MODEL = "qwen2.5:3b"
STAGE_BC_MODEL = "qwen14b:latest"
OLLAMA_API_URL = "http://localhost:11434/api/chat"
OLLAMA_BASE_URL = "http://localhost:11434"

# 3. 【保命修改】并发数死死锁在 1 或 2！
# 你的物理内存只有 16G，如果设为 2，Ollama 会同时处理 2 个 8K 上下文，内存必爆！
# 建议设为 1（最稳，绝对不会用虚拟内存，速度反而最快），如果不怕偶尔卡顿可以设为 2。
STAGE_BC_WORKERS = int(os.getenv("STAGE_BC_WORKERS", 2))

MATCH_THRESHOLD = 85
# 【硬件优化2】ChromaDB 批量写入从 100 降到 50，大幅降低内存峰值，防止16G内存爆掉
CHROMA_BATCH_SIZE = 50
SPLIT_THRESHOLD = 3500
# 14B模型跑8K上下文大约占用 10.5G 显存，你的 16G 显卡完全吃得消，且能完美吞下 3500 字的切块+Prompt！
OLLAMA_NUM_CTX = 8192
OLLAMA_NUM_PREDICT = 2048
OLLAMA_TIMEOUT = 600
SQL_COMMIT_CHUNK = 5000

HTTP_ADAPTER = HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=3)
RETRY_LOCK = threading.Lock()

# 全局共享 Session
GLOBAL_SESSION = requests.Session()
GLOBAL_SESSION.mount("http://", HTTP_ADAPTER)
GLOBAL_SESSION.mount("https://", HTTP_ADAPTER)

# 【硬件优化4】强制设置 Ollama 环境变量，防止用户忘记设置
if os.environ.get("OLLAMA_NUM_PARALLEL") is None:
    os.environ["OLLAMA_NUM_PARALLEL"] = "2"
    print("\n" + "=" * 50)
    print("🚀 已开启 OLLAMA_NUM_PARALLEL = 2，压榨 16G 显存双并发性能！")
    print("=" * 50 + "\n")

# 限制 PyTorch/Ollama 抢占过多 CPU 线程，给系统留点余量
os.environ["OMP_NUM_THREADS"] = "8"

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
MANIFEST_FILE = os.path.join(BASE_DIR, "process_manifest.json")
UNMATCHED_LOG = os.path.join(BASE_DIR, "unmatched_quotes.jsonl")


def get_state_file(book_name: str, stage: str = "A") -> str:
    safe_name = re.sub(r'[\\/*?:"<>|]', "", book_name)
    return os.path.join(BASE_DIR, f"state_{stage}_{safe_name}.json")


def get_window_file(book_name: str) -> str:
    safe_name = re.sub(r'[\\/*?:"<>|]', "", book_name)
    return os.path.join(BASE_DIR, f"state_A_window_{safe_name}.json")


JSON_BLOCK_RE = re.compile(r"`{3}json\s*(\{.*\})\s*`{3}", re.DOTALL)


# ===================== Ollama 健康检查 =====================
def check_ollama_health():
    print("🔍 正在检查 Ollama 服务状态...")
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        resp.raise_for_status()
        available_models = [m["name"] for m in resp.json().get("models", [])]
        print(f"✅ Ollama 服务正常运行，已安装模型：{available_models}")
    except requests.exceptions.ConnectionError:
        print("\n" + "!" * 60)
        print("❌ 致命错误：无法连接到 Ollama 服务！")
        print("   请在命令行运行：ollama serve")
        print("!" * 60)
        sys.exit(1)
    except Exception as e:
        print(f"⚠️ Ollama 服务状态检查失败：{e}")
        sys.exit(1)

    required_models = {STAGE_A_MODEL, STAGE_BC_MODEL}
    for model in required_models:
        model_base = model.split(":")[0]
        found = any(model_base in m for m in available_models)
        if not found:
            print(f"\n❌ 模型 [{model}] 未安装！请运行：ollama pull {model}")
            sys.exit(1)
        else:
            print(f"✅ 模型 [{model}] 已就绪")
    print("✅ Ollama 环境检查通过！\n")


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
        if isinstance(val, (dict, list)):
            flat_result[k] = json.dumps(val, ensure_ascii=False)
        else:
            flat_result[k] = str(val)
    return flat_result


def compress_state_to_text(state_dict: dict) -> str:
    if not state_dict or (len(state_dict) == 1 and "_raw" in state_dict):
        return "暂无明确人物状态"
    text_parts = []
    for name, state in state_dict.items():
        if name == "_raw":
            continue
        state_str = ""
        if isinstance(state, dict):
            state_str = "/".join([str(v) for v in state.values() if v])
        elif isinstance(state, str) and state.strip().startswith("{"):
            try:
                parsed = json.loads(state)
                if isinstance(parsed, dict):
                    state_str = "/".join([str(v) for v in parsed.values() if v])
                elif isinstance(parsed, list):
                    state_str = "/".join([str(v) for v in parsed if v])
                else:
                    state_str = str(state)
            except Exception:
                state_str = str(state)
        elif isinstance(state, str) and state.strip().startswith("["):
            try:
                parsed = json.loads(state)
                state_str = (
                    "/".join([str(v) for v in parsed if v])
                    if isinstance(parsed, list)
                    else str(state)
                )
            except Exception:
                state_str = str(state)
        else:
            state_str = str(state)
        if state_str:
            text_parts.append(f"{name}:{state_str}")
    return "; ".join(text_parts) if text_parts else "暂无明确人物状态"


def compress_character_state(
    current_state: Dict[str, str], recent_texts: List[str], protagonist_names: Set[str]
) -> Dict[str, str]:
    recent_names = set()
    for text in recent_texts:
        possible_names = set(re.findall(r"[\u4e00-\u9fa5]{2,4}", text))
        recent_names.update(possible_names)
    compressed = {}
    for name, state in current_state.items():
        if name in protagonist_names or name in recent_names or name == "_raw":
            compressed[name] = state
    if not compressed:
        compressed["旁白"] = "当前无核心人物出场"
    return compressed


def safe_append_edge_attr(graph, u, v, attr_name: str, attr_value: str):
    if not attr_value:
        return
    if graph.has_edge(u, v):
        old_val = safe_str(graph[u][v].get(attr_name, ""))
        old_list = (
            [x.strip() for x in old_val.split(",") if x.strip()] if old_val else []
        )
        if attr_value not in old_list:
            old_list.append(attr_value)
        graph[u][v][attr_name] = ",".join(old_list)
    else:
        graph.add_edge(u, v, **{attr_name: attr_value})


def sanitize_graph_for_graphml(graph: nx.DiGraph):
    def clean_val(v):
        if v is None:
            return ""
        if isinstance(v, (dict, list)):
            return json.dumps(v, ensure_ascii=False)
        return str(v)

    for _, node_attr in graph.nodes(data=True):
        for k in list(node_attr.keys()):
            node_attr[k] = clean_val(node_attr[k])
    for _, _, edge_attr in graph.edges(data=True):
        for k in list(edge_attr.keys()):
            edge_attr[k] = clean_val(edge_attr[k])


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
    for i in range(len(positions) - 1):
        start = positions[i]
        end = positions[min(i + 8, len(positions) - 1)]
        combined = text_scope[start:end]
        if combined.strip() and fuzz.WRatio(quote, combined) >= MATCH_THRESHOLD:
            return start
    return -1


def extract_raw_json(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return text
    brace_count, in_string, escape, last_brace_idx = 0, False, False, -1
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
        return text[start : start + last_brace_idx + 1] + "}" * brace_count
    return text[start:]


def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    match = JSON_BLOCK_RE.search(text)
    json_text = match.group(1) if match else extract_raw_json(text)
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        pass
    try:
        return json_repair.repair_json(json_text, return_objects=True)
    except Exception:
        return None


def save_state_atomic(filepath: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    temp_path = filepath + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    for i in range(20):
        try:
            os.replace(temp_path, filepath)
            return
        except PermissionError:
            time.sleep(min(0.1 * (2**i), 30))
        except OSError:
            shutil.move(temp_path, filepath)
            return
    shutil.move(temp_path, filepath + ".fallback")


def load_manifest() -> Dict:
    if os.path.exists(MANIFEST_FILE):
        try:
            with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"completed_books": [], "current_processing": None}


def save_manifest(data: Dict):
    save_state_atomic(MANIFEST_FILE, data)


def ollama_chat(prompt: str, temperature: float = 0.2, stage: str = "A") -> str:
    use_model = STAGE_A_MODEL if stage == "A" else STAGE_BC_MODEL
    payload = {
        "model": use_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": OLLAMA_NUM_PREDICT,
            "temperature": temperature,
            # 【硬件优化5】强制模型留在显存中，如果显存不够直接报错，防止偷偷用CPU导致卡死
            "num_gpu": 99,
        },
    }
    for retry_idx in range(3):
        try:
            with GLOBAL_SESSION.post(
                OLLAMA_API_URL,
                json=payload,
                stream=True,
                timeout=(30, OLLAMA_TIMEOUT),
            ) as resp:
                resp.raise_for_status()
                full_content = []
                for line in resp.iter_lines():
                    if line:
                        try:
                            json_resp = json.loads(line.decode("utf-8"))
                            if (
                                "message" in json_resp
                                and "content" in json_resp["message"]
                            ):
                                full_content.append(json_resp["message"]["content"])
                        except json.JSONDecodeError:
                            continue
                return "".join(full_content)
        except Exception as err:
            if retry_idx == 2:
                raise RuntimeError(f"Ollama请求最终失败：{str(err)}") from err
            with RETRY_LOCK:
                sleep_time = (2**retry_idx) + random.uniform(0, 1)
                print(
                    f"\n⚠️ Ollama 请求异常，{sleep_time:.1f}秒后重试... [{type(err).__name__}: {str(err)[:120]}]"
                )
                time.sleep(sleep_time)


# ===================== 全局日志系统配置 =====================
# 1. 主日志：双通道输出（同时打印到控制台 + 写入 pipeline_run.log 文件）
main_log_path = os.path.join(BASE_DIR, "pipeline_run.log")

# 清除之前可能存在的默认 handler，防止日志重复打印
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(
            main_log_path, encoding="utf-8", mode="a"
        ),  # 追加写入主日志文件
        logging.StreamHandler(sys.stdout),  # 同步输出到控制台
    ],
)
# 🌟 定义全局主 logger，这就是 insert_knowledge 里用的那个！
logger = logging.getLogger("NovelPipeline")

# 2. 切分错误专用日志（独立文件，只记录警告，不干扰主日志）
split_error_log_path = os.path.join(BASE_DIR, "split_error.log")
split_logger = logging.getLogger("SplitError")
split_logger.setLevel(logging.WARNING)
split_logger.propagate = False  # 阻止它向上传递，防止在控制台重复打印
split_logger.addHandler(
    logging.FileHandler(split_error_log_path, encoding="utf-8", mode="a")
)


def clean_novel_text(text: str) -> str:
    """
    🧹 网文专属文本清洗引擎：剔除杂质，提纯正文，保护大模型注意力
    """
    # 1. 统一换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. 去除防盗章节特征（大量连续的生僻字、无意义符号、乱码）
    # 匹配连续 10 个以上的非中文字符、非英文、非数字、非常见标点符号的行
    text = re.sub(
        r"^[^\u4e00-\u9fa5a-zA-Z0-9\s\.,;:!?，。；：！？、\n]{10,}$",
        "",
        text,
        flags=re.MULTILINE,
    )

    # 3. 剔除“求月票/求订阅/作者的话”等单行废话
    # 匹配以这些词开头或包含这些词的短行（通常作者的话都是独立成段的）
    noise_patterns = [
        r"^[\s ]*(求月票|求订阅|求推荐|求收藏|求打赏|拜求|感谢.*?打赏|感谢.*?万赏).*?$",
        r"^[\s ]*(PS|ps|Ps|pS)[：:].*?$",
        r"^[\s ]*(作者的话|作者说|题外话|碎碎念)[：:].*?$",
        r"^[\s ]*(本章未完|点击下一页继续阅读|最新网址|手机阅读).*?$",
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)

    # 4. 剔除“作者的话”块状区域（通常是 chapter 末尾的一大段）
    # 匹配“作者的话”或“ps”直到下一个章节标题或文本结尾
    text = re.sub(
        r"(?:作者的话|PS|ps)[：:\s]*\n[\s\S]*?(?=(?:第[零一二三四五六七八九十百千万两\d]+[章节回])|$)",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # 5. 清理多余空行（将 3 个以上的连续换行符压缩为 2 个）
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 6. 清理行首行尾的空白字符
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


def smart_split_chapters(text, book_name="未知书籍", max_chunk=3500):
    """
    平滑记录版章节切分引擎：
    1. 精准识别，大章分块，无章硬切。
    2. 遇到丢失率超 5% 的情况，不中断程序，而是写入日志放行。
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    original_length = len(text.replace("\n", "").replace(" ", ""))

    # 1. 强特征正则：只匹配明确的章节标题
    strong_pattern = r"(?:^|\n+)\s*((?:第[零一二三四五六七八九十百千万两\d\-]+[章节回卷集部篇].{0,30})|(?:[Cc]hapter\s*\d+.{0,30}))\s*(?:\n+|$)"

    parts = re.split(strong_pattern, text)

    chapters = []
    if parts[0].strip():
        chapters.append({"title": "序言/前言", "content": parts[0].strip()})

    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if title:
            chapters.append({"title": title, "content": content})

    # 2. 没章硬切
    if len(chapters) <= 1 and len(text) > max_chunk:
        chapters = [
            {"title": f"自动分块_{idx+1}", "content": text[i : i + max_chunk]}
            for idx, i in enumerate(range(0, len(text), max_chunk))
        ]
    elif not chapters:
        chapters = [{"title": "全篇", "content": text}]

    # 3. 大章切碎
    final_chapters = []
    for ch in chapters:
        content_len = len(ch["content"])
        if content_len > max_chunk:
            num_chunks = (content_len + max_chunk - 1) // max_chunk
            for idx in range(num_chunks):
                start_idx = idx * max_chunk
                end_idx = min(start_idx + max_chunk, content_len)

                if "分块" in ch["title"] or ch["title"] in ["序言/前言", "全篇"]:
                    new_title = f"{ch['title']}_{idx+1}"
                else:
                    new_title = f"{ch['title']}_分块{idx+1}"

                final_chapters.append(
                    {"title": new_title, "content": ch["content"][start_idx:end_idx]}
                )
        else:
            final_chapters.append(ch)

    # 4. 📝 平滑日志记录：字数守恒检查（不阻断程序）
    split_text_combined = "".join(
        [ch["title"] + ch["content"] for ch in final_chapters]
    )
    split_length = len(split_text_combined.replace("\n", "").replace(" ", ""))

    loss_rate = (
        (original_length - split_length) / original_length if original_length > 0 else 0
    )

    if loss_rate > 0.05:
        # 超过 5% 丢失率，记录到日志文件，并在控制台打印黄色警告，但【不报错、不停止】
        warn_msg = f"⚠️ 文本丢失警告: 《{book_name}》 丢失了 {loss_rate:.2%} (原始: {original_length}字, 切分后: {split_length}字)"
        split_logger.warning(warn_msg)
        print(f"\033[93m{warn_msg} (已记入 split_error.log)\033[0m")  # 控制台显示黄色
    else:
        print(
            f"✅ 切分校验通过：《{book_name}》 共 {len(final_chapters)} 块，完整度 {1 - loss_rate:.2%}"
        )

    return final_chapters


def load_chapters_from_txt(
    txt_path: str, book_name: str, category: str, split_threshold=3500
) -> List[Dict]:
    file_size_mb = os.path.getsize(txt_path) / (1024 * 1024)
    if file_size_mb > 50:
        print(f"⚠️ 警告：文件 {txt_path} 大小 {file_size_mb:.1f}MB，占用较多内存。")

    full_text = ""
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            full_text = f.read()
    except UnicodeDecodeError:
        try:
            with open(txt_path, "r", encoding="gbk") as f:
                full_text = f.read()
        except UnicodeDecodeError:
            try:
                with open(txt_path, "r", encoding="utf-16") as f:
                    full_text = f.read()
            except UnicodeDecodeError:
                with open(txt_path, "rb") as f:
                    full_text = f.read().decode("latin-1", errors="ignore")

    # 调用万能切分引擎
    # raw_chapters = smart_split_chapters(full_text, book_name, max_chunk=split_threshold)

    # 🌟 修改为：先清洗，再切分！
    pure_text = clean_novel_text(full_text)

    # 如果清洗后文本太短（说明可能是全篇防盗乱码），给个警告
    if len(pure_text) < 500:
        print(f"⚠️ 警告：《{book_name}》 清洗后正文不足500字，可能是防盗章节或空文件！")

    raw_chapters = smart_split_chapters(pure_text, book_name, max_chunk=split_threshold)

    del full_text
    try:
        gc.collect()
    except Exception:
        pass

    # 转换为程序后续需要的格式
    chapter_list = []
    for ch in raw_chapters:
        chapter_list.append(
            {"id": ch["title"], "text": ch["content"], "slice_tag": "full"}
        )
    return chapter_list


# ===================== 阶段 A：剧情与状态 =====================
def run_stage_a(
    chapters: List[Dict], book_name: str, current_category: str
) -> Tuple[List[Dict], str, Set[str]]:
    print("=== 阶段一：生成剧情上下文与智能推断分类 ===")
    STATE_FILE = get_state_file(book_name, "A")
    WINDOW_FILE = get_window_file(book_name)

    processed_chaps, last_char_state, finish_count, inferred_category = (
        [],
        {},
        0,
        current_category,
    )
    protagonist_names = set()
    recent_texts = []
    cache_valid = False
    consecutive_fails = 0

    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                state_cache = json.load(f)
            if state_cache.get("stage") == "A" and len(
                state_cache.get("data", [])
            ) <= len(chapters):
                cached_data = state_cache["data"]
                if all(
                    cached_data[i].get("id") == chapters[i].get("id")
                    for i in range(len(cached_data))
                ):
                    cache_valid = True
                    for i, item in enumerate(cached_data):
                        chapters[i]["summary"] = item.get("summary", "")
                        chapters[i]["character_state"] = item.get("character_state", {})
                        processed_chaps.append(chapters[i])
                    finish_count = len(cached_data)
                    last_char_state = processed_chaps[-1]["character_state"]
                    inferred_category = state_cache.get(
                        "inferred_category", current_category
                    )
                    protagonist_names = set(state_cache.get("protagonist_names", []))
                    recent_texts = [c["text"] for c in processed_chaps[-3:]]
                    print(f"✅ [阶段A] 恢复全量断点：从第 {finish_count + 1} 章继续")
        except Exception:
            pass

    if not cache_valid and os.path.exists(WINDOW_FILE):
        try:
            with open(WINDOW_FILE, "r", encoding="utf-8") as f:
                win = json.load(f)
            if win.get("stage") == "A_window":
                offset, w_data = win.get("offset", 0), win["data"]
                if offset + len(w_data) <= len(chapters) and all(
                    chapters[offset + i]["id"] == w_data[i]["id"]
                    for i in range(len(w_data))
                ):
                    for i in range(offset):
                        emergency_summary = "【前文摘要丢失，请仅根据本章内容推断】"
                        chapters[i].setdefault("summary", emergency_summary)
                        chapters[i].setdefault("character_state", {})
                        processed_chaps.append(chapters[i])
                    for i, item in enumerate(w_data):
                        chapters[offset + i]["summary"] = item["summary"]
                        chapters[offset + i]["character_state"] = item[
                            "character_state"
                        ]
                        processed_chaps.append(chapters[offset + i])
                    finish_count = offset + len(w_data)
                    last_char_state = processed_chaps[-1]["character_state"]
                    recent_texts = [c["text"] for c in processed_chaps[-3:]]
                    inferred_category = win.get("inferred_category", current_category)
                    protagonist_names = set(win.get("protagonist_names", []))
                    cache_valid = True
                    print(
                        f"✅ [阶段A] 窗口抢救成功！从第 {finish_count + 1} 章续跑，主角：{protagonist_names}"
                    )
        except Exception:
            pass

    remaining_chaps = chapters[finish_count:]
    pbar = tqdm(remaining_chaps, desc="阶段A进度")

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

        prompt_a = f"""你是网文剧情摘要助手。结合前文笔记生成本章摘要与人物状态。仅输出JSON。
【前文人物笔记】{safe_state_str}
【本章正文】{chap_text}
输出JSON：{{
  "chapter_summary": "200字内剧情摘要",
  "character_state": {{"角色名": "当前状态/位置/关系"}},{category_prompt}
}}"""
        try:
            resp = ollama_chat(prompt_a, 0.1, "A")
            data = safe_parse_json(resp)
            if not data:
                raise ValueError("解析失败")
            chap["character_state"] = flatten_character_state(
                data.get("character_state", {})
            )
            chap["summary"] = data.get("chapter_summary", "")
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

        last_char_state = chap["character_state"]
        processed_chaps.append(chap)
        recent_texts.append(chap["text"])
        if len(recent_texts) > 3:
            recent_texts.pop(0)

        if len(processed_chaps) % 10 == 0:
            save_state_atomic(
                WINDOW_FILE,
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
                        }
                        for c in processed_chaps[-50:]
                    ],
                },
            )
        if len(processed_chaps) % 200 == 0:
            save_state_atomic(
                STATE_FILE,
                {
                    "stage": "A",
                    "inferred_category": inferred_category,
                    "protagonist_names": list(protagonist_names),
                    "data": [
                        {
                            "id": c["id"],
                            "summary": c["summary"],
                            "character_state": c["character_state"],
                        }
                        for c in processed_chaps
                    ],
                },
            )

    save_state_atomic(
        STATE_FILE,
        {
            "stage": "A",
            "inferred_category": inferred_category,
            "protagonist_names": list(protagonist_names),
            "data": [
                {
                    "id": c["id"],
                    "summary": c["summary"],
                    "character_state": c["character_state"],
                }
                for c in processed_chaps
            ],
        },
    )
    return processed_chaps, inferred_category, protagonist_names


# ===================== 阶段 B：技法与爽点 =====================
def process_single_chapter_b(
    chap: Dict, book_name: str, category: str
) -> Dict[str, Any]:
    text = chap["text"]
    safe_text = text
    state_text = compress_state_to_text(chap["character_state"])

    prompt_b = f"""你是网文技法分析师。基于原文提取写作模板，输出纯JSON。
【书名】{book_name} 【章节】{chap["id"]} 【分类】{category}
【摘要】{chap["summary"]} 【人物状态】{state_text}
【正文】{safe_text}
输出JSON：{{
  "scene_type": "场景(打脸/升级/日常等)", 
  "narrative_skills": [{{"skill_name": "", "original_example": "", "analysis": "", "reuse_scenario": ""}}],
  "cool_point": {{"has_cool_point": false, "type": "", "quote": ""}}, 
  "style_feature": {{"tone": "文风调性"}}
}} (无爽点/技法请留空，禁止使用反引号)"""

    raw_resp = ollama_chat(prompt_b, 0.2, "B")
    res = safe_parse_json(raw_resp)
    if not res:
        if raw_resp.count("{") > raw_resp.count("}"):
            res = safe_parse_json(raw_resp + "}")
        if not res:
            raise Exception("JSON解析彻底失败")

    res.setdefault("narrative_skills", [])
    res.setdefault("scene_type", "未知")
    res.setdefault("cool_point", {"has_cool_point": False, "quote": ""})
    res.setdefault("style_feature", {"tone": "无"})
    res["raw_text"] = text

    if res["cool_point"].get("has_cool_point") and res["cool_point"].get("quote"):
        pos = find_quote_position_fast(text, res["cool_point"]["quote"])
        res["cool_point"]["char_pos"] = pos
        if pos == -1:
            res["_unmatched_log"] = {
                "chapter": chap["id"],
                "quote": res["cool_point"]["quote"],
            }

    res.update({"chapter_id": chap["id"], "book_name": book_name, "category": category})
    return res


def run_stage_b(chapters: List[Dict], book_name: str, category: str) -> List[Dict]:
    print("=== 阶段二：多线程提取技法与爽点 ===")
    CACHE_FILE = get_state_file(book_name, "B")
    success_list, fail_list, log_buffer = [], [], []
    completed_ids = set()

    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f).get("data", [])
                success_list.extend(cache)
                completed_ids = {x["chapter_id"] for x in cache}
            print(f"✅ [阶段B] 恢复断点：已完成 {len(cache)} 章")
        except Exception:
            pass

    pending = [c for c in chapters if c["id"] not in completed_ids]
    if not pending:
        return success_list

    def worker_task(chap):
        return process_single_chapter_b(chap, book_name, category)

    with ThreadPoolExecutor(max_workers=STAGE_BC_WORKERS) as executor:
        futures = {executor.submit(worker_task, c): c["id"] for c in pending}
        for task in tqdm(as_completed(futures), total=len(futures), desc="阶段B进度"):
            chap_id = futures[task]
            try:
                res = task.result()
                success_list.append(res)
                completed_ids.add(chap_id)
                if res.get("_unmatched_log"):
                    log_buffer.append(res["_unmatched_log"])
                if len(completed_ids) % 10 == 0:
                    save_state_atomic(CACHE_FILE, {"data": success_list})
                    gc.collect()  # 新增：每10章强制清理一次内存垃圾
            except Exception as e:
                fail_list.append((chap_id, str(e)))

    save_state_atomic(CACHE_FILE, {"data": success_list})
    if log_buffer:
        with open(UNMATCHED_LOG, "a", encoding="utf-8") as f:
            for item in log_buffer:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    if fail_list:
        print(f"⚠️ [阶段B] {len(fail_list)} 章处理失败：{fail_list[:5]}...")
    return success_list


# ===================== 阶段 C：文风指纹与感官映射 =====================
def process_single_chapter_c(
    chap: Dict, book_name: str, category: str
) -> Dict[str, Any]:
    text = chap["text"]
    safe_text = text

    # 【1. 从源头修复】在 Prompt 中明确强调必须是纯字符串数组，禁止嵌套对象
    prompt_c = f"""你是顶尖文学编辑。请深度拆解本章原文的"文风指纹"、"情绪感官映射"，并【原封不动】摘录经典段落，输出纯JSON。
【书名】{book_name} 【分类】{category}
【正文】{safe_text}
输出JSON：{{
  "author_fingerprint": {{
    "preferred_verbs": ["作者偏爱的特色动词，限5个，必须是纯字符串"],
    "preferred_adjectives": ["偏爱的特色形容词，限5个，必须是纯字符串"],
    "environmental_imagery": ["环境描写常用意象，限5个，必须是纯字符串"],
    "signature_transitions": ["标志性的过渡句或修辞手法，限2个，必须是纯字符串，绝对禁止使用对象或字典嵌套！"]
  }},
  "sensory_mappings": [
    {{
      "emotion": "核心情绪",
      "show_not_tell": "原著中展示该情绪的生理反应/动作/环境细节(限50字)",
      "analysis": "为什么这种描写比直接写情绪更有质感(20字内)"
    }}
  ],
  "classic_excerpts": [
    {{
      "excerpt_text": "从原文中原封不动地摘录 1 段最能代表该作者文风的完整段落（严格控制在300到400字之间，包含标点）。必须是原汁原味的原文，禁止修改任何字词！必须保持句子完整，绝不能在句子中间截断（必须以句号、问号、叹号或省略号结尾）。优先选择包含完整'环境铺垫+动作冲突+情绪反馈'的段落。",
      "scene_type": "场景类型(如:战斗/环境/对话/心理)",
      "style_tag": "风格标签(如:肃杀/幽默/细腻/宏大)"
    }}
  ]
}} (如果没有明显特征或情绪，对应数组留空。classic_excerpts必须严格摘录原文，禁止使用反引号)"""

    raw_resp = ollama_chat(prompt_c, 0.3, "C")
    res = safe_parse_json(raw_resp)
    if not res:
        if raw_resp.count("{") > raw_resp.count("}"):
            res = safe_parse_json(raw_resp + "}")
        if not res:
            raise Exception("阶段C JSON解析失败")

    res.setdefault("author_fingerprint", {})
    res.setdefault("sensory_mappings", [])
    res.setdefault("classic_excerpts", [])

    # 【2. JSON解析增强】强制清洗 author_fingerprint，过滤非字符串项并安全转换
    fp = res.get("author_fingerprint", {})
    if isinstance(fp, dict):
        for key in [
            "preferred_verbs",
            "preferred_adjectives",
            "environmental_imagery",
            "signature_transitions",
        ]:
            val = fp.get(key, [])
            if isinstance(val, list):
                # 过滤掉 dict/list 等复杂类型，只保留基础类型并强制转为 str
                fp[key] = [
                    str(v) for v in val if isinstance(v, (str, int, float, bool))
                ]
            else:
                fp[key] = []
    else:
        fp = {}
    res["author_fingerprint"] = fp

    res.update({"chapter_id": chap["id"], "book_name": book_name, "category": category})
    return res


def run_stage_c(chapters: List[Dict], book_name: str, category: str) -> List[Dict]:
    print("=== 阶段三：多线程提取文风指纹与感官映射 ===")
    CACHE_FILE = get_state_file(book_name, "C")
    success_list, fail_list = [], []
    completed_ids = set()

    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f).get("data", [])
                success_list.extend(cache)
                completed_ids = {x["chapter_id"] for x in cache}
            print(f"✅ [阶段C] 恢复断点：已完成 {len(cache)} 章")
        except Exception:
            pass

    pending = [c for c in chapters if c["id"] not in completed_ids]
    if not pending:
        return success_list

    def worker_task(chap):
        return process_single_chapter_c(chap, book_name, category)

    with ThreadPoolExecutor(max_workers=STAGE_BC_WORKERS) as executor:
        futures = {executor.submit(worker_task, c): c["id"] for c in pending}
        for task in tqdm(as_completed(futures), total=len(futures), desc="阶段C进度"):
            chap_id = futures[task]
            try:
                res = task.result()
                success_list.append(res)
                completed_ids.add(chap_id)
                if len(completed_ids) % 10 == 0:
                    save_state_atomic(CACHE_FILE, {"data": success_list})
                    gc.collect()  # 新增：每10章强制清理一次内存垃圾
            except Exception as e:
                fail_list.append((chap_id, str(e)))

    save_state_atomic(CACHE_FILE, {"data": success_list})
    if fail_list:
        print(f"⚠️ [阶段C] {len(fail_list)} 章处理失败：{fail_list[:5]}...")
    return success_list


# ===================== 数据库与入库逻辑 =====================
def init_database_resource(db_conn: Optional[sqlite3.Connection] = None):
    if db_conn is None:
        db_conn = sqlite3.connect(SQLITE_PATH, timeout=30.0, check_same_thread=False)
        db_conn.execute("PRAGMA journal_mode=WAL;")
        db_conn.execute("PRAGMA synchronous=NORMAL;")

    cursor = db_conn.cursor()
    TABLE_SCHEMAS = {
        "skills": "(id TEXT PRIMARY KEY, book_name TEXT, chapter_id TEXT, category TEXT, scene_type TEXT, skill_name TEXT, analysis TEXT, original_example TEXT, tags TEXT)",
        "plot_arcs": "(chapter_id TEXT PRIMARY KEY, book_name TEXT, category TEXT, summary TEXT, character_state_json TEXT)",
        "author_fingerprints": "(id TEXT PRIMARY KEY, book_name TEXT, category TEXT, verbs TEXT, adjectives TEXT, imagery TEXT, transitions TEXT)",
        "sensory_mappings": "(id TEXT PRIMARY KEY, book_name TEXT, chapter_id TEXT, category TEXT, emotion TEXT, show_not_tell TEXT, analysis TEXT)",
    }
    CHAPTER_ID_TABLES = ["skills", "plot_arcs", "sensory_mappings"]

    print("🔍 正在执行数据库结构强制校验...")
    for table_name, schema in TABLE_SCHEMAS.items():
        try:
            cols = [
                row[1]
                for row in cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
            ]
            need_rebuild = len(cols) == 0
            if (
                not need_rebuild
                and table_name in CHAPTER_ID_TABLES
                and "chapter_id" not in cols
            ):
                need_rebuild = True
            if need_rebuild:
                print(f"🔧 表 [{table_name}] 结构缺失或不兼容，正在强制重建...")
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                cursor.execute(f"CREATE TABLE {table_name} {schema}")
            else:
                cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} {schema}")
        except Exception as e:
            print(f"⚠️ 表 [{table_name}] 校验异常: {e}，强制重建...")
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            cursor.execute(f"CREATE TABLE {table_name} {schema}")
    db_conn.commit()
    print("✅ 数据库结构校验完毕。")

    # 初始化 ChromaDB（使用默认配置，兼容新版 Rust 后端）
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    skill_collection = chroma_client.get_or_create_collection(name="novel_skills")
    sensory_collection = chroma_client.get_or_create_collection(name="sensory_details")
    # 🌟 新增：经典文风段落集合 (Few-Shot 典例)
    excerpts_collection = chroma_client.get_or_create_collection(
        name="classic_excerpts"
    )

    graph_path = os.path.join(BASE_DIR, "knowledge_graph.graphml")
    graph = nx.DiGraph()
    if os.path.exists(graph_path):
        try:
            graph = nx.read_graphml(graph_path)
        except Exception:
            try:
                os.remove(graph_path)
            except Exception:
                pass

    return (
        db_conn,
        skill_collection,
        sensory_collection,
        excerpts_collection,
        graph,
        graph_path,
    )


def insert_knowledge(
    stage_a_res: List[Dict],
    stage_b_res: List[Dict],
    stage_c_res: List[Dict],
    db_conn: sqlite3.Connection,
    author: str = "未知作者",  # 🌟 新增参数
):
    # 1. 初始化数据库与向量库资源
    (
        db_conn,
        skill_collection,
        sensory_collection,
        excerpts_collection,
        graph,
        graph_path,
    ) = init_database_resource(db_conn)
    cursor = db_conn.cursor()

    # 2. 提取书名和分类
    book, category = "", ""
    for res_list in [stage_b_res, stage_c_res, stage_a_res]:
        if res_list:
            book = res_list[0].get("book_name", "未知")
            category = res_list[0].get("category", "未知")
            break

    # 🌟 新增：确保 author 变量在函数内可用
    safe_author = author if author else "未知作者"

    # 🌟 3. 初始化全局战报计数器
    stats = {
        "plot_arcs": 0,
        "graph_nodes": 0,
        "skills_db": 0,
        "skills_chroma": 0,
        "fingerprints_db": 0,
        "sensory_db": 0,
        "sensory_chroma": 0,
        "excerpts_chroma": 0,
    }

    # ================= 入库 Stage A (剧情与图谱) =================
    if stage_a_res:
        logger.info("📥 正在入库剧情脉络与人物图谱...")
        existing_plot_ids = set(
            row[0]
            for row in cursor.execute(
                "SELECT chapter_id FROM plot_arcs WHERE book_name = ?", (book,)
            ).fetchall()
        )
        for chap in tqdm(stage_a_res, desc="入库剧情"):
            if chap["id"] not in existing_plot_ids:
                cursor.execute(
                    "INSERT OR REPLACE INTO plot_arcs VALUES (?,?,?,?,?)",
                    (
                        chap["id"],
                        book,
                        category,
                        chap.get("summary", ""),
                        json.dumps(chap.get("character_state", {}), ensure_ascii=False),
                    ),
                )
                stats["plot_arcs"] += 1

            # 图谱人物节点与边
            for char_name, char_state in chap.get("character_state", {}).items():
                if char_name in ("_raw", "旁白"):
                    continue
                char_node = f"char:{char_name}"
                if not graph.has_node(char_node):
                    stats["graph_nodes"] += 1
                graph.add_node(char_node, node_type="character", book_list=book)
                safe_append_edge_attr(
                    graph,
                    char_node,
                    f"chap:{chap['id']}",
                    "action",
                    str(char_state)[:50],
                )

        db_conn.commit()
        logger.info(
            f"   ✅ [阶段A战报] 剧情表(plot_arcs)新增: {stats['plot_arcs']} 条 | 图谱新增人物节点: {stats['graph_nodes']} 个"
        )

    # ================= 入库 Stage B (写作技法) =================
    # 统一清洗 chapter_id (去除上下半段后缀)
    for item in stage_c_res:
        if "chapter_id" in item:
            item["chapter_id"] = (
                item["chapter_id"].replace("_上半段", "").replace("_下半段", "")
            )

    if stage_b_res:
        logger.info("📥 正在入库写作技法...")
        existing_skill_ids = set(
            row[0]
            for row in cursor.execute(
                "SELECT id FROM skills WHERE book_name = ?", (book,)
            ).fetchall()
        )

        # 合并上下半段数据
        merge_map = defaultdict(list)
        for item in stage_b_res:
            # 【安全提取】防止大模型返回残缺字典
            raw_id = item.get("chapter_id", "未知章节")
            pure_id = raw_id.replace("_上半段", "").replace("_下半段", "")
            item["chapter_id"] = pure_id
            merge_map[pure_id].append(item)

        unified_b = []
        for chap_id, slices in merge_map.items():
            base = copy.deepcopy(slices[0])
            base["narrative_skills"] = [
                s for sl in slices for s in sl.get("narrative_skills", [])
            ]
            base["scene_type"] = (
                "/".join(
                    set(
                        sl.get("scene_type", "未知")
                        for sl in slices
                        if sl.get("scene_type")
                    )
                )
                or "未知"
            )
            if len(slices) > 1:
                base["raw_text"] = "\n".join(sl.get("raw_text", "") for sl in slices)
            unified_b.append(base)

        batch_ids, batch_docs, batch_metas, db_count = [], [], [], 0
        for item in tqdm(unified_b, desc="入库技法"):
            # 确保 narrative_skills 是列表，且过滤掉非字典的脏数据
            skills_list = item.get("narrative_skills", [])
            if not isinstance(skills_list, list):
                continue

            for skill in skills_list:
                if not isinstance(skill, dict):
                    continue

                # 【安全提取】防止 KeyError，缺失字段用默认值填补
                s_name = skill.get("skill_name", "未命名技法")
                s_example = skill.get("original_example", "无示例")
                s_analysis = skill.get("analysis", "")

                # 兼容大模型把字段名写错的情况（如写成 reason 或 description）
                if not s_analysis:
                    s_analysis = skill.get(
                        "reason", skill.get("description", "大模型未返回分析")
                    )

                sid = hashlib.md5(
                    f"{book}|{item['chapter_id']}|{s_name}|{s_example}".encode()
                ).hexdigest()
                if sid in existing_skill_ids:
                    continue

                # SQLite 入库
                cursor.execute(
                    "INSERT OR IGNORE INTO skills VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        sid,
                        book,
                        item["chapter_id"],
                        category,
                        item.get("scene_type", "未知"),
                        s_name,
                        s_analysis,
                        s_example,
                        f"|{item.get('scene_type', '未知')}|{category}|{s_name}|",
                    ),
                )
                stats["skills_db"] += 1
                db_count += 1

                # ChromaDB 批次准备 (同步使用安全提取后的变量)
                batch_ids.append(sid)
                batch_docs.append(f"技法:{s_name}\n逻辑:{s_analysis}\n示例:{s_example}")
                batch_metas.append(
                    {
                        "book_name": book,
                        "author": safe_author,  # 🌟 新增
                        "chapter": item["chapter_id"],
                        "category": category,
                        "scene": item.get("scene_type", "未知"),
                    }
                )

                # 图谱逻辑
                scene_node = f"scene:{safe_str(item.get('scene_type', '未知'))}"
                skill_node = f"skill:{safe_str(s_name)}:{safe_str(category)}"
                graph.add_node(scene_node, node_type="scene")
                if graph.has_node(skill_node):
                    old_books = safe_str(graph.nodes[skill_node].get("book_list", ""))
                    if book not in old_books:
                        graph.nodes[skill_node]["book_list"] = (
                            f"{old_books},{book}" if old_books else book
                        )
                else:
                    graph.add_node(skill_node, node_type="skill", book_list=book)
                safe_append_edge_attr(graph, scene_node, skill_node, "relation", "包含")

                # 批次提交 ChromaDB
                if len(batch_ids) >= CHROMA_BATCH_SIZE:
                    try:
                        skill_collection.upsert(
                            ids=batch_ids, documents=batch_docs, metadatas=batch_metas
                        )
                        stats["skills_chroma"] += len(batch_ids)
                    except Exception as e:
                        logger.error(f"⚠️ ChromaDB skills upsert 失败: {e}")
                    batch_ids, batch_docs, batch_metas = [], [], []

                # 批次提交 SQLite
                if db_count >= SQL_COMMIT_CHUNK:
                    db_conn.commit()
                    db_count = 0

        # 处理剩余尾批
        if db_count > 0:
            db_conn.commit()
        if batch_ids:
            try:
                skill_collection.upsert(
                    ids=batch_ids, documents=batch_docs, metadatas=batch_metas
                )
                stats["skills_chroma"] += len(batch_ids)
            except Exception as e:
                logger.error(f"⚠️ ChromaDB skills 最终批次 upsert 失败: {e}")

        logger.info(
            f"   ✅ [阶段B战报] 技法表(skills)新增: {stats['skills_db']} 条 | 向量库(novel_skills)新增: {stats['skills_chroma']} 条"
        )

    # ================= 入库 Stage C (文风指纹与感官映射) =================
    if stage_c_res:
        logger.info("📥 正在入库文风指纹与感官映射...")
        existing_fp_ids = set(
            row[0]
            for row in cursor.execute(
                "SELECT id FROM author_fingerprints WHERE book_name = ?", (book,)
            ).fetchall()
        )
        existing_sm_ids = set(
            row[0]
            for row in cursor.execute(
                "SELECT id FROM sensory_mappings WHERE book_name = ?", (book,)
            ).fetchall()
        )
        s_batch_ids, s_batch_docs, s_batch_metas = [], [], []
        c_db_count = 0  # 【新增】Stage C 的 SQLite 提交计数器

        for item in tqdm(stage_c_res, desc="入库文风"):
            # 1. 作者指纹入库
            fp = item.get("author_fingerprint", {})
            if any(fp.values()):
                # 【安全提取】
                c_id = item.get("chapter_id", "未知章节")
                fp_id = hashlib.md5(f"{book}|{c_id}|fp".encode()).hexdigest()
                if fp_id not in existing_fp_ids:
                    # 【3. 安全转换 + 过滤非字符串项】构建绝对安全的字符串列表
                    safe_verbs = [
                        str(v)
                        for v in fp.get("preferred_verbs", [])
                        if isinstance(v, (str, int, float))
                    ]
                    safe_adjs = [
                        str(v)
                        for v in fp.get("preferred_adjectives", [])
                        if isinstance(v, (str, int, float))
                    ]
                    safe_imgs = [
                        str(v)
                        for v in fp.get("environmental_imagery", [])
                        if isinstance(v, (str, int, float))
                    ]
                    safe_trans = [
                        str(v)
                        for v in fp.get("signature_transitions", [])
                        if isinstance(v, (str, int, float))
                    ]

                    cursor.execute(
                        "INSERT OR IGNORE INTO author_fingerprints VALUES (?,?,?,?,?,?,?)",
                        (
                            fp_id,
                            book,
                            category,
                            ",".join(safe_verbs),
                            ",".join(safe_adjs),
                            ",".join(safe_imgs),
                            "||".join(safe_trans),
                        ),
                    )
                    stats["fingerprints_db"] += 1

            # 2. 感官映射入库 (SQLite + ChromaDB)
            for sm in item.get("sensory_mappings", []):
                if not isinstance(sm, dict):
                    continue

                # 【安全提取】
                emotion = sm.get("emotion", "未知情绪")
                show_detail = sm.get("show_not_tell", "")
                analysis = sm.get("analysis", "")

                if show_detail:
                    sm_id = hashlib.md5(
                        f"{book}|{c_id}|{emotion}|{show_detail}".encode()
                    ).hexdigest()
                    if sm_id in existing_sm_ids:
                        continue
                    cursor.execute(
                        "INSERT OR IGNORE INTO sensory_mappings VALUES (?,?,?,?,?,?,?)",
                        (
                            sm_id,
                            book,
                            item["chapter_id"],
                            category,
                            emotion,
                            show_detail,
                            analysis,
                        ),
                    )
                    stats["sensory_db"] += 1

                    s_batch_ids.append(sm_id)
                    s_batch_docs.append(
                        f"情绪:{emotion}\n细节展示:{show_detail}\n分析:{analysis}"
                    )
                    s_batch_metas.append(
                        {
                            "book_name": book,
                            "author": safe_author,  # 🌟 新增
                            "category": category,
                            "emotion": emotion,
                        }
                    )

                    if len(s_batch_ids) >= CHROMA_BATCH_SIZE:
                        try:
                            sensory_collection.upsert(
                                ids=s_batch_ids,
                                documents=s_batch_docs,
                                metadatas=s_batch_metas,
                            )
                            stats["sensory_chroma"] += len(s_batch_ids)
                        except Exception as e:
                            logger.error(f"⚠️ ChromaDB sensory upsert 失败: {e}")
                        s_batch_ids, s_batch_docs, s_batch_metas = [], [], []

            c_db_count += 1
            # 【4. 断点续传优化】每 500 条强制提交一次 SQLite，防止中途崩溃导致数据全丢
            if c_db_count >= SQL_COMMIT_CHUNK:
                db_conn.commit()
                c_db_count = 0

        # 处理感官映射尾批
        if s_batch_ids:
            try:
                sensory_collection.upsert(
                    ids=s_batch_ids, documents=s_batch_docs, metadatas=s_batch_metas
                )
                stats["sensory_chroma"] += len(s_batch_ids)
            except Exception as e:
                logger.error(f"⚠️ ChromaDB sensory 最终批次 upsert 失败: {e}")

        # 【新增】Stage C 最终 SQLite 提交
        if c_db_count > 0:
            db_conn.commit()

        # 3. 经典文风段落入库 (Few-Shot 典例)
        logger.info("📥 正在入库经典文风典例...")
        e_batch_ids, e_batch_docs, e_batch_metas = [], [], []
        for item in tqdm(stage_c_res, desc="入库典例"):
            for exc in item.get("classic_excerpts", []):
                if not isinstance(exc, dict):
                    continue
                excerpt_text = exc.get("excerpt_text", "")

                # 【新增】乱码拦截：检测 GBK 误读为 UTF-8 产生的典型乱码特征
                is_garbled = False
                if excerpt_text:
                    try:
                        # 尝试反向编码验证，如果成功说明是乱码
                        excerpt_text.encode("utf-8").decode("ascii")
                    except UnicodeDecodeError:
                        pass
                    # 检测连续的非CJK、非标点的异常Unicode字符（GBK乱码的典型特征）
                    garbled_pattern = re.compile(r"(?:[\x80-\xff]{3,}|[À-ÿ]{2,})")
                    if garbled_pattern.search(excerpt_text):
                        is_garbled = True

                if is_garbled:
                    logger.warning(
                        f"⚠️ 跳过乱码典例: chapter={item.get('chapter_id', '未知')}, text={excerpt_text[:50]}..."
                    )
                    continue

                if excerpt_text and len(excerpt_text) > 20:
                    e_id = hashlib.md5(
                        f"{book}|{item['chapter_id']}|{excerpt_text[:50]}".encode()
                    ).hexdigest()
                    e_batch_ids.append(e_id)
                    e_batch_docs.append(excerpt_text)
                    e_batch_metas.append(
                        {
                            "book_name": book,
                            "author": safe_author,  # 🌟 新增
                            "category": category,
                            "chapter": item["chapter_id"],
                            "scene_type": exc.get("scene_type", "未知"),
                            "style_tag": exc.get("style_tag", "未知"),
                        }
                    )

                    if len(e_batch_ids) >= CHROMA_BATCH_SIZE:
                        try:
                            excerpts_collection.upsert(
                                ids=e_batch_ids,
                                documents=e_batch_docs,
                                metadatas=e_batch_metas,
                            )
                            stats["excerpts_chroma"] += len(e_batch_ids)
                        except Exception as e:
                            logger.error(f"⚠️ ChromaDB excerpts upsert 失败: {e}")
                        e_batch_ids, e_batch_docs, e_batch_metas = [], [], []

        # 处理典例尾批
        if e_batch_ids:
            try:
                excerpts_collection.upsert(
                    ids=e_batch_ids, documents=e_batch_docs, metadatas=e_batch_metas
                )
                stats["excerpts_chroma"] += len(e_batch_ids)
            except Exception as e:
                logger.error(f"⚠️ ChromaDB excerpts 最终批次 upsert 失败: {e}")

        db_conn.commit()
        logger.info(
            f"   ✅ [阶段C战报] 指纹表(fingerprints)新增: {stats['fingerprints_db']} 条 | 感官表(sensory)新增: {stats['sensory_db']} 条"
        )
        logger.info(
            f"      ↳ 向量库(sensory)新增: {stats['sensory_chroma']} 条 | 向量库(典例excerpts)新增: {stats['excerpts_chroma']} 条"
        )

    # ================= 清理资源与保存图谱 =================
    del skill_collection, sensory_collection, excerpts_collection
    gc.collect()

    sanitize_graph_for_graphml(graph)
    try:
        nx.write_graphml(graph, graph_path)
    except Exception as e:
        logger.error(f"⚠️ 图谱保存失败: {e}")

    # 🌟 最终汇总战报
    logger.info("")
    logger.info("=" * 50)
    logger.info(f"🏆 《{book}》 终极入库战报汇总：")
    logger.info(
        f"   📊 SQLite 关系库总计新增: {stats['plot_arcs'] + stats['skills_db'] + stats['fingerprints_db'] + stats['sensory_db']} 条"
    )
    logger.info(
        f"   🧠 ChromaDB 向量库总计新增: {stats['skills_chroma'] + stats['sensory_chroma'] + stats['excerpts_chroma']} 条"
    )
    logger.info("=" * 50)
    logger.info("✅ 所有数据入库完成！")


# ===================== 调度与文件处理 =====================
def clean_book_name(raw_name: str) -> tuple:
    """
    🧹 核心修复：从规范的文件名中精准剥离“书名”和“后缀标记”
    输入: 《老婆孩子热炕头》作者：水千丞[番外]
    输出: ('老婆孩子热炕头', '[番外]')
    """
    # 1. 提取书名号内的内容作为纯净书名
    match = re.search(r"《(.*?)》", raw_name)
    if match:
        pure_book_name = match.group(1).strip()
    else:
        # 如果没有书名号，尝试去掉“作者：”及之后的内容
        pure_book_name = re.split(r"作者[：:]|by\s*", raw_name, flags=re.IGNORECASE)[
            0
        ].strip()
        # 如果还是没切开，就去掉常见的后缀
        pure_book_name = re.sub(
            r"\[番外\]|\(番外\)|番外|补车|精校版|未删减", "", pure_book_name
        ).strip()

    # 2. 提取后缀标记（如 [番外]）
    suffix_match = re.search(r"(\[番外\]|\[补车\]|\[精校\]|\(番外\))", raw_name)
    suffix = suffix_match.group(1) if suffix_match else ""

    return pure_book_name, suffix


def scan_novel_library(root_dir: str) -> List[Dict[str, Any]]:
    print(f"🔍 正在扫描小说库：{root_dir}")
    all_txt = glob.glob(os.path.join(root_dir, "**", "*.txt"), recursive=True)

    book_list = []
    for path in all_txt:
        rel_path = os.path.relpath(path, root_dir)
        parts = rel_path.split(os.sep)

        # 🚨 核心修复 1：智能提取分类（优先取第二级文件夹，即作者名/合集名）
        if len(parts) >= 3:
            # 例如: 作者合集小说 / 东度日（12本） / 文件.txt
            category = parts[1]
            # 清洗掉文件夹名里的“（12本）”、“合集”等字眼，保留纯净的作者名
            category = (
                re.sub(r"[\(（].*?[\)）]", "", category).replace("合集", "").strip()
            )
        elif len(parts) == 2:
            category = parts[0]
        else:
            category = "未分类"

        raw_file_name = os.path.splitext(os.path.basename(path))[0]

        # 🚨 核心修复 2：精准剥离书名和后缀
        pure_book_name, suffix = clean_book_name(raw_file_name)

        # 🌟 新增：精准提取作者名 (支持 "作者：xxx" 或 "by xxx")
        author_match = re.search(
            r"作者[：:]\s*([^\[\/]+)|by\s+([^\[\/]+)", raw_file_name, re.IGNORECASE
        )
        author_name = (
            (author_match.group(1) or author_match.group(2)).strip()
            if author_match
            else "未知作者"
        )
        # 如果带有 [番外] 后缀，我们在内部处理时把它追加到书名后面，防止和正文主键冲突
        # 但在日志和图谱显示时，它依然属于同一本书
        db_book_name = f"{pure_book_name}{suffix}" if suffix else pure_book_name

        book_list.append(
            {
                "book_name": db_book_name,
                "pure_name": pure_book_name,
                "author": author_name,  # 🌟 新增作者字段
                "category": category,  # 这里的 category 后续会被阶段A的 inferred_category (如"悬疑修仙") 覆盖
                "all_files": [path],
            }
        )

    print(f"📊 扫描完成，共发现 {len(book_list)} 本独立小说。")
    return book_list


def merge_txt_files(file_list: List[str], output_path: str) -> str:
    if len(file_list) == 1:
        return file_list[0]
    with open(output_path, "w", encoding="utf-8") as out:
        for f in sorted(file_list):
            try:
                with open(f, "r", encoding="utf-8") as inp:
                    out.write(inp.read() + "\n\n")
            except Exception:
                try:
                    with open(f, "r", encoding="gbk") as inp:
                        out.write(inp.read() + "\n\n")
                except Exception:
                    with open(f, "rb") as inp:
                        out.write(
                            inp.read().decode("latin-1", errors="ignore") + "\n\n"
                        )
    return output_path


def process_single_book(book_info: Dict, manifest: Dict, db_conn: sqlite3.Connection):
    book_name = book_info["book_name"]
    cursor = db_conn.cursor()
    merge_path = os.path.join(BASE_DIR, f"temp_{book_name}.txt")
    text_path = merge_txt_files(book_info["all_files"], merge_path)

    try:
        chapters = load_chapters_from_txt(
            text_path, book_name, book_info["category"], SPLIT_THRESHOLD
        )
        total_chapters = len(chapters)

        try:
            cursor.execute(
                "SELECT COUNT(DISTINCT chapter_id) FROM plot_arcs WHERE book_name = ?",
                (book_name,),
            )
            db_a = cursor.fetchone()[0]
            cursor.execute(
                "SELECT COUNT(DISTINCT chapter_id) FROM skills WHERE book_name = ?",
                (book_name,),
            )
            db_b = cursor.fetchone()[0]
            cursor.execute(
                "SELECT COUNT(DISTINCT chapter_id) FROM sensory_mappings WHERE book_name = ?",
                (book_name,),
            )
            db_c = cursor.fetchone()[0]
        except Exception:
            db_a = db_b = db_c = 0

        if db_a >= total_chapters and db_b >= total_chapters and db_c >= total_chapters:
            print(f"⏭️ 跳过已完美入库书籍：{book_name}")
            if book_name not in manifest["completed_books"]:
                manifest["completed_books"].append(book_name)
                save_manifest(manifest)
            return

        if book_name in manifest["completed_books"]:
            manifest["completed_books"].remove(book_name)
            save_manifest(manifest)

        print(
            f"\n{'='*20} 开始处理：《{book_name}》 (总章数:{total_chapters}) {'='*20}"
        )
        manifest["current_processing"] = book_name
        save_manifest(manifest)

        stage_a_res, inferred_cat, _ = run_stage_a(
            chapters, book_name, book_info["category"]
        )
        if inferred_cat and inferred_cat != "未分类":
            book_info["category"] = inferred_cat
            for c in stage_a_res:
                c["category"] = inferred_cat

        stage_b_res = run_stage_b(stage_a_res, book_name, book_info["category"])
        stage_c_res = run_stage_c(stage_a_res, book_name, book_info["category"])

        insert_knowledge(
            stage_a_res,
            stage_b_res,
            stage_c_res,
            db_conn,
            author=book_info.get("author", "未知作者"),
        )

        manifest["completed_books"].append(book_name)
        manifest["current_processing"] = None
        save_manifest(manifest)

        for stage in ["A", "B", "C"]:
            f = get_state_file(book_name, stage)
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass
        f_win = get_window_file(book_name)
        if os.path.exists(f_win):
            try:
                os.remove(f_win)
            except Exception:
                pass
        print(f"🎉 《{book_name}》 知识库构建完成！")
    finally:
        if os.path.exists(merge_path) and text_path == merge_path:
            try:
                os.remove(merge_path)
            except Exception:
                pass
        gc.collect()


def main():
    NOVELS_ROOT_DIR = r"D:\WorkFish\Novel-Knowledge-Base\novels"

    # 【重要】启动前健康检查
    check_ollama_health()

    db_conn = sqlite3.connect(SQLITE_PATH, timeout=30.0, check_same_thread=False)
    db_conn.execute("PRAGMA journal_mode=WAL;")
    db_conn.execute("PRAGMA synchronous=NORMAL;")

    init_database_resource(db_conn)
    manifest = load_manifest()
    novel_list = scan_novel_library(NOVELS_ROOT_DIR)

    if not novel_list:
        print("❌ 未找到任何 TXT 小说。")
        db_conn.close()
        return

    new_books = [
        b for b in novel_list if b["book_name"] not in manifest["completed_books"]
    ]
    print(
        f"📊 调度清单：共扫描 {len(novel_list)} 本，已完工 {len(manifest['completed_books'])} 本，待处理 {len(new_books)} 本。"
    )

    for idx, book_info in enumerate(new_books):
        print(
            f"\n🚀 进度：[{idx+1}/{len(new_books)}] 目标：《{book_info['book_name']}》 [分类:{book_info['category']}]"
        )
        try:
            process_single_book(book_info, manifest, db_conn)
        except Exception as e:
            error_msg = traceback.format_exc()
            print(f"❌ 处理《{book_info['book_name']}》时发生致命错误：\n{error_msg}")
            with open(
                os.path.join(BASE_DIR, "fatal_errors.log"), "a", encoding="utf-8"
            ) as f:
                f.write(f"=== {book_info['book_name']} ===\n{error_msg}\n")

    db_conn.close()
    GLOBAL_SESSION.close()
    print("\n🏆 恭喜！整个小说库工业化构建全部执行完成！")


if __name__ == "__main__":
    main()
