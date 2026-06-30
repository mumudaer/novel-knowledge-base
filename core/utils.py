"""
通用工具函数模块
提供文本处理、JSON 解析、文件操作等通用功能
"""
import os
import re
import json
import hashlib
import logging
from typing import List, Dict, Any, Tuple, Optional
from config.settings import BASE_DIR, MANIFEST_FILE, SPLIT_THRESHOLD, SPLIT_OVERLAP

logger = logging.getLogger(__name__)


# ===================== 数据库查询工具 =====================

def query_to_dicts(
    cursor,
    sql: str,
    params: tuple = (),
    columns: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    执行 SQL 查询并将结果转换为字典列表
    
    Args:
        cursor: 数据库游标
        sql: SQL 查询语句
        params: SQL 参数
        columns: 列名列表（如果不提供，从 cursor.description 自动获取）
    
    Returns:
        字典列表，每个字典代表一行数据
    
    Example:
        >>> rows = query_to_dicts(cursor, "SELECT * FROM books WHERE id = ?", (book_id,))
        >>> for row in rows:
        ...     print(row["title"])
    """
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    
    # 如果没有提供列名，从 cursor.description 获取
    if columns is None and cursor.description:
        columns = [desc[0] for desc in cursor.description]
    
    if not columns:
        return []
    
    return [dict(zip(columns, row)) for row in rows]


# ===================== 文本处理工具 =====================

def clean_novel_text(text: str) -> str:
    """清理小说文本（去除广告、乱码、多余空白）"""
    if not text:
        return ""

    # 移除常见广告文本
    ad_patterns = [
        r"最新网址：.*?\.com",
        r"手机版阅读网址：.*",
        r"天才一秒记住.*?秒",
        r"本站.*?域名",
        r"www\..*?\.com",
        r"手机阅读.*",
        r"一秒记住.*",
    ]
    for pattern in ad_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # 移除乱码字符
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    # 规范化空白字符
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def _find_semantic_boundary(text: str, start_pos: int, max_chunk: int) -> int:
    """
    在 [start_pos, start_pos + max_chunk] 范围内，寻找最佳语义切分点。
    优先级从高到低：
    1. 场景转换标记（"...", "---", "***"）
    2. 双换行（段落边界）
    3. 对话结束标记（"..."、"。"后的换行）
    4. 句号/问号/叹号后的换行
    5. 如果找不到自然边界，回退到 max_chunk 位置
    """
    search_region = text[start_pos:start_pos + max_chunk]
    best_pos = -1
    best_priority = 999

    # 从后往前搜索，优先在靠近 max_chunk 的位置切分（最大化块大小）
    search_start = max(len(search_region) // 2, 100)  # 至少保留一半内容

    for i in range(len(search_region) - 1, search_start - 1, -1):
        ch = search_region[i]

        # 优先级1：场景转换标记
        if i > 2 and search_region[i-2:i+1] in ("...", "---", "***", "……"):
            if best_priority > 1:
                best_pos = start_pos + i + 1
                best_priority = 1
                break

        # 优先级2：双换行（段落边界）
        if i > 0 and search_region[i-1:i+1] == "\n\n":
            if best_priority > 2:
                best_pos = start_pos + i + 1
                best_priority = 2

        # 优先级3：对话结束后的换行
        if i > 0 and ch == "\n" and i >= 2 and search_region[i-2] in ("”", "\u201d", '"', '。', '！', '？', '.', '!', '?'):
            if best_priority > 3:
                best_pos = start_pos + i + 1
                best_priority = 3

        # 优先级4：句号/问号/叹号后的换行
        if i > 0 and ch == "\n" and search_region[i-1] in ("。", "！", "？", ".", "!", "?"):
            if best_priority > 4:
                best_pos = start_pos + i + 1
                best_priority = 4

        # 找到优先级1或2就可以停了
        if best_priority <= 2:
            break

    if best_pos > 0:
        return best_pos

    # 找不到自然边界，硬切
    return start_pos + max_chunk


def smart_split_chapters(
    text: str,
    book_name: str = "未知书籍",
    max_chunk: int = SPLIT_THRESHOLD,
    overlap: int = SPLIT_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    智能切分章节
    1. 优先按章节标题切分
    2. 对过长章节进行二次切分，支持语义边界感知和滑动窗口重叠
    3. 对无章节标题的长文，按滑动窗口切分
    """
    # 常见的章节标题模式
    chapter_patterns = [
        r"^第[一二三四五六七八九十百千万零\d]+[章节回卷集部篇]\s*.*$",
        r"^Chapter\s*\d+.*$",
        r"^\d+[\.\s].*$",
    ]

    lines = text.split("\n")
    chapters = []
    current_chapter = {"id": "序章", "text": "", "book_name": book_name}
    chapter_index = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 检查是否是章节标题
        is_chapter_title = False
        for pattern in chapter_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                is_chapter_title = True
                break

        if is_chapter_title and current_chapter["text"].strip():
            # 保存当前章节
            if len(current_chapter["text"]) > 100:  # 过滤过短的章节
                chapters.append(current_chapter)
                chapter_index += 1

            # 开始新章节
            current_chapter = {
                "id": f"第{chapter_index + 1}章",
                "text": "",
                "book_name": book_name,
            }

        current_chapter["text"] += line + "\n"

    # 保存最后一章
    if current_chapter["text"].strip() and len(current_chapter["text"]) > 100:
        chapters.append(current_chapter)

    # 二次切分过长的章节（语义边界感知 + 滑动窗口重叠）
    final_chapters = []
    for chap in chapters:
        if len(chap["text"]) > max_chunk * 2:
            # 使用语义边界感知切分
            text_content = chap["text"]
            pos = 0
            sub_index = 1

            while pos < len(text_content):
                # 寻找语义边界
                cut_pos = _find_semantic_boundary(text_content, pos, max_chunk)

                chunk_text = text_content[pos:cut_pos].strip()
                if chunk_text and len(chunk_text) > 50:  # 过滤过短的片段
                    final_chapters.append({
                        "id": f"{chap['id']}_{sub_index}",
                        "text": chunk_text,
                        "book_name": book_name,
                    })
                    sub_index += 1

                # 如果已经切到文本末尾，结束循环
                if cut_pos >= len(text_content):
                    break

                # 滑动窗口：下一个块的起点回退 overlap 字符
                pos = cut_pos
                if overlap > 0:
                    new_pos = max(pos - overlap, 0)
                    # 防止死循环：确保每轮至少前进 1 个字符
                    if new_pos <= pos - max_chunk:
                        pos = cut_pos  # 回退太多，不回退
                    else:
                        pos = new_pos

        else:
            final_chapters.append(chap)

    return final_chapters


# ===================== 人物状态处理工具 =====================

def safe_str(val: Any, default: str = "未知") -> str:
    """安全转换为字符串"""
    if val is None:
        return default
    if isinstance(val, str):
        return val.strip() or default
    return str(val)


def flatten_character_state(state: Any) -> Dict[str, str]:
    """扁平化人物状态（处理嵌套结构）"""
    if not isinstance(state, dict):
        return {}

    flattened = {}
    for key, value in state.items():
        if isinstance(value, dict):
            # 递归扁平化
            for sub_key, sub_value in value.items():
                flattened[f"{key}_{sub_key}"] = safe_str(sub_value)
        elif isinstance(value, list):
            flattened[key] = ", ".join([safe_str(v) for v in value])
        else:
            flattened[key] = safe_str(value)

    return flattened


def compress_state_to_text(state_dict: Dict[str, str]) -> str:
    """将人物状态压缩为文本（用于 Prompt）"""
    if not state_dict:
        return "无"

    lines = []
    for name, state in state_dict.items():
        if name in ("_raw", "旁白"):
            continue
        lines.append(f"{name}: {state}")

    return "\n".join(lines) if lines else "无"


def compress_character_state(
    state: Dict[str, str],
    recent_texts: List[str],
    protagonist_names: set,
) -> Dict[str, str]:
    """压缩人物状态（只保留主角和最近出现的人物）"""
    if not state:
        return {}

    compressed = {}

    # 优先保留主角
    for name in protagonist_names:
        if name in state:
            compressed[name] = state[name]

    # 保留最近文本中提到的人物
    recent_text = " ".join(recent_texts[-3:])
    for name, value in state.items():
        if name in compressed:
            continue
        if name in recent_text:
            compressed[name] = value

    # 限制总数量
    if len(compressed) > 10:
        compressed = dict(list(compressed.items())[:10])

    return compressed


# ===================== 文件操作工具 =====================

def get_state_file(book_name: str, stage: str = "A") -> str:
    """获取状态文件路径"""
    safe_name = re.sub(r'[\\/*?:"<>|]', "", book_name)
    return os.path.join(BASE_DIR, f"state_{stage}_{safe_name}.json")


def get_window_file(book_name: str) -> str:
    """获取窗口文件路径"""
    safe_name = re.sub(r'[\\/*?:"<>|]', "", book_name)
    return os.path.join(BASE_DIR, f"state_A_window_{safe_name}.json")


def save_state_atomic(filepath: str, data: Dict[str, Any]):
    """原子化保存状态（先写临时文件再重命名）"""
    temp_path = filepath + ".tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, filepath)
    except Exception as e:
        logger.error(f"❌ 保存状态失败: {e}")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def load_manifest() -> Dict:
    """加载处理进度清单（支持新旧两种格式）"""
    if os.path.exists(MANIFEST_FILE):
        try:
            with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 确保新字段存在
            data.setdefault("completed_books", [])
            data.setdefault("current_processing", None)
            data.setdefault("book_progress", {})
            return data
        except Exception:
            pass

    return {
        "completed_books": [],
        "current_processing": None,
        "book_progress": {},
    }


def save_manifest(data: Dict):
    """保存处理进度清单"""
    try:
        with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"❌ 保存进度清单失败: {e}")


# ===================== 哈希工具 =====================

def generate_id(*parts: str) -> str:
    """生成 MD5 哈希 ID"""
    combined = "|".join(parts)
    return hashlib.md5(combined.encode()).hexdigest()


# ===================== 文本匹配工具 =====================

def find_quote_position_fast(text_scope: str, quote: str) -> int:
    """快速查找引文在文本中的位置"""
    if not quote or not text_scope:
        return -1

    # 精确匹配
    pos = text_scope.find(quote)
    if pos != -1:
        return pos

    # 模糊匹配（去除空白后匹配）
    quote_clean = re.sub(r"\s+", "", quote)
    text_clean = re.sub(r"\s+", "", text_scope)
    pos = text_clean.find(quote_clean)

    if pos != -1:
        # 映射回原始位置
        original_pos = 0
        clean_pos = 0
        for i, char in enumerate(text_scope):
            if not char.isspace():
                if clean_pos == pos:
                    original_pos = i
                    break
                clean_pos += 1
        return original_pos

    return -1


# ===================== 书名处理工具 =====================

def clean_book_name(raw_name: str) -> Tuple[str, str]:
    """
    从文件名中提取纯净书名和后缀标记
    输入: 《老婆孩子热炕头》作者：水千丞[番外]
    输出: ('老婆孩子热炕头', '[番外]')
    """
    # 提取书名号内的内容
    match = re.search(r"《(.*?)》", raw_name)
    if match:
        pure_book_name = match.group(1).strip()
    else:
        # 去掉"作者："及之后的内容
        pure_book_name = re.split(r"作者[：:]|by\s*", raw_name, flags=re.IGNORECASE)[0].strip()
        # 去掉常见后缀
        pure_book_name = re.sub(
            r"\[番外\]|\(番外\)|番外|补车|精校版|未删减", "", pure_book_name
        ).strip()

    # 提取后缀标记
    suffix_match = re.search(r"(\[番外\]|\[补车\]|\[精校\]|\(番外\))", raw_name)
    suffix = suffix_match.group(1) if suffix_match else ""

    return pure_book_name, suffix
