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

# 预编译正则表达式（避免每次调用重新编译）
_AD_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"最新网址：.*?\.com",
    r"手机版阅读网址：.*",
    r"天才一秒记住.*?秒",
    r"本站.*?域名",
    r"www\..*?\.com",
    r"手机阅读.*",
    r"一秒记住.*",
]]
_ANTI_PIRACY_RE = re.compile(
    r"^[^\u4e00-\u9fa5a-zA-Z0-9\s\.,;:!?，。；：！？、\n]{10,}$", re.MULTILINE
)
_NOISE_PATTERNS = [re.compile(p, re.MULTILINE | re.IGNORECASE) for p in [
    r"^[\s ]*(求月票|求订阅|求推荐|求收藏|求打赏|拜求|感谢.*?打赏|感谢.*?万赏).*?$",
    r"^[\s ]*(PS|ps|Ps|pS)[：:].*?$",
    r"^[\s ]*(作者的话|作者说|题外话|碎碎念)[：:].*?$",
    r"^[\s ]*(本章未完|点击下一页继续阅读|最新网址|手机阅读).*?$",
]]
_AUTHOR_NOTE_BLOCK_RE = None  # 不再用正则（O(n²) 问题），改用线性扫描函数
_GARBLED_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")

# 章节标题强特征正则（全文搜索，兼容各种格式）
# 匹配: 第X章/回/节/卷/集/部/篇 + 可选标题, Chapter N
_STRONG_CHAPTER_RE = re.compile(
    r"(?:^|\n+)\s*"
    r"("
    r"(?:第[零一二三四五六七八九十百千万两\d\-]+[章节回卷集部篇][^\n]{0,30})"
    r"|(?:[Cc]hapter\s*\d+[^\n]{0,30})"
    r")"
    r"\s*(?:\n+|$)",
)

# 作者的话块状区域检测关键词
_AUTHOR_NOTE_TRIGGERS_CN = ("作者的话", "作者说", "题外话", "碎碎念")  # 中文触发词（子串匹配）
_PS_TRIGGER_RE = re.compile(r"^(PS|ps|Ps|pS)[\uff1a:\s]|^(PS|ps|Ps|pS)$")  # PS 需要行首 + 冒号/空白/行尾
_CHAPTER_HEADER_RE = re.compile(r"^第[零一二三四五六七八九十百千万两\d]+[章节回]")


def _is_author_note_trigger(stripped: str) -> bool:
    """检查行是否是作者话块的起始行"""
    # 中文触发词：子串匹配
    for trigger in _AUTHOR_NOTE_TRIGGERS_CN:
        if trigger in stripped:
            return True
    # PS 触发：行首 + 后跟冒号/空白/行尾
    if _PS_TRIGGER_RE.match(stripped):
        return True
    return False


def _remove_author_note_blocks(text: str) -> str:
    """
    剔除“作者的话”/PS 块状区域（线性扫描 O(n)）
    从触发关键词开始，到下一个章节标题为止，整块删除。
    """
    lines = text.split("\n")
    result = []
    in_block = False

    for line in lines:
        stripped = line.strip()
        if in_block:
            # 检查是否到达下一个章节标题
            if _CHAPTER_HEADER_RE.match(stripped):
                in_block = False
                result.append(line)
            # 否则跳过该行
        else:
            # 检查是否触发作者话块
            if _is_author_note_trigger(stripped):
                in_block = True
            else:
                result.append(line)

    return "\n".join(result)


def clean_novel_text(text: str) -> str:
    """网文专属文本清洗引擎：剔除广告、防盗章节、作者废话，提纯正文"""
    if not text:
        return ""

    # 1. 统一换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. 移除常见广告/导航文本
    for pattern in _AD_PATTERNS:
        text = pattern.sub("", text)

    # 3. 移除防盗章节特征
    text = _ANTI_PIRACY_RE.sub("", text)

    # 4. 剔除求月票/求订阅/作者的话等单行废话
    for pattern in _NOISE_PATTERNS:
        text = pattern.sub("", text)

    # 5. 剔除“作者的话”块状区域（线性扫描，避免正则 O(n²) 问题）
    text = _remove_author_note_blocks(text)

    # 6. 移除乱码字符
    text = _GARBLED_CHARS_RE.sub("", text)

    # 7. 规范化空白字符
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)

    # 8. 清理行首行尾空白
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    logger.debug(f"clean_novel_text: 全部清理完成, 最终长度 {len(text)}")

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
    章节切分引擎（适配各种格式）
    策略：能识别章节标题就切，识别不了就按固定大小切。
    标题不重要，内容才重要。
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    original_length = len(text.replace("\n", "").replace(" ", ""))

    # === 第一步：用 re.split 在全文中搜索章节标记 ===
    logger.info(f"《{book_name}》 smart_split 步骤1: re.split 开始 (文本 {len(text)} 字符)...")
    parts = _STRONG_CHAPTER_RE.split(text)
    logger.info(f"《{book_name}》 smart_split 步骤1: re.split 完成, 共 {len(parts)} 个片段")

    chapters = []
    # parts[0] 是第一个章节标题之前的内容（序言/前言）
    if parts[0].strip() and len(parts[0].strip()) > 100:
        chapters.append({
            "id": "序章",
            "text": parts[0].strip(),
            "book_name": book_name,
        })

    # parts[1::2] 是章节标题，parts[2::2] 是对应内容
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if content and len(content) > 100:
            chapters.append({
                "id": title if title else f"第{len(chapters) + 1}章",
                "text": content,
                "book_name": book_name,
            })

    # === 第二步：如果没识别到章节或太少，退化为固定大小切分 ===
    if len(chapters) <= 1 and len(text) > max_chunk:
        chapters = []
        pos = 0
        idx = 1
        while pos < len(text):
            chunk = text[pos:pos + max_chunk].strip()
            if chunk and len(chunk) > 50:
                chapters.append({
                    "id": f"分块_{idx}",
                    "text": chunk,
                    "book_name": book_name,
                })
                idx += 1
            pos += max_chunk
        logger.info(
            f"《{book_name}》 未识别到章节标题，全文按 {max_chunk} 字符切分为 {len(chapters)} 块"
        )

    # === 第三步：对过长章节进行二次切分（语义边界 + 滑动窗口） ===
    _long_chapters = [c for c in chapters if len(c["text"]) > max_chunk * 2]
    if _long_chapters:
        logger.info(f"《{book_name}》 smart_split 步骤3: {len(_long_chapters)} 个过长章节需要二次切分 (max_chunk={max_chunk})")
    final_chapters = []
    _step3_max_iterations = len(text) // 100 + 1000  # 安全上限：防止死循环
    for chap in chapters:
        if len(chap["text"]) > max_chunk * 2:
            text_content = chap["text"]
            pos = 0
            sub_index = 1
            _step3_iter = 0

            while pos < len(text_content):
                _step3_iter += 1
                if _step3_iter > _step3_max_iterations:
                    logger.error(f"《{book_name}》 smart_split_chapters 第三步疑似死循环，强制退出！pos={pos}, len={len(text_content)}, max_chunk={max_chunk}")
                    break
                cut_pos = _find_semantic_boundary(text_content, pos, max_chunk)

                chunk_text = text_content[pos:cut_pos].strip()
                if chunk_text and len(chunk_text) > 50:
                    final_chapters.append({
                        "id": f"{chap['id']}_{sub_index}",
                        "text": chunk_text,
                        "book_name": book_name,
                    })
                    sub_index += 1

                if cut_pos >= len(text_content):
                    break

                pos = cut_pos
                if overlap > 0:
                    new_pos = max(pos - overlap, 0)
                    if new_pos <= pos - max_chunk:
                        pos = cut_pos
                    else:
                        pos = new_pos
                    # 防护：剩余文本不足 max_chunk 时取消 overlap，
                    # 避免 pos 回退形成死循环
                    remaining = len(text_content) - pos
                    if remaining <= max_chunk:
                        pos = cut_pos
        else:
            final_chapters.append(chap)

    # === 第四步：字数守恒检查 ===
    if original_length > 0 and final_chapters:
        split_text_combined = "".join(ch["text"] for ch in final_chapters)
        split_length = len(split_text_combined.replace("\n", "").replace(" ", ""))
        _slice_suffix_re = re.compile(r"_\d+$")
        total_overlap = sum(
            overlap for ch in final_chapters
            if _slice_suffix_re.search(ch.get("id", ""))
        )
        net_split_length = split_length - total_overlap
        loss_rate = (original_length - net_split_length) / original_length
        if loss_rate > 0.05:
            logger.warning(
                f"文本丢失警告: 《{book_name}》 丢失了 {loss_rate:.2%} "
                f"(原始: {original_length}字, 切分后: {split_length}字, 重叠: {total_overlap}字)"
            )

    logger.info(f"《{book_name}》 切分结果: {len(final_chapters)} 个章节/块")
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
        # 非字典输入保存为 _raw，防止 LLM 返回异常格式时丢失数据
        return {"_raw": str(state)} if state else {}

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
    """将人物状态压缩为文本（用于 Prompt），能解析 JSON 字符串化的状态值"""
    if not state_dict or (len(state_dict) == 1 and "_raw" in state_dict):
        return "暂无明确人物状态"

    text_parts = []
    for name, state in state_dict.items():
        if name in ("_raw", "旁白"):
            continue

        state_str = ""
        if isinstance(state, dict):
            # 直接是字典，拼接值
            state_str = "/".join([str(v) for v in state.values() if v])
        elif isinstance(state, str) and state.strip().startswith("{"):
            # JSON 字符串化的字典，解析后再拼接
            try:
                parsed = json.loads(state)
                if isinstance(parsed, dict):
                    state_str = "/".join([str(v) for v in parsed.values() if v])
                elif isinstance(parsed, list):
                    state_str = "/".join([str(v) for v in parsed if v])
                else:
                    state_str = str(state)
            except (json.JSONDecodeError, TypeError):
                state_str = str(state)
        elif isinstance(state, str) and state.strip().startswith("["):
            # JSON 字符串化的列表
            try:
                parsed = json.loads(state)
                state_str = (
                    "/".join([str(v) for v in parsed if v])
                    if isinstance(parsed, list)
                    else str(state)
                )
            except (json.JSONDecodeError, TypeError):
                state_str = str(state)
        else:
            state_str = str(state)

        if state_str:
            text_parts.append(f"{name}:{state_str}")

    return "; ".join(text_parts) if text_parts else "暂无明确人物状态"


def compress_character_state(
    state: Dict[str, str],
    recent_texts: List[str],
    protagonist_names: set,
) -> Dict[str, str]:
    """压缩人物状态（只保留主角和最近出现的人物）"""
    if not state:
        return {}

    recent_text = " ".join(recent_texts[-3:])
    compressed = {}

    # 优先保留主角
    for name in protagonist_names:
        if name in state:
            compressed[name] = state[name]

    # 保留最近文本中提到的人物：用已知人物名在文本中查找（避免正则匹配所有 2-4 字词）
    for name, value in state.items():
        if name in compressed:
            continue
        if name in ("_raw", "旁白"):
            continue
        # 直接检查人物名是否出现在最近文本中
        if name in recent_text:
            compressed[name] = value

    # 如果完全没有人匹配，加个旁白说明
    if not compressed:
        compressed["旁白"] = "当前无核心人物出场"

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
    """
    原子化保存状态（先写临时文件再重命名）
    增强：os.fsync 强制刷盘 + PermissionError 指数退避重试 + shutil.move 兆底
    """
    import shutil
    import time

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    temp_path = filepath + ".tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())  # 强制刷盘，防止断电数据丢失
    except Exception as e:
        logger.error(f"\u2764 保存状态写入失败: {e}")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        return

    # 重命名：指数退避重试（Windows 文件锁冲突常见）
    for i in range(10):
        try:
            os.replace(temp_path, filepath)
            return
        except PermissionError:
            time.sleep(min(0.1 * (2 ** i), 5))  # 最多等 5 秒，总计最多 ~10 秒
        except OSError:
            # shutil.move 兆底
            try:
                shutil.move(temp_path, filepath)
                return
            except Exception:
                break

    # 最终兆底
    try:
        shutil.move(temp_path, filepath + ".fallback")
        logger.warning(f"保存状态失败，已写入兆底文件: {filepath}.fallback")
    except Exception as e:
        logger.error(f"\u2764 保存状态最终失败: {e}")


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
    """保存处理进度清单（复用原子写入逻辑，防止断电损坏）"""
    save_state_atomic(MANIFEST_FILE, data)


# ===================== 哈希工具 =====================

def generate_id(*parts: str) -> str:
    """生成 MD5 哈希 ID"""
    combined = "|".join(parts)
    return hashlib.md5(combined.encode()).hexdigest()


# ===================== 文本匹配工具 =====================

def find_quote_position_fast(text_scope: str, quote: str) -> int:
    """快速查找引文在文本中的位置（精确匹配 → 去空白匹配 → 模糊匹配三级降级）"""
    if not quote or not text_scope:
        return -1

    # 第一级：精确匹配
    pos = text_scope.find(quote)
    if pos != -1:
        return pos

    # 第二级：去除空白后精确匹配
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

    # 第三级：模糊匹配（用 thefuzz 库，处理 LLM 返回的引文与原文有微小差异的情况）
    try:
        from thefuzz import fuzz
        from config.settings import MATCH_THRESHOLD

        # 按标点分句，然后用滑动窗口做模糊匹配
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
    except ImportError:
        pass  # thefuzz 未安装，跳过模糊匹配

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
