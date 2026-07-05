"""
小说知识库构建系统 - 主入口（三层弹性架构版）

架构设计：
  Layer 1 (基础层): Stage A — 剧情摘要与人物状态追踪，必须最先完成
  Layer 2 (分析层): Stage B/C/D/I — 依赖 Layer 1，彼此完全独立，并行执行
  Layer 3 (综合层): Stage E/F/G/H — 依赖 Layer 1+2，彼此可并行

断点续跑粒度：Layer 级别 + Stage 级别双重保障
"""

import os
import sys
import glob
import re
import logging
import argparse
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import BASE_DIR
from core.ollama_client import get_ollama_client
from core.db import get_db_manager
from core.chroma_client import get_chroma_manager
from core.graph import get_graph_manager
from core.utils import (
    clean_novel_text,
    smart_split_chapters,
    clean_book_name,
    load_manifest,
    save_manifest,
    get_state_file,
    get_window_file,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(BASE_DIR, "novel_analyzer.log"),
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)


# ===================== Layer 定义 =====================

LAYER_STAGES = {
    1: ["A"],
    2: ["B", "C", "D", "I"],
    3: ["E", "F", "G", "H", "O"],  # Stage O: 事件因果图谱
}

# manifest 字典多线程写入保护锁
_manifest_lock = threading.Lock()


def is_stage_complete(manifest: Dict, book_name: str, stage: str) -> bool:
    """检查某个 Stage 是否已完成"""
    progress = manifest.get("book_progress", {}).get(book_name, {})
    return progress.get("stage_status", {}).get(stage) == "complete"


def is_layer_complete(manifest: Dict, book_name: str, layer: int) -> bool:
    """检查某个 Layer 是否已全部完成"""
    progress = manifest.get("book_progress", {}).get(book_name, {})
    stage_status = progress.get("stage_status", {})
    for s in LAYER_STAGES[layer]:
        if stage_status.get(s) != "complete":
            return False
    return True


def mark_stage_complete(manifest: Dict, book_name: str, stage: str):
    """标记某个 Stage 完成（线程安全）"""
    with _manifest_lock:
        if "book_progress" not in manifest:
            manifest["book_progress"] = {}
        if book_name not in manifest["book_progress"]:
            manifest["book_progress"][book_name] = {"stage_status": {}}
        if "stage_status" not in manifest["book_progress"][book_name]:
            manifest["book_progress"][book_name]["stage_status"] = {}
        manifest["book_progress"][book_name]["stage_status"][stage] = "complete"
        save_manifest(manifest)


def mark_stage_failed(manifest: Dict, book_name: str, stage: str, error: str):
    """标记某个 Stage 失败（线程安全）"""
    with _manifest_lock:
        if "book_progress" not in manifest:
            manifest["book_progress"] = {}
        if book_name not in manifest["book_progress"]:
            manifest["book_progress"][book_name] = {"stage_status": {}}
        if "stage_status" not in manifest["book_progress"][book_name]:
            manifest["book_progress"][book_name]["stage_status"] = {}
        manifest["book_progress"][book_name]["stage_status"][
            stage
        ] = f"failed:{error[:100]}"
        save_manifest(manifest)


def mark_stage_skipped(manifest: Dict, book_name: str, stage: str, reason: str):
    """
    标记某个 Stage 被跳过（前置依赖未完成）。
    下次运行时 is_stage_complete 返回 False，会自动重试。
    """
    with _manifest_lock:
        if "book_progress" not in manifest:
            manifest["book_progress"] = {}
        if book_name not in manifest["book_progress"]:
            manifest["book_progress"][book_name] = {"stage_status": {}}
        if "stage_status" not in manifest["book_progress"][book_name]:
            manifest["book_progress"][book_name]["stage_status"] = {}
        manifest["book_progress"][book_name]["stage_status"][
            stage
        ] = f"skipped:{reason}"
        save_manifest(manifest)


def print_progress_matrix(manifest: Dict, novel_list: List[Dict]):
    """打印所有书籍的处理状态矩阵"""
    print("\n" + "=" * 70)
    print("处理状态矩阵 (L1=基础层  L2=分析层  L3=综合层)")
    print("=" * 70)
    header = f"{'书名':<20} | {'A':^3} | {'B':^3} {'C':^3} {'D':^3} {'I':^3} | {'E':^3} {'F':^3} {'G':^3} {'H':^3} {'O':^3} | 状态"
    print(header)
    print("-" * 70)

    for book_info in novel_list:
        book_name = book_info["book_name"]
        progress = manifest.get("book_progress", {}).get(book_name, {})
        stage_status = progress.get("stage_status", {})

        # 截断书名显示
        display_name = book_name[:18] if len(book_name) > 18 else book_name

        def status_icon(s):
            st = stage_status.get(s, "")
            if st == "complete":
                return "OK"
            elif st.startswith("failed"):
                return "XX"
            elif st.startswith("skipped"):
                return "SK"
            else:
                return "--"

        icons = [status_icon(s) for s in ["A", "B", "C", "D", "I", "E", "F", "G", "H", "O"]]

        # 判断整体状态
        if all(
            stage_status.get(s) == "complete"
            for s in ["A", "B", "C", "D", "I", "E", "F", "G", "H", "O"]
        ):
            overall = "DONE"
        elif any(
            stage_status.get(s, "").startswith("failed")
            for s in ["A", "B", "C", "D", "I", "E", "F", "G", "H", "O"]
        ):
            overall = "ERR"
        elif any(
            stage_status.get(s, "").startswith("skipped")
            for s in ["A", "B", "C", "D", "I", "E", "F", "G", "H", "O"]
        ):
            overall = "SKIP"
        elif any(
            stage_status.get(s) == "complete"
            for s in ["A", "B", "C", "D", "I", "E", "F", "G", "H", "O"]
        ):
            overall = "WIP"
        else:
            overall = "NEW"

        line = f"{display_name:<20} | {icons[0]:^3} | {icons[1]:^3} {icons[2]:^3} {icons[3]:^3} {icons[4]:^3} | {icons[5]:^3} {icons[6]:^3} {icons[7]:^3} {icons[8]:^3} {icons[9]:^3} | {overall}"
        print(line)

    print("=" * 70 + "\n")


# ===================== 扫描与预处理 =====================


def scan_novel_library(root_dir: str) -> List[Dict[str, Any]]:
    """
    扫描小说库目录
    支持文件名格式：《书名》作者：作者名 分类：分类 标签：标签1、标签2
    兼容旧格式：从目录结构提取分类
    """
    print(f"正在扫描小说库：{root_dir}")
    all_txt = glob.glob(os.path.join(root_dir, "**", "*.txt"), recursive=True)

    book_list = []
    for path in all_txt:
        raw_file_name = os.path.splitext(os.path.basename(path))[0]
        pure_book_name, suffix = clean_book_name(raw_file_name)

        # 提取作者名（到“分类”或“标签”为止）
        author_match = re.search(
            r"作者[：:]\s*(.+?)(?=\s*分类[：:]|\s*标签[：:]|\s*$)",
            raw_file_name,
        )
        if not author_match:
            author_match = re.search(r"by\s+(.+?)(?=\s*分类[：:]|\s*标签[：:]|\s*$)", raw_file_name, re.IGNORECASE)
        author_name = author_match.group(1).strip() if author_match else "未知作者"

        # 提取分类（优先从文件名，降级到目录结构）
        category_match = re.search(r"分类[：:]\s*(.+?)(?=\s*标签[：:]|\s*$)", raw_file_name)
        if category_match:
            category = category_match.group(1).strip()
        else:
            # 降级：从目录结构提取
            rel_path = os.path.relpath(path, root_dir)
            parts = rel_path.split(os.sep)
            if len(parts) >= 3:
                category = parts[1]
                category = re.sub(r"[\(（].*?[\)）]", "", category).replace("合集", "").strip()
            elif len(parts) == 2:
                category = parts[0]
            else:
                category = "未分类"

        # 提取标签
        tags_match = re.search(r"标签[：:]\s*(.+?)$", raw_file_name)
        tags = [t.strip() for t in tags_match.group(1).split("、") if t.strip()] if tags_match else []

        db_book_name = f"{pure_book_name}{suffix}" if suffix else pure_book_name

        book_list.append(
            {
                "book_name": db_book_name,
                "pure_name": pure_book_name,
                "author": author_name,
                "category": category,
                "tags": tags,
                "all_files": [path],
            }
        )

    print(f"扫描完成，共发现 {len(book_list)} 本独立小说。")
    return book_list


def merge_txt_files(file_list: List[str], output_path: str) -> str:
    """合并多个 TXT 文件"""
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


# ===================== Layer 执行函数 =====================


def run_layer_1(book_name: str, category: str, chapters: List[Dict], manifest: Dict):
    """
    Layer 1 (基础层): Stage A — 剧情摘要与人物状态追踪
    必须最先完成，其结果是下游所有 Stage 的数据基础
    """
    from stages.stage_a import StageA

    if is_stage_complete(manifest, book_name, "A"):
        print(f"  [Layer 1] Stage A 已完成，跳过")
        # 从缓存恢复结果
        stage_a = StageA(book_name, category)
        cache = stage_a.load_cache()
        if cache and cache.get("stage") == "A":
            cached_data = cache.get("data", [])
            for i, item in enumerate(cached_data):
                if i < len(chapters):
                    chapters[i]["summary"] = item.get("summary", "")
                    chapters[i]["character_state"] = item.get("character_state", {})
                    chapters[i]["information_flow"] = item.get("information_flow", {})
                    chapters[i]["emotion_arc"] = item.get("emotion_arc", "")
                    chapters[i]["time_progression"] = item.get("time_progression", "")
            inferred_cat = cache.get("inferred_category", category)
            protagonist_names = set(cache.get("protagonist_names", []))
            return chapters, inferred_cat, protagonist_names
        # 缓存不存在则需要重新跑
        print(f"  [Layer 1] 缓存不存在，重新执行 Stage A")

    print(f"  [Layer 1] 执行 Stage A（剧情摘要与人物状态追踪）...")
    stage_a = StageA(book_name, category)
    stage_a_res, inferred_cat, protagonist_names = stage_a.run(chapters)
    stats_a = stage_a.insert((stage_a_res, inferred_cat, protagonist_names))
    logger.info(f"Stage A 入库完成: {stats_a}")

    # 质量自检
    stage_a.run_quality_check(stage_a_res)

    mark_stage_complete(manifest, book_name, "A")
    return stage_a_res, inferred_cat, protagonist_names


def run_layer_2(
    book_name: str, category: str, author: str, stage_a_res: List[Dict], manifest: Dict
):
    """
    Layer 2 (分析层): Stage B/C/D/I — 分组串行
    Group 1: I(统计) + B(7b) + C(7b) 并行
    Group 2: D(14b) 单独运行（避免 7b/14b 并发导致模型切换开销和超时）
    """
    from stages.stage_b import StageB
    from stages.stage_c import StageC
    from stages.stage_d import StageD
    from stages.stage_i import StageI

    tasks = {}

    # Stage I 不依赖 stage_a_res，直接用原始 chapters
    if not is_stage_complete(manifest, book_name, "I"):
        tasks["I"] = ("stage_i_raw", StageI(book_name, category))

    if not is_stage_complete(manifest, book_name, "B"):
        tasks["B"] = ("stage_a_res", StageB(book_name, category))
    if not is_stage_complete(manifest, book_name, "C"):
        tasks["C"] = ("stage_a_res", StageC(book_name, category))
    if not is_stage_complete(manifest, book_name, "D"):
        tasks["D"] = ("stage_a_res", StageD(book_name, category, author))

    if not tasks:
        print(f"  [Layer 2] 所有 Stage 已完成，跳过")
        return

    def run_stage(stage_key, stage_obj, input_data):
        """在线程中运行单个 Stage"""
        return stage_key, stage_obj.run(stage_a_res)

    def _execute_group(group_tasks):
        """执行一组 Stage 并入库"""
        if not group_tasks:
            return
        with ThreadPoolExecutor(max_workers=len(group_tasks)) as executor:
            futures = {}
            for stage_key, (input_type, stage_obj) in group_tasks.items():
                future = executor.submit(run_stage, stage_key, stage_obj, input_type)
                futures[future] = stage_key

            for future in as_completed(futures):
                stage_key = futures[future]
                try:
                    key, result = future.result()
                    stage_obj = group_tasks[key][1]
                    stats = stage_obj.insert(result)
                    logger.info(f"Stage {key} 入库完成: {stats}")
                    stage_obj.run_quality_check(result)
                    mark_stage_complete(manifest, book_name, key)
                except Exception as e:
                    logger.error(f"Stage {stage_key} 执行失败: {e}")
                    mark_stage_failed(manifest, book_name, stage_key, str(e))

    # Group 1: I(统计) + B(7b) + C(7b) — 并行，无模型冲突
    group1 = {k: v for k, v in tasks.items() if k in ("I", "B", "C")}
    group2 = {k: v for k, v in tasks.items() if k == "D"}

    if group1:
        print(f"  [Layer 2] Group1 (I+B+C, 7b/统计): {list(group1.keys())}")
        _execute_group(group1)
    if group2:
        print(f"  [Layer 2] Group2 (D, 14b): {list(group2.keys())}")
        _execute_group(group2)


def run_layer_3(book_name: str, category: str, stage_a_res: List[Dict], manifest: Dict):
    """
    Layer 3 (综合层): Stage E/F/G/H/O — 分组串行
    Group 1: E(7b) 单独运行（H 依赖 E）
    Group 2: F(14b)+G(14b)+O(14b)+H(14b) E完成后并行，避免 7b/14b 并发
    """
    from stages.stage_e import StageE
    from stages.stage_f import StageF
    from stages.stage_g import StageG
    from stages.stage_h import StageH
    from stages.stage_o import StageO

    tasks = {}

    if not is_stage_complete(manifest, book_name, "E"):
        tasks["E"] = StageE(book_name, category)
    if not is_stage_complete(manifest, book_name, "F"):
        tasks["F"] = StageF(book_name, category)
    if not is_stage_complete(manifest, book_name, "G"):
        tasks["G"] = StageG(book_name, category)
    if not is_stage_complete(manifest, book_name, "H"):
        tasks["H"] = StageH(book_name, category)
    if not is_stage_complete(manifest, book_name, "O"):
        tasks["O"] = StageO(book_name, category)

    if not tasks:
        print(f"  [Layer 3] 所有 Stage 已完成，跳过")
        return

    print(f"  [Layer 3] 并行执行: {list(tasks.keys())}")

    # Stage H 需要 Stage E 的结果作为额外输入，需要特殊处理
    # 先检查 E 是否在之前已完成
    stage_e_done = is_stage_complete(manifest, book_name, "E")
    stage_e_res = None
    if stage_e_done:
        # 从数据库恢复 Stage E 结果
        db = get_db_manager()
        cursor = db.connect().cursor()
        cursor.execute("SELECT * FROM macro_outlines WHERE book_name = ?", (book_name,))
        rows = cursor.fetchall()
        if rows:
            stage_e_res = {
                "macro_outlines": [
                    dict(
                        zip(
                            [
                                "id",
                                "book_name",
                                "category",
                                "volume_index",
                                "chapter_range",
                                "theme",
                                "conflict",
                                "beats_json",
                                "arc",
                            ],
                            row,
                        )
                    )
                    for row in rows
                ]
            }

    def run_stage_e(stage_obj):
        result = stage_obj.run(stage_a_res)
        return "E", stage_obj, result

    def run_stage_h(stage_obj, e_res):
        result = stage_obj.run(stage_a_res, e_res or {})
        return "H", stage_obj, result

    def run_stage_generic(stage_key, stage_obj):
        result = stage_obj.run(stage_a_res)
        return stage_key, stage_obj, result

    def _insert_and_check(stage_key, stage_obj, result):
        stats = stage_obj.insert(result)
        logger.info(f"Stage {stage_key} 入库完成: {stats}")
        stage_obj.run_quality_check(result)
        mark_stage_complete(manifest, book_name, stage_key)
        return stats

    # Group 1: E(7b) 单独运行
    e_task = tasks.pop("E", None)
    stage_e_result_local = None
    if e_task:
        try:
            result_key, stage_obj, result = run_stage_e(e_task)
            stage_e_result_local = result
            _insert_and_check(result_key, stage_obj, result)
        except Exception as e:
            logger.error(f"Stage E 执行失败: {e}")
            mark_stage_failed(manifest, book_name, "E", str(e))

    # Group 2: F(14b)+G(14b)+O(14b) 并行，H(14b) 依赖 E 结果最后执行
    group2 = {k: v for k, v in tasks.items() if k in ("F", "G", "O")}
    h_obj_final = tasks.get("H")
    
    if group2:
        with ThreadPoolExecutor(max_workers=len(group2)) as executor:
            futures = {}
            for key, stage_obj in group2.items():
                future = executor.submit(run_stage_generic, key, stage_obj)
                futures[future] = key
            for future in as_completed(futures):
                key = futures[future]
                try:
                    result_key, stage_obj, result = future.result()
                    _insert_and_check(result_key, stage_obj, result)
                except Exception as e:
                    logger.error(f"Stage {key} 执行失败: {e}")
                    mark_stage_failed(manifest, book_name, key, str(e))

    if h_obj_final is not None:
        if not is_stage_complete(manifest, book_name, "E"):
            logger.warning("Stage E 未完成，跳过 Stage H")
            mark_stage_skipped(manifest, book_name, "H", "dependency_E_not_complete")
        else:
            e_res_for_h = stage_e_result_local or stage_e_res or {}
            try:
                h_key, h_obj, h_result = run_stage_h(h_obj_final, e_res_for_h)
                _insert_and_check(h_key, h_obj, h_result)
            except Exception as e:
                logger.error(f"Stage H 执行失败: {e}")
                mark_stage_failed(manifest, book_name, "H", str(e))


# ===================== 后处理 =====================


def generate_book_style_summary(book_name: str, category: str, manifest: Dict):
    """生成书籍风格概述并更新 book_metadata"""
    db = get_db_manager()
    cursor = db.connect().cursor()

    try:
        style_info = []

        # 从 author_fingerprints 表取数据
        cursor.execute(
            "SELECT verbs, adjectives, imagery, narrative_perspective, sentence_rhythm FROM author_fingerprints WHERE book_name = ? LIMIT 1",
            (book_name,),
        )
        fp_row = cursor.fetchone()
        if fp_row:
            style_info.extend(
                [
                    f"常用动词：{fp_row[0]}",
                    f"常用形容词：{fp_row[1]}",
                    f"意象偏好：{fp_row[2]}",
                    f"叙事视角：{fp_row[3]}",
                    f"句式节奏：{fp_row[4]}",
                ]
            )

        # 从 book_structure 表取数据
        cursor.execute(
            "SELECT structure_type, surface_theme, deep_theme FROM book_structure WHERE book_name = ? LIMIT 1",
            (book_name,),
        )
        bs_row = cursor.fetchone()
        if bs_row:
            style_info.extend(
                [
                    f"结构类型：{bs_row[0]}",
                    f"表层主题：{bs_row[1]}",
                    f"深层主题：{bs_row[2]}",
                ]
            )

        # 从 climax_point_distribution 表取数据
        cursor.execute(
            "SELECT rhythm_pattern FROM climax_point_distribution WHERE book_name = ? LIMIT 1",
            (book_name,),
        )
        cpd_row = cursor.fetchone()
        if cpd_row:
            style_info.append(f"节奏模式：{cpd_row[0]}")

        author_desc = " | ".join(
            [s for s in style_info if s and "：" in s and s.split("：", 1)[1]]
        )

        if author_desc:
            cursor.execute(
                "UPDATE book_metadata SET description = ? WHERE book_name = ?",
                (author_desc, book_name),
            )
            db.commit()
            logger.info(f"书籍风格概述已生成：{len(author_desc)} 字符")

    except Exception as e:
        logger.warning(f"生成书籍风格概述失败: {e}")


def finalize_book(book_name: str, manifest: Dict):
    """完成一本书的处理：保存图谱、清理临时文件、标记完成"""
    # 保存知识图谱
    graph_manager = get_graph_manager()
    graph_manager.save()

    # 标记为完成
    if book_name not in manifest["completed_books"]:
        manifest["completed_books"].append(book_name)
    manifest["current_processing"] = None
    save_manifest(manifest)

    # 清理临时状态文件
    for stage in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "O"]:
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

    print(f"[完成] 《{book_name}》 知识库构建完成！")
    logger.info(f"《{book_name}》 处理完成")


# ===================== 主处理函数 =====================


def process_single_book(book_info: Dict, manifest: Dict, start_from_layer: int = 1):
    """处理单本小说 — 三层弹性架构"""
    from core.ollama_client import ollama_chat, safe_parse_json
    from core.utils import generate_id

    book_name = book_info["book_name"]
    author = book_info.get("author", "未知作者")
    category = book_info.get("category", "未分类")
    filename_tags = book_info.get("tags", [])

    # 合并文件
    merge_path = os.path.join(BASE_DIR, f"temp_{book_name}.txt")
    text_path = merge_txt_files(book_info["all_files"], merge_path)

    try:
        # 读取文本（编码自适应：charset-normalizer + BOM 快路径 + 多编码降级）
        with open(text_path, "rb") as f:
            raw_bytes = f.read()

        raw_text = ""
        detected_encoding = None

        # 快路径：根据 BOM 字节头直接判断
        if raw_bytes[:3] == b"\xef\xbb\xbf":
            raw_text = raw_bytes[3:].decode("utf-8", errors="replace")
            detected_encoding = "utf-8-sig"
        elif raw_bytes[:2] in (b"\xff\xfe", b"\xfe\xff"):
            raw_text = raw_bytes.decode("utf-16", errors="replace")
            detected_encoding = "utf-16"
        else:
            # 无 BOM：用 charset-normalizer 智能检测编码
            try:
                from charset_normalizer import from_bytes
                detection = from_bytes(raw_bytes)
                best = detection.best()
                if best is not None:
                    raw_text = str(best)
                    detected_encoding = best.encoding
                    logger.info(
                        f"《{book_name}》 charset-normalizer 检测编码: {detected_encoding} "
                        f"(置信度: {best.language or 'N/A'}, {len(raw_bytes)} 字节)"
                    )
            except ImportError:
                logger.warning("charset-normalizer 未安装，使用降级编码检测（pip install charset-normalizer）")
            except Exception as e:
                logger.warning(f"charset-normalizer 检测失败: {e}，使用降级编码检测")

            # charset-normalizer 失败或未安装：多编码降级尝试
            if not raw_text:
                for encoding in ("utf-8", "gbk", "gb2312", "utf-16-le", "big5"):
                    try:
                        candidate = raw_bytes.decode(encoding)
                        replacement_count = candidate.count("\ufffd")
                        control_chars = sum(
                            1 for ch in candidate[:5000]
                            if ord(ch) < 32 and ch not in ("\n", "\r", "\t")
                        )
                        if replacement_count > 10 or control_chars > 50:
                            continue
                        raw_text = candidate
                        detected_encoding = encoding
                        break
                    except (UnicodeDecodeError, ValueError):
                        continue

        if not raw_text:
            raw_text = raw_bytes.decode("latin-1", errors="ignore")
            detected_encoding = "latin-1"
            logger.warning(f"《{book_name}》 所有编码均失败，使用 latin-1 降级读取（可能有乱码）")
        elif detected_encoding and detected_encoding != "latin-1":
            logger.info(f"《{book_name}》 检测编码: {detected_encoding} ({len(raw_bytes)} 字节)")

        # 清理 BOM 残留和零宽字符
        raw_text = raw_text.lstrip("\ufeff").replace("\ufeff", "")

        # 清洗并切分章节
        logger.debug(f"开始 clean_novel_text (文本长度: {len(raw_text)} 字符)...")
        raw_text = clean_novel_text(raw_text)
        logger.debug(f"clean_novel_text 完成 (清洗后长度: {len(raw_text)} 字符)")

        if len(raw_text) < 500:
            logger.warning(f"\u300a{book_name}\u300b 清洗后正文不足500字，可能是防盗章节或空文件")

        logger.debug(f"开始 smart_split_chapters...")
        chapters = smart_split_chapters(raw_text, book_name)
        total_chapters = len(chapters)
        total_words = len(raw_text)
        logger.info(f"《{book_name}》 章节切分完成: {total_chapters} 章 (总字数: {total_words})")

        # 生成类型标签：优先使用文件名中的标签，降级到 LLM 生成
        if filename_tags:
            genre_tags = ",".join(filename_tags)
            logger.info(f"使用文件名标签: {genre_tags}")
        else:
            genre_tags = ""
            try:
                sample_text = raw_text[:2000] if len(raw_text) > 2000 else raw_text
                tag_prompt = f"""根据以下小说信息，生成5-8个类型标签（用逗号分隔）。

书名：{book_name}
作者：{author}
分类：{category}
开头内容：
{sample_text}

请输出纯 JSON 格式：
{{"tags": "标签1,标签2,标签3,..."}}
(要求：标签应包含题材类型、风格特点、目标读者等维度，如：玄幻,升级流,热血,男频)"""

                print(f"  [INFO] 正在通过 LLM 生成类型标签 (首次调用需加载模型，约30-90秒)...", flush=True)
                tag_resp = ollama_chat(tag_prompt, 0.3, "A")
                logger.debug(f"ollama_chat 完成 (响应长度: {len(tag_resp)} 字符)")
                tag_data = safe_parse_json(tag_resp)
                if tag_data and "tags" in tag_data:
                    genre_tags = tag_data["tags"]
            except Exception as e:
                logger.warning(f"生成类型标签失败: {e}")
                genre_tags = category

        # 写入 book_metadata 表
        db = get_db_manager()
        metadata_id = generate_id(book_name, "metadata")
        cursor = db.connect().cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO book_metadata VALUES (?,?,?,?,?,?,?,?,?)",
            (
                metadata_id,
                book_name,
                author,
                category,
                genre_tags,
                total_chapters,
                total_words,
                "",
                datetime.now().isoformat(),
            ),
        )
        db.commit()
        logger.info(f"book_metadata 入库完成: {book_name} ({category}, {total_chapters}章, {total_words}字)")

        print(
            f"\n{'='*20} 开始处理：《{book_name}》 (总章数:{total_chapters}) {'='*20}"
        )
        manifest["current_processing"] = book_name
        save_manifest(manifest)

        # === Layer 1 (基础层) ===
        if start_from_layer <= 1 and not is_layer_complete(manifest, book_name, 1):
            stage_a_res, inferred_cat, protagonist_names = run_layer_1(
                book_name, category, chapters, manifest
            )
            if inferred_cat and inferred_cat != category:
                category = inferred_cat
                book_info["category"] = category
        else:
            # 从缓存恢复 Stage A 结果
            stage_a_res, inferred_cat, protagonist_names = run_layer_1(
                book_name, category, chapters, manifest
            )
            if inferred_cat and inferred_cat != category:
                category = inferred_cat

        # === Layer 2 (分析层) ===
        if start_from_layer <= 2 and not is_layer_complete(manifest, book_name, 2):
            run_layer_2(book_name, category, author, stage_a_res, manifest)
        elif is_layer_complete(manifest, book_name, 2):
            print(f"  [Layer 2] 已完成，跳过")

        # === Layer 3 (综合层) ===
        if start_from_layer <= 3 and not is_layer_complete(manifest, book_name, 3):
            run_layer_3(book_name, category, stage_a_res, manifest)
        elif is_layer_complete(manifest, book_name, 3):
            print(f"  [Layer 3] 已完成，跳过")

        # 后处理：生成风格概述
        if is_layer_complete(manifest, book_name, 3):
            generate_book_style_summary(book_name, category, manifest)
            finalize_book(book_name, manifest)

    finally:
        if os.path.exists(merge_path) and text_path == merge_path:
            try:
                os.remove(merge_path)
            except Exception:
                pass


# ===================== 主入口 =====================


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(description="小说知识库构建系统（三层弹性架构版）")
    parser.add_argument(
        "--novels-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "novels"),
        help="小说库根目录路径",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="从指定 Layer 开始处理（1=基础层, 2=分析层, 3=综合层），已完成的 Layer 会自动跳过",
    )
    parser.add_argument(
        "--reset-chroma",
        action="store_true",
        help="清空并重建 ChromaDB 集合（切换 embedding 模型时必须使用此参数）",
    )
    parser.add_argument(
        "--only",
        type=str,
        help="只处理指定书名的小说（支持部分匹配）",
    )
    args = parser.parse_args()

    NOVELS_ROOT_DIR = args.novels_dir

    # 健康检查
    client = get_ollama_client()
    if not client.check_health():
        print(
            "Ollama 服务检查失败，请确保 Ollama 已启动并安装了所需模型（qwen2.5:7b, qwen14b:latest）"
        )
        return

    # 初始化数据库
    db = get_db_manager()
    db.init_tables()

    # 初始化 ChromaDB（支持重置）
    chroma = get_chroma_manager()
    chroma.init_collections(reset=args.reset_chroma)

    if args.reset_chroma:
        print(
            "WARNING: ChromaDB 集合已重置。由于旧向量数据与新 embedding 模型不兼容，需要重新处理所有小说。"
        )
        print("  正在清除所有 book_progress 记录，以便全量重建...")
        manifest = load_manifest()
        if "book_progress" in manifest:
            manifest["book_progress"] = {}
        manifest["completed_books"] = []
        save_manifest(manifest)
        print("  进度已重置，准备全量重建。")

    # 加载进度清单
    manifest = load_manifest()

    # 扫描小说库
    novel_list = scan_novel_library(NOVELS_ROOT_DIR)
    if not novel_list:
        print("未找到任何 TXT 小说。")
        return

    # 过滤指定书名
    if args.only:
        novel_list = [b for b in novel_list if args.only in b["book_name"]]
        if not novel_list:
            print(f"未找到匹配 '{args.only}' 的小说。")
            return

    # 打印进度矩阵
    print_progress_matrix(manifest, novel_list)

    # 筛选待处理书籍（未完成所有 Layer 的书）
    pending_books = [
        b for b in novel_list if b["book_name"] not in manifest["completed_books"]
    ]

    print(
        f"调度清单：共扫描 {len(novel_list)} 本，已完工 {len(manifest.get('completed_books', []))} 本，待处理 {len(pending_books)} 本。"
    )

    if not pending_books:
        print("所有小说均已处理完成！")
        return

    # 逐本处理
    for idx, book_info in enumerate(pending_books):
        book_name = book_info["book_name"]
        progress = manifest.get("book_progress", {}).get(book_name, {})
        stage_status = progress.get("stage_status", {})
        completed_stages = [s for s, v in stage_status.items() if v == "complete"]
        failed_stages = [
            s for s, v in stage_status.items() if str(v).startswith("failed")
        ]

        status_info = f"已完成: {completed_stages}" if completed_stages else "新书"
        if failed_stages:
            status_info += f" | 失败: {failed_stages}"

        print(
            f"\n[{idx+1}/{len(pending_books)}] 《{book_name}》 [{book_info['category']}] ({status_info})"
        )

        try:
            process_single_book(book_info, manifest, start_from_layer=args.start_from)
        except Exception as e:
            import traceback

            error_msg = traceback.format_exc()
            print(f"处理《{book_name}》时发生致命错误：\n{error_msg}")
            with open(
                os.path.join(BASE_DIR, "fatal_errors.log"), "a", encoding="utf-8"
            ) as f:
                f.write(f"=== {book_name} ===\n{error_msg}\n")

    # 最终进度矩阵
    print_progress_matrix(load_manifest(), novel_list)

    # 提示是否需要运行高级 Stage
    total_books = len(manifest.get("completed_books", []))
    if total_books >= 2:
        print(f"\n已有 {total_books} 本书完成基础构建，可以运行高级功能：")
        print(
            f"  python run_advanced_stages.py               # 执行全部高级功能 (L/M/N)"
        )
        print(f"  python run_advanced_stages.py --only L      # 只执行跨书对比分析")
        print(
            f"  python run_advanced_stages.py --incremental # 增量模式：只处理新增书籍"
        )

    # 截断验证：确认是否有正文损失
    from core.ollama_client import get_truncation_count
    truncations = get_truncation_count()
    if truncations == 0:
        print("\n✅ 截断验证通过：0 次正文截断，所有章节完整喂给 LLM。")
    else:
        print(f"\n❌ 截断验证失败：{truncations} 次正文截断！存在正文损失！")

    # 质量抽查提醒
    total_completed = len(manifest.get("completed_books", []))
    if total_completed > 0 and total_completed % 5 == 0:
        print(f"\n{'='*60}")
        print(f"📋 已完成 {total_completed} 本书，建议进行 Stage F 质量抽查：")
        print(f"   1. 从每本书的 dialogue_samples / climax_excerpts / memorable_quotes")
        print(f"      中各取 3 条 writing_quality=9-10 的样本，人工判断是否真的优秀")
        print(f"   2. 如果 9-10 分样本里混了平庸内容 → 虚高，需调整 Stage F prompt")
        print(f"   3. 如果 9-10 分确实优秀 → 评分系统有效，继续积累基准")
        print(f"   抽查 SQL: SELECT writing_quality, original_text FROM dialogue_samples")
        print(f"            WHERE book_name='X' ORDER BY writing_quality DESC LIMIT 3")
        print(f"{'='*60}")

    print("\n小说库工业化构建全部执行完成！")


if __name__ == "__main__":
    main()
