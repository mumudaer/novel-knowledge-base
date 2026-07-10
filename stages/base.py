"""
Stage 基类模块
定义所有 Stage 的通用接口和共享功能
"""
import os
import json
import gc
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.ollama_client import ollama_chat, safe_parse_json
from core.db import get_db_manager
from core.chroma_client import get_chroma_manager
from core.graph import get_graph_manager
from core.utils import (
    get_state_file,
    save_state_atomic,
    generate_id,
)
from stages.stage_q import get_quality_checker

logger = logging.getLogger(__name__)


class BaseStage(ABC):
    """Stage 基类"""

    def __init__(self, stage_name: str, book_name: str, category: str):
        """
        初始化 Stage

        Args:
            stage_name: Stage 标识（A/B/C/D/E/F/G/H）
            book_name: 书名
            category: 分类
        """
        self.stage_name = stage_name
        self.book_name = book_name
        self.category = category
        self.db = get_db_manager()
        self.chroma = get_chroma_manager()
        self.graph = get_graph_manager()
        self.cache_file = get_state_file(book_name, stage_name)
        self.logger = logging.getLogger(f"stage.{stage_name}")

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """执行 Stage 主逻辑"""
        pass

    @abstractmethod
    def insert(self, results: Any) -> Dict[str, int]:
        """将结果写入数据库，返回写入统计"""
        pass

    def load_cache(self, expected_book: str = None, expected_chapter_count: int = None) -> Optional[Dict]:
        """加载断点缓存，可选校验 book_name 和章节数"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                # 校验缓存与当前输入是否一致
                meta = cache.get("_meta", {})
                if expected_book and meta.get("book_name") != expected_book:
                    self.logger.warning(f"⚠️ 缓存书名不匹配 ({meta.get('book_name')} vs {expected_book})，丢弃旧缓存")
                    return None
                if expected_chapter_count and meta.get("chapter_count") != expected_chapter_count:
                    self.logger.warning(f"⚠️ 缓存章节数不匹配 ({meta.get('chapter_count')} vs {expected_chapter_count})，丢弃旧缓存")
                    return None
                self.logger.info(f"✅ 恢复断点缓存: {self.cache_file}")
                return cache
            except Exception as e:
                self.logger.warning(f"⚠️ 缓存加载失败: {e}")
        return None

    def save_cache(self, data: Dict, book_name: str = None, chapter_count: int = None):
        """保存断点缓存，附带元数据用于加载时校验"""
        if book_name or chapter_count:
            data["_meta"] = {}
            if book_name:
                data["_meta"]["book_name"] = book_name
            if chapter_count:
                data["_meta"]["chapter_count"] = chapter_count
        save_state_atomic(self.cache_file, data)

    def cleanup(self):
        """清理临时资源"""
        if os.path.exists(self.cache_file):
            try:
                os.remove(self.cache_file)
            except Exception:
                pass
        gc.collect()

    def run_quality_check(self, results: Any):
        """
        运行质量自检并保存结果
        在每个 Stage 的 insert 完成后调用
        """
        try:
            checker = get_quality_checker()
            issues = checker.run_check(self.stage_name, self.book_name, results)
            if issues:
                checker.save_issues(issues)
                if checker.should_warn(issues):
                    critical_count = sum(1 for i in issues if i.severity == "critical")
                    self.logger.warning(
                        f"\u26a0\ufe0f 质量自检警告: 发现 {critical_count} 个严重问题，"
                        f"共 {len(issues)} 个问题。请查看 quality_checks 表了解详情。"
                    )
            else:
                self.logger.info(f"\u2705 质量自检通过: Stage {self.stage_name}")
        except Exception as e:
            self.logger.warning(f"质量自检执行失败（不影响主流程）: {e}")

    def run_parallel(
        self,
        items: List[Any],
        process_func,
        max_workers: int,
        desc: str = "处理进度",
        save_interval: int = 10,
    ) -> Tuple[List[Dict], List[Tuple]]:
        """
        并行处理通用方法

        Args:
            items: 待处理项列表
            process_func: 处理函数
            max_workers: 最大并发数
            desc: 进度条描述
            save_interval: 保存间隔

        Returns:
            (成功列表, 失败列表)
        """
        success_list = []
        fail_list = []
        completed_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_func, item): i for i, item in enumerate(items)}

            for task in tqdm(as_completed(futures), total=len(futures), desc=desc):
                try:
                    result = task.result()
                    if result:
                        success_list.append(result)
                    completed_count += 1

                    # 定期保存和清理
                    if completed_count % save_interval == 0:
                        # 保存前过滤 _ 前缀临时字段
                        clean_list = [{k:v for k,v in r.items() if not k.startswith('_')} for r in success_list]
                        self.save_cache({"data": clean_list})
                        gc.collect()

                except Exception as e:
                    item_idx = futures[task]
                    fail_list.append((item_idx, str(e)))

        # 最终保存前过滤 _ 前缀临时字段
        clean_list = [{k:v for k,v in r.items() if not k.startswith('_')} for r in success_list]
        self.save_cache({"data": clean_list})

        if fail_list:
            self.logger.warning(f"⚠️ {len(fail_list)} 项处理失败: {fail_list[:5]}...")

        return success_list, fail_list
