"""
Stage I: 纯统计模块（无需 LLM）
从原文和已有数据中计算量化指标：字数、对话占比、段落长度分布、节奏模式
"""
import re
import logging
import statistics
from typing import List, Dict, Any
from stages.base import BaseStage
from core.utils import generate_id

logger = logging.getLogger(__name__)


class StageI(BaseStage):
    """Stage I: 纯统计模块（无需 LLM）"""

    def __init__(self, book_name: str, category: str):
        super().__init__("I", book_name, category)

    def run(self, chapters: List[Dict], **kwargs) -> Dict[str, List[Dict]]:
        """
        执行 Stage I

        Args:
            chapters: 章节列表（包含 text 字段）

        Returns:
            包含 book_statistics 的字典
        """
        logger.info(f"=== 阶段九：纯统计模块 ({self.book_name}) ===")

        result = {"book_statistics": []}

        if not chapters:
            logger.warning("⚠️ [阶段I] 没有章节数据，跳过统计。")
            return result

        # 收集每章的字数
        chapter_word_counts = []
        total_dialogue_chars = 0
        total_description_chars = 0
        total_text_chars = 0
        all_paragraph_lengths = []

        for chap in chapters:
            text = chap.get("text", "")
            chapter_word_counts.append(len(text))
            total_text_chars += len(text)

            # 统计对话字数（引号内的内容）
            dialogue_matches = re.findall(r'[""「『](.*?)[""」』]', text, re.DOTALL)
            dialogue_chars = sum(len(m) for m in dialogue_matches)
            total_dialogue_chars += dialogue_chars

            # 统计段落长度
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
            for para in paragraphs:
                all_paragraph_lengths.append(len(para))

        # 计算字数统计
        total_words = sum(chapter_word_counts)
        avg_chapter_words = int(statistics.mean(chapter_word_counts)) if chapter_word_counts else 0
        min_chapter_words = min(chapter_word_counts) if chapter_word_counts else 0
        max_chapter_words = max(chapter_word_counts) if chapter_word_counts else 0
        median_chapter_words = int(statistics.median(chapter_word_counts)) if chapter_word_counts else 0

        # 计算对话占比
        dialogue_ratio = round(total_dialogue_chars / total_text_chars, 4) if total_text_chars > 0 else 0.0

        # 计算描写占比（非对话部分，简化为 1 - 对话占比）
        description_ratio = round(1.0 - dialogue_ratio, 4)

        # 计算段落长度分布
        avg_paragraph_length = round(statistics.mean(all_paragraph_lengths), 2) if all_paragraph_lengths else 0.0

        # 段落长度分类：短（<50字）、中（50-150字）、长（>150字）
        short_para_count = sum(1 for length in all_paragraph_lengths if length < 50)
        medium_para_count = sum(1 for length in all_paragraph_lengths if 50 <= length <= 150)
        long_para_count = sum(1 for length in all_paragraph_lengths if length > 150)
        total_paras = len(all_paragraph_lengths)

        short_para_ratio = round(short_para_count / total_paras, 4) if total_paras > 0 else 0.0
        medium_para_ratio = round(medium_para_count / total_paras, 4) if total_paras > 0 else 0.0
        long_para_ratio = round(long_para_count / total_paras, 4) if total_paras > 0 else 0.0

        # 节奏模式分析
        rhythm_pattern = self._analyze_rhythm_pattern(
            short_para_ratio, medium_para_ratio, long_para_ratio, dialogue_ratio
        )

        result["book_statistics"].append({
            "book_name": self.book_name,
            "total_words": total_words,
            "avg_chapter_words": avg_chapter_words,
            "min_chapter_words": min_chapter_words,
            "max_chapter_words": max_chapter_words,
            "median_chapter_words": median_chapter_words,
            "dialogue_ratio": dialogue_ratio,
            "description_ratio": description_ratio,
            "avg_paragraph_length": avg_paragraph_length,
            "short_para_ratio": short_para_ratio,
            "medium_para_ratio": medium_para_ratio,
            "long_para_ratio": long_para_ratio,
            "rhythm_pattern": rhythm_pattern,
        })

        logger.info(
            f"✅ [阶段I战报] 总字数: {total_words} | "
            f"平均章节字数: {avg_chapter_words} | "
            f"对话占比: {dialogue_ratio:.2%} | "
            f"段落平均长度: {avg_paragraph_length} | "
            f"节奏模式: {rhythm_pattern}"
        )
        return result

    def _analyze_rhythm_pattern(
        self,
        short_ratio: float,
        medium_ratio: float,
        long_ratio: float,
        dialogue_ratio: float,
    ) -> str:
        """分析节奏模式"""
        patterns = []

        # 段落长度模式
        if short_ratio > 0.5:
            patterns.append("短句密集（快节奏）")
        elif long_ratio > 0.5:
            patterns.append("长句为主（慢节奏）")
        else:
            patterns.append("长短交替（均衡节奏）")

        # 对话密度模式
        if dialogue_ratio > 0.4:
            patterns.append("对话驱动型")
        elif dialogue_ratio < 0.2:
            patterns.append("叙述驱动型")
        else:
            patterns.append("对话与叙述均衡")

        return "；".join(patterns)

    def insert(self, results: Dict[str, List[Dict]]) -> Dict[str, int]:
        """将 Stage I 结果写入数据库"""
        cursor = self.db.connect().cursor()
        stats = {"book_statistics": 0}

        for bs in results.get("book_statistics", []):
            bs_id = generate_id(bs["book_name"], "statistics")
            cursor.execute(
                "INSERT OR REPLACE INTO book_statistics VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    bs_id, bs["book_name"],
                    bs["total_words"], bs["avg_chapter_words"],
                    bs["min_chapter_words"], bs["max_chapter_words"],
                    bs["median_chapter_words"],
                    bs["dialogue_ratio"], bs["description_ratio"],
                    bs["avg_paragraph_length"],
                    bs["short_para_ratio"], bs["medium_para_ratio"],
                    bs["long_para_ratio"], bs["rhythm_pattern"],
                ),
            )
            stats["book_statistics"] += 1

        self.db.commit()
        logger.info(f"   ✅ [阶段I战报] 统计指标: {stats['book_statistics']}")
        return stats
