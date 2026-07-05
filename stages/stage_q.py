"""
质量自检模块 (Stage Q)
在每个 Stage 完成后自动运行轻量级质量检查，发现提取错误并记录
"""
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from core.db import get_db_manager
from core.utils import generate_id

logger = logging.getLogger(__name__)


class QualityIssue:
    """质量问题记录"""

    def __init__(self, stage: str, book_name: str, severity: str,
                 chapter_id: str = "", description: str = "", suggestion: str = ""):
        self.stage = stage
        self.book_name = book_name
        self.severity = severity  # critical / high / medium / low
        self.chapter_id = chapter_id
        self.description = description
        self.suggestion = suggestion

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "book_name": self.book_name,
            "severity": self.severity,
            "chapter_id": self.chapter_id,
            "description": self.description,
            "suggestion": self.suggestion,
        }


class QualityChecker:
    """轻量级质量自检器"""

    # 每个 Stage 的 critical 问题阈值（超过则报告警告）
    CRITICAL_THRESHOLDS = {
        "A": 0.1,   # 摘要失败率超过 10%
        "B": 0.15,  # 技法提取失败率超过 15%
        "C": 0.15,
        "D": 0.2,   # 世界观/人物提取缺失率超过 20%
        "E": 0.2,
        "F": 0.15,
        "G": 0.15,
        "H": 0.2,
        "I": 0.1,
    }

    def check_stage_a(self, book_name: str, chapters: List[Dict]) -> List[QualityIssue]:
        """
        检查 Stage A（剧情摘要与人物状态）质量
        - 摘要是否为空或过短
        - 摘要是否全是"处理失败"
        - 人物状态是否合理
        """
        issues = []
        total = len(chapters)
        if total == 0:
            return issues

        fail_count = 0
        short_count = 0

        for chap in chapters:
            chap_id = chap.get("id", "unknown")
            summary = chap.get("summary", "")
            char_state = chap.get("character_state", {})

            # Critical: 摘要完全失败
            if summary == "处理失败" or not summary:
                issues.append(QualityIssue(
                    stage="A", book_name=book_name, severity="critical",
                    chapter_id=chap_id,
                    description="摘要生成失败或为空",
                    suggestion="建议重新处理该章节，可能是模型返回格式异常",
                ))
                fail_count += 1
                continue

            # High: 摘要过短（不足50字），可能遗漏关键信息
            if len(summary) < 50:
                issues.append(QualityIssue(
                    stage="A", book_name=book_name, severity="high",
                    chapter_id=chap_id,
                    description=f"摘要过短（{len(summary)}字），可能遗漏关键剧情",
                    suggestion="建议检查原文是否包含重要转折或事件",
                ))
                short_count += 1

            # Medium: 人物状态为空
            if not char_state or (len(char_state) == 1 and "旁白" in char_state):
                issues.append(QualityIssue(
                    stage="A", book_name=book_name, severity="medium",
                    chapter_id=chap_id,
                    description="人物状态为空或仅有旁白",
                    suggestion="可能是纯描写章节或模型未识别人物",
                ))

        # 统计级别的警告
        if total > 0:
            fail_rate = fail_count / total
            if fail_rate > self.CRITICAL_THRESHOLDS["A"]:
                issues.append(QualityIssue(
                    stage="A", book_name=book_name, severity="critical",
                    description=f"摘要失败率过高: {fail_rate:.1%} ({fail_count}/{total})",
                    suggestion="建议检查 Ollama 服务状态和模型是否正常加载",
                ))

        return issues

    def check_stage_b(self, book_name: str, results: Any) -> List[QualityIssue]:
        """检查 Stage B（写作技法）质量"""
        issues = []

        if isinstance(results, list):
            for item in results:
                skills = item.get("narrative_skills", [])
                chap_id = item.get("id", "unknown")
                if not skills:
                    issues.append(QualityIssue(
                        stage="B", book_name=book_name, severity="medium",
                        chapter_id=chap_id,
                        description="该章节未提取到任何写作技法",
                        suggestion="可能是纯叙述章节或技法不明显",
                    ))

        return issues

    def check_stage_d(self, book_name: str, results: Any) -> List[QualityIssue]:
        """
        检查 Stage D（世界观与人物）质量
        - 世界观维度覆盖度
        - 人物档案完整性
        """
        issues = []

        if isinstance(results, dict):
            # 检查世界观数据
            world_settings = results.get("world_settings", [])
            if not world_settings:
                issues.append(QualityIssue(
                    stage="D", book_name=book_name, severity="high",
                    description="未提取到任何世界观设定",
                    suggestion="建议检查采样章节是否包含足够的世界观信息",
                ))
            else:
                # 检查维度覆盖度
                modules = set()
                for ws in world_settings:
                    module = ws.get("module", "")
                    if module:
                        modules.add(module)
                expected_modules = {"力量体系", "地理版图", "社会结构", "经济系统", "文化习俗", "历史年表", "因果法则"}
                missing = expected_modules - modules
                if len(missing) > 3:
                    issues.append(QualityIssue(
                        stage="D", book_name=book_name, severity="medium",
                        description=f"世界观维度覆盖不足，缺少: {', '.join(list(missing)[:3])}等",
                        suggestion="可能需要增加采样章节数量",
                    ))

            # 检查人物数据
            char_profiles = results.get("character_profiles", [])
            if not char_profiles:
                issues.append(QualityIssue(
                    stage="D", book_name=book_name, severity="high",
                    description="未提取到任何人物档案",
                    suggestion="建议检查 Stage A 的主角识别是否正确",
                ))
            else:
                for cp in char_profiles:
                    name = cp.get("name", "unknown")
                    # 检查关键字段是否缺失
                    key_fields = ["motivation", "personality", "identity"]
                    empty_fields = [f for f in key_fields if not cp.get(f, "")]
                    if len(empty_fields) >= 2:
                        issues.append(QualityIssue(
                            stage="D", book_name=book_name, severity="medium",
                            description=f"人物 '{name}' 关键档案缺失: {', '.join(empty_fields)}",
                            suggestion="该人物可能在采样章节中出场较少",
                        ))

        return issues

    def check_stage_e(self, book_name: str, results: Any) -> List[QualityIssue]:
        """检查 Stage E（宏观大纲）质量"""
        issues = []

        if isinstance(results, dict):
            outlines = results.get("macro_outlines", [])
            if not outlines:
                issues.append(QualityIssue(
                    stage="E", book_name=book_name, severity="high",
                    description="未生成任何宏观大纲",
                    suggestion="建议检查 Stage A 的摘要是否完整",
                ))

        return issues

    def check_stage_f(self, book_name: str, results: Any) -> List[QualityIssue]:
        """检查 Stage F（样本库）质量"""
        issues = []
        if not isinstance(results, dict):
            return issues

        # 各样本类型的检查配置: (key, label, min_text_len, min_wq)
        sample_checks = [
            ("dialogue_samples", "对话样本", 30, 3),
            ("description_samples", "描写样本", 30, 3),
            ("transition_samples", "转场样本", 20, 3),
            ("narrative_distance", "叙事距离", 20, 3),
            ("show_tell_patterns", "Show/Tell", 20, 3),
            ("action_scene_samples", "动作场景", 30, 3),
            ("climax_excerpts", "高潮段落", 50, 3),
            ("memorable_quotes", "金句名句", 15, 3),
        ]

        for key, label, min_len, min_wq in sample_checks:
            samples = results.get(key, [])
            if not samples:
                continue  # 该类型无样本，不报错（可能本章确实没有）

            empty_count = 0
            short_count = 0
            low_wq_count = 0

            for s in samples:
                text = s.get("original_text") or s.get("original_example") or s.get("quote_text") or ""
                wq = s.get("writing_quality", 5)

                if not text or not text.strip():
                    empty_count += 1
                elif len(text) < min_len:
                    short_count += 1

                if isinstance(wq, (int, float)) and wq < min_wq:
                    low_wq_count += 1

            total = len(samples)
            if empty_count > 0:
                issues.append(QualityIssue(
                    stage="F", book_name=book_name, severity="high",
                    description=f"{label} 中 {empty_count}/{total} 条 original_text 为空",
                    suggestion="可能是 LLM 返回格式异常，检查对应章节的 prompt 响应",
                ))
            if short_count > total * 0.3:
                issues.append(QualityIssue(
                    stage="F", book_name=book_name, severity="medium",
                    description=f"{label} 中 {short_count}/{total} 条文本过短 (<{min_len}字)",
                    suggestion="采样章节可能信息密度偏低，或 LLM 摘录过于保守",
                ))
            if low_wq_count > total * 0.5:
                issues.append(QualityIssue(
                    stage="F", book_name=book_name, severity="medium",
                    description=f"{label} 中 {low_wq_count}/{total} 条 writing_quality < {min_wq}",
                    suggestion="该类型样本质量偏低，考虑增加采样章节或调整 prompt",
                ))

        # 高潮段落专项：检查 emotional_impact
        for climax in results.get("climax_excerpts", []):
            if not climax.get("emotional_impact", "").strip():
                issues.append(QualityIssue(
                    stage="F", book_name=book_name, severity="medium",
                    description="存在高潮段落缺少 emotional_impact 分析",
                    suggestion="LLM 可能未按要求输出情感冲击力分析",
                ))

        return issues

    def check_stage_c(self, book_name: str, results: Any) -> List[QualityIssue]:
        """检查 Stage C（文风指纹）质量"""
        issues = []
        if not isinstance(results, dict):
            return issues

        fingerprints = results.get("author_fingerprints", [])
        for fp in fingerprints:
            if not fp.get("narrative_perspective", ""):
                issues.append(QualityIssue(
                    stage="C", book_name=book_name, severity="medium",
                    description="文风指纹缺少叙事视角字段",
                ))

        sensory = results.get("sensory_db") or results.get("sensory_mappings", [])
        if not sensory:
            issues.append(QualityIssue(
                stage="C", book_name=book_name, severity="high",
                description="未提取到任何感官映射",
                suggestion="可能是纯对话小说或模型未正常输出",
            ))

        return issues

    def check_stage_g(self, book_name: str, results: Any) -> List[QualityIssue]:
        """检查 Stage G（人物深度特征）质量"""
        issues = []
        if not isinstance(results, dict):
            return issues

        speech = results.get("speech_style") or results.get("character_speech_style", [])
        if not speech:
            issues.append(QualityIssue(
                stage="G", book_name=book_name, severity="high",
                description="未提取到任何人物语言风格",
            ))

        behavior = results.get("behavior_marks") or results.get("character_behavior_marks", [])
        if not behavior:
            issues.append(QualityIssue(
                stage="G", book_name=book_name, severity="medium",
                description="未提取到任何人物行为标志",
            ))

        relations = results.get("relationship_dynamics") or results.get("character_relationship_dynamics", [])
        if not relations:
            issues.append(QualityIssue(
                stage="G", book_name=book_name, severity="medium",
                description="未提取到任何人物关系动态",
            ))

        return issues

    def check_stage_o(self, book_name: str, results: Any) -> List[QualityIssue]:
        """检查 Stage O（事件因果图谱）质量"""
        issues = []
        if not isinstance(results, dict):
            return issues

        events = results.get("story_events", [])
        if not events:
            issues.append(QualityIssue(
                stage="O", book_name=book_name, severity="high",
                description="未提取到任何关键事件",
            ))

        edges = results.get("event_causal_edges", [])
        if edges:
            for edge in edges:
                if not edge.get("relation_type", ""):
                    issues.append(QualityIssue(
                        stage="O", book_name=book_name, severity="low",
                        description="存在因果边缺少 relation_type",
                    ))

        return issues

    def check_stage_h(self, book_name: str, results: Any) -> List[QualityIssue]:
        """检查 Stage H（全书宏观分析）质量"""
        issues = []
        if isinstance(results, dict):
            if not results.get("book_structure"):
                issues.append(QualityIssue(
                    stage="H", book_name=book_name, severity="high",
                    description="未生成书籍结构分析",
                    suggestion="建议检查 Stage A 的摘要是否覆盖了全书",
                ))
        return issues

    def run_check(self, stage: str, book_name: str, results: Any) -> List[QualityIssue]:
        """根据 Stage 类型自动选择检查方法"""
        checker_map = {
            "A": self.check_stage_a,
            "B": self.check_stage_b,
            "C": self.check_stage_c,
            "D": self.check_stage_d,
            "E": self.check_stage_e,
            "F": self.check_stage_f,
            "G": self.check_stage_g,
            "H": self.check_stage_h,
            "O": self.check_stage_o,
        }

        checker = checker_map.get(stage)
        if checker:
            return checker(book_name, results)
        return []

    def save_issues(self, issues: List[QualityIssue]):
        """将质量问题写入数据库"""
        if not issues:
            return

        db = get_db_manager()
        cursor = db.connect().cursor()

        for issue in issues:
            issue_id = generate_id(
                issue.book_name, issue.stage, issue.chapter_id,
                issue.description[:50], datetime.now().isoformat()
            )
            try:
                cursor.execute(
                    """INSERT OR REPLACE INTO quality_checks 
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        issue_id,
                        issue.book_name,
                        issue.stage,
                        issue.chapter_id,
                        issue.severity,
                        issue.description,
                        issue.suggestion,
                        json.dumps(issue.to_dict(), ensure_ascii=False),
                        datetime.now().isoformat(),
                    ),
                )
            except Exception as e:
                logger.warning(f"保存质量问题记录失败: {e}")

        db.commit()

        # 统计报告
        critical = sum(1 for i in issues if i.severity == "critical")
        high = sum(1 for i in issues if i.severity == "high")
        medium = sum(1 for i in issues if i.severity == "medium")

        if critical > 0 or high > 0:
            logger.warning(
                f"质量自检报告 [{issues[0].stage}][{issues[0].book_name}]: "
                f"critical={critical}, high={high}, medium={medium}"
            )
        else:
            logger.info(
                f"质量自检通过 [{issues[0].stage}][{issues[0].book_name}]: "
                f"medium={medium}"
            )

    def should_warn(self, issues: List[QualityIssue]) -> bool:
        """判断是否需要发出警告（有 critical 或超过 5 个 high）"""
        critical_count = sum(1 for i in issues if i.severity == "critical")
        high_count = sum(1 for i in issues if i.severity == "high")
        return critical_count > 0 or high_count > 5


# 全局质量检查器实例
_global_checker: Optional[QualityChecker] = None


def get_quality_checker() -> QualityChecker:
    """获取全局质量检查器"""
    global _global_checker
    if _global_checker is None:
        _global_checker = QualityChecker()
    return _global_checker
