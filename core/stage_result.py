"""
Stage 执行结果数据结构
提供统一的错误处理和执行报告
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class StageResult:
    """Stage 执行结果"""
    data: Dict[str, List[Dict]] = field(default_factory=dict)
    failures: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)

    def add_failure(self, chapter_id: str, error: str, stage: str):
        """记录失败"""
        self.failures.append({
            "chapter_id": chapter_id,
            "error": error,
            "stage": stage,
        })

    def get_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        return {
            "success_count": sum(len(v) for v in self.data.values() if v is not None),
            "failure_count": len(self.failures),
            "failures": self.failures[:10],  # 只返回前10个失败记录
        }

    def merge(self, other: "StageResult"):
        """合并另一个 StageResult"""
        for key, items in other.data.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].extend(items)
        self.failures.extend(other.failures)
        for key, value in other.stats.items():
            self.stats[key] = self.stats.get(key, 0) + value
