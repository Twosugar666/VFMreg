"""
utils/timer.py
==============
计时器：上下文管理器风格 + 实时统计平均/最值，便于推理延迟测量。
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Deque, Dict


class TimerStats:
    """记录多个 tag 的累计耗时统计"""

    def __init__(self, window: int = 100):
        self._buf: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window))

    def add(self, tag: str, ms: float) -> None:
        self._buf[tag].append(ms)

    def avg(self, tag: str) -> float:
        b = self._buf[tag]
        return sum(b) / len(b) if b else 0.0

    def min(self, tag: str) -> float:
        return min(self._buf[tag]) if self._buf[tag] else 0.0

    def max(self, tag: str) -> float:
        return max(self._buf[tag]) if self._buf[tag] else 0.0

    def report(self) -> Dict[str, Dict[str, float]]:
        return {
            tag: {
                "avg_ms": self.avg(tag),
                "min_ms": self.min(tag),
                "max_ms": self.max(tag),
                "n": len(self._buf[tag]),
            }
            for tag in self._buf
        }


_GLOBAL = TimerStats()


@contextmanager
def timer(tag: str, stats: TimerStats = _GLOBAL):
    """用法：
    with timer('inference'):
        model(x)
    print(stats.avg('inference'))
    """
    t0 = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    stats.add(tag, elapsed_ms)


def get_global_stats() -> TimerStats:
    return _GLOBAL
