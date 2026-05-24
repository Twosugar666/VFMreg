"""
Loss 工具函数
=========================
- LossMeter         : 累计平均的 loss 计量器
- AdaptiveWeights   : 自适应权重 (uncertainty weighting)
- LossLogger        : 训练日志记录与导出
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class LossMeter:
    """简单的累计平均器，用于跟踪 epoch 内的 loss"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)

    def update(self, breakdown: Dict[str, float], n: int = 1):
        for k, v in breakdown.items():
            self.totals[k] += float(v) * n
            self.counts[k] += n

    def avg(self, key: Optional[str] = None):
        if key is not None:
            if self.counts[key] == 0:
                return 0.0
            return self.totals[key] / self.counts[key]
        return {k: self.totals[k] / max(1, self.counts[k]) for k in self.totals}


class AdaptiveWeights(nn.Module):
    """🌟 多任务自适应损失权重 (Kendall et al., 2018, CVPR)
    L = sum_i exp(-s_i) * L_i + s_i
    其中 s_i 为每个 task 的可学习对数方差
    """

    def __init__(self, n_tasks: int):
        super().__init__()
        self.log_var = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        assert len(losses) == self.log_var.numel()
        total = 0.0
        for i, L in enumerate(losses):
            total = total + torch.exp(-self.log_var[i]) * L + self.log_var[i]
        return total


class LossLogger:
    """训练损失日志记录器，可导出为 json 用于绘图"""

    def __init__(self, out_path: Optional[str] = None):
        self.history: Dict[str, List[float]] = defaultdict(list)
        self.steps: List[int] = []
        self.out_path = out_path

    def log(self, step: int, breakdown: Dict[str, float]):
        self.steps.append(step)
        for k, v in breakdown.items():
            self.history[k].append(float(v))

    def save(self, path: Optional[str] = None):
        path = path or self.out_path
        if path is None:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"steps": self.steps, "history": dict(self.history)},
                f, ensure_ascii=False, indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "LossLogger":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ll = cls()
        ll.steps = data["steps"]
        ll.history = defaultdict(list, data["history"])
        return ll
