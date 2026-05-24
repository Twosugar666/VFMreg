"""
utils/io.py
===========
IO 帮手：JSON / YAML 安全读写、ckpt 管理、日志器、原子写入。
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# -----------------------------------------------------------
# 文件 IO
# -----------------------------------------------------------
def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def atomic_write(path: str, content: str, encoding: str = "utf-8") -> None:
    """原子写文本：先写临时文件再重命名，避免中断导致半截文件"""
    ensure_dir(os.path.dirname(path) or ".")
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str, indent: int = 2) -> None:
    atomic_write(path, json.dumps(obj, indent=indent, ensure_ascii=False))


def load_yaml(path: str) -> Any:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: Any, path: str) -> None:
    import yaml
    atomic_write(path, yaml.safe_dump(obj, sort_keys=False, allow_unicode=True))


# -----------------------------------------------------------
# Checkpoint 管理（保留最近 N 个）
# -----------------------------------------------------------
class CheckpointManager:
    """
    保存 / 加载训练 ckpt，自动维护最多 keep 个最新文件。
    """

    def __init__(self, ckpt_dir: str, keep: int = 3, prefix: str = "ckpt"):
        self.dir = ensure_dir(ckpt_dir)
        self.keep = keep
        self.prefix = prefix

    def save(self, state: Dict[str, Any], step: int) -> str:
        try:
            import torch
        except ImportError as e:
            raise ImportError("CheckpointManager.save 需要 torch") from e

        path = os.path.join(self.dir, f"{self.prefix}_{step:08d}.pt")
        torch.save(state, path)
        self._cleanup()
        return path

    def latest(self) -> Optional[str]:
        files = self._list()
        return files[-1] if files else None

    def load_latest(self) -> Optional[Dict[str, Any]]:
        try:
            import torch
        except ImportError as e:
            raise ImportError("CheckpointManager.load_latest 需要 torch") from e

        latest = self.latest()
        if latest is None:
            return None
        return torch.load(latest, map_location="cpu")

    def _list(self) -> List[str]:
        files = [
            os.path.join(self.dir, f)
            for f in os.listdir(self.dir)
            if f.startswith(self.prefix) and f.endswith(".pt")
        ]
        return sorted(files)

    def _cleanup(self) -> None:
        files = self._list()
        for f in files[: max(0, len(files) - self.keep)]:
            try:
                os.unlink(f)
            except OSError:
                pass


# -----------------------------------------------------------
# 日志器
# -----------------------------------------------------------
def get_logger(
    name: str = "meg",
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """获取一个带控制台 + 文件输出的 logger（可重复调用，不重复添加 handler）"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_dir is not None:
        ensure_dir(log_dir)
        fname = os.path.join(log_dir, f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
        fh = logging.FileHandler(fname, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# -----------------------------------------------------------
# 复制 / 备份
# -----------------------------------------------------------
def snapshot_config(cfg_path: str, run_dir: str) -> str:
    """把当前配置文件复制一份到 run_dir 下，便于复现"""
    ensure_dir(run_dir)
    dst = os.path.join(run_dir, os.path.basename(cfg_path))
    shutil.copyfile(cfg_path, dst)
    return dst
