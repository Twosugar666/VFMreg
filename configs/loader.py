"""
configs/loader.py
=================
配置加载器：支持 YAML 继承（inherit 字段）、点号路径覆盖（dotted-path overrides）、
环境变量插值。专为本项目 ablation 流程设计。

用法
----
>>> from configs.loader import load_config, expand_variants
>>> cfg = load_config("configs/base.yaml")
>>> for variant_cfg, name in expand_variants("configs/ablation_vfmreg.yaml"):
...     run_train(variant_cfg, exp_name=name)
"""

from __future__ import annotations

import copy
import os
import re
from typing import Any, Dict, Iterator, Tuple

try:
    import yaml
except ImportError as e:                                       # pragma: no cover
    raise ImportError("请先安装 pyyaml: pip install pyyaml") from e


# --------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------- #
_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-(.*?))?\}")


def _interpolate_env(value: Any) -> Any:
    """递归展开 ${ENV:-default} 形式的环境变量"""
    if isinstance(value, str):
        def repl(m):
            return os.environ.get(m.group(1), m.group(2) or "")
        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    return value


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """字典深合并 override 覆盖 base，返回新字典"""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _set_dotted(cfg: Dict, dotted: str, value: Any) -> None:
    """根据 'a.b.c' 形式的 key 设置嵌套字典的值"""
    keys = dotted.split(".")
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    last = keys[-1]
    if isinstance(value, dict) and isinstance(cur.get(last), dict):
        cur[last] = _deep_merge(cur[last], value)
    else:
        cur[last] = value


# --------------------------------------------------------------------- #
# 公共 API
# --------------------------------------------------------------------- #
def load_config(path: str) -> Dict[str, Any]:
    """加载 YAML 配置；自动跟随 ``inherit:`` 字段递归合并父配置。"""
    path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    parent_path = cfg.pop("inherit", None)
    if parent_path:
        parent_full = os.path.normpath(
            os.path.join(os.path.dirname(path), parent_path)
        )
        parent_cfg = load_config(parent_full)
        cfg = _deep_merge(parent_cfg, cfg)

    return _interpolate_env(cfg)


def expand_variants(path: str) -> Iterator[Tuple[Dict[str, Any], str]]:
    """展开 ablation 配置：依次 yield (合并后配置, 变体名)。"""
    cfg = load_config(path)
    variants = cfg.pop("variants", None)
    group = cfg.pop("experiment_group", "default")

    if not variants:
        yield cfg, group
        return

    for variant in variants:
        v_cfg = copy.deepcopy(cfg)
        for dotted, value in (variant.get("overrides") or {}).items():
            _set_dotted(v_cfg, dotted, value)
        v_cfg["_variant_name"] = variant["name"]
        v_cfg["_variant_desc"] = variant.get("desc", "")
        v_cfg["_experiment_group"] = group
        yield v_cfg, f"{group}/{variant['name']}"


# --------------------------------------------------------------------- #
# CLI（快速预览展开后的某变体）
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="YAML 路径")
    parser.add_argument("--variant", default=None, help="变体名 (不传则列出全部)")
    args = parser.parse_args()

    if args.variant is None:
        for cfg, name in expand_variants(args.config):
            print(f"[{name}] {cfg.get('_variant_desc', '')}")
    else:
        for cfg, name in expand_variants(args.config):
            if name.endswith("/" + args.variant) or name == args.variant:
                print(json.dumps(cfg, indent=2, ensure_ascii=False))
                break
