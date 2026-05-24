"""临时脚本：只重跑 methods + metrics（复用现有 features）。"""
import sys

# 加载 runner 但屏蔽其顶层的 run_experiment 调用
src = open("sim_all_in_one_runner.py", "r").read()
src_safe = src.replace(
    'result = run_experiment(config_tag="sim64", n_repeats=5, apply_repair=True)',
    '# result = run_experiment(config_tag="sim64", n_repeats=5, apply_repair=True)',
)
mod = type(sys)("sim_runner_mod")
mod.__file__ = "sim_all_in_one_runner.py"
exec(compile(src_safe, "sim_all_in_one_runner.py", "exec"), mod.__dict__)
run_experiment = mod.run_experiment

result = run_experiment(
    config_tag="sim64",
    apply_repair=True,
    do_generate=False,
    do_features=False,
    do_methods=True,
    do_metrics=True,
    reuse_existing_features=True,
    plot_subjects=False,
    sequential=True,
)
print("DONE rerun_methods_only")
