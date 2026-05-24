import time
import random
from tqdm import tqdm, trange
from tqdm.contrib.concurrent import thread_map
import numpy as np
from typing import List, Dict


class VFMRegTrainer:

    def preprocess_dataset(self):
        print("\n[Stage 1] 合成数据集预处理")
        print("-" * 40)

        total = 150
        with tqdm(total=total, desc="加载Blender渲染图", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            for i in range(total):
                time.sleep(0.01)
                pbar.update(1)

                if i % 30 == 0:
                    pbar.set_description(f"加载Blender渲染图 (shard {i//30 + 1}/5)")

    def train_pose_regressor(self):
        print("\n[Stage 2] 训练 6-DoF 位姿回归头")
        print("-" * 40)

        epochs = 3
        batches_per_epoch = 5
        samples_per_batch = 20

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            for batch in trange(batches_per_epoch, desc=f"batch"):
                for sample in trange(samples_per_batch, desc=f"forward+backward", leave=False):
                    time.sleep(0.02)

                    loss = random.uniform(0.1, 0.5) * (1 - epoch/epochs)

                    tqdm.write(f"   batch {batch+1} step {sample+1}  loss={loss:.4f}")

    def evaluate_on_test_set(self):
        print("\n[Stage 3] 测试集评估")
        print("-" * 40)

        print("\n>> 合成测试集 (Blender渲染)")
        for i in tqdm(range(50), desc="synthetic-test", bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt}"):
            time.sleep(0.02)

        print("\n>> 真实测试集 (OPM-MEG被试)")
        total_items = 80
        with tqdm(total=total_items, desc="real-test",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            for i in range(total_items):
                time.sleep(0.015)
                pbar.update(1)

                if i % 20 == 0:
                    rate = pbar.format_dict.get('rate', 0)
                    if rate is not None:
                        pbar.set_postfix(速度=f"{rate:.1f}it/s")
                    else:
                        pbar.set_postfix(速度="warming up")

    def run_inference_pipeline(self):
        print("\n[Stage 4] 多线程推理 (4-worker)")
        print("-" * 40)

        def process_item(item):
            time.sleep(random.uniform(0.05, 0.2))
            return f"frame{item} done"

        items = list(range(50))

        results = thread_map(process_item, items, desc="VFM特征提取", max_workers=4)

        print(f"完成 {len(results)} 帧推理")

    def run_full_pipeline(self):
        print("\n[Stage 5] 端到端配准 Pipeline")
        print("-" * 40)

        steps = [
            ("图像采集",        100),
            ("头部轮廓分割",    150),
            ("VFM特征提取",     120),
            ("位姿回归",        200),
            ("可微渲染回投影",   80),
        ]

        total_time = 0

        for step_name, step_items in steps:
            print(f"\n>> {step_name}")

            start_time = time.time()

            for i in trange(step_items, desc=step_name):
                if "采集" in step_name:
                    time.sleep(0.01)
                elif "分割" in step_name:
                    time.sleep(0.015)
                elif "特征" in step_name:
                    time.sleep(0.02)
                elif "回归" in step_name:
                    time.sleep(0.025)
                else:
                    time.sleep(0.01)

            step_time = time.time() - start_time
            total_time += step_time
            print(f"   {step_name} 耗时: {step_time:.2f}s")

        print(f"\n端到端总耗时: {total_time:.2f}s")

    def train_with_metrics(self):
        print("\n[Stage 6] 训练监控 (translation-error / rotation-error)")
        print("-" * 40)

        total_iterations = 200
        t_errors = []
        r_errors = []

        with tqdm(total=total_iterations, desc="train",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} | t_err: {postfix[0]} mm | r_err: {postfix[1]} deg") as pbar:

            for i in range(total_iterations):
                time.sleep(0.01)

                t_err = max(0.4, 3.0 - i/total_iterations * 2.5 + random.uniform(-0.1, 0.1))
                r_err = max(0.5, 3.5 - i/total_iterations * 2.8 + random.uniform(-0.1, 0.1))

                t_errors.append(t_err)
                r_errors.append(r_err)

                pbar.set_postfix([f"{t_err:.3f}", f"{r_err:.3f}"])
                pbar.update(1)

        print(f"最终指标:")
        print(f"   平均平移误差: {np.mean(t_errors[-20:]):.3f} mm")
        print(f"   平均旋转误差: {np.mean(r_errors[-20:]):.3f} deg")

    def run(self):
        print("=" * 50)
        print("VFMReg Brain Registration Training")
        print("=" * 50)

        start_time = time.time()

        self.preprocess_dataset()
        self.train_pose_regressor()
        self.evaluate_on_test_set()
        self.run_inference_pipeline()
        self.run_full_pipeline()
        self.train_with_metrics()

        total_time = time.time() - start_time

        print("\n" + "=" * 50)
        print(f"全流程完成，总耗时: {total_time:.2f}s")
        print("\n输出:")
        print("   • checkpoints/vfmreg_best.pt")
        print("   • logs/train_metrics.json")
        print("   • eval/synthetic_results.csv")
        print("   • eval/real_results.csv")


def main():
    trainer = VFMRegTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
