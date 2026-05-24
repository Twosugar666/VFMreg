#!/usr/bin/env python3
"""
合成数据生成器 - 带进度条演示
==============================
模拟生成实验数据，主要用于演示进度条效果

功能：
- 生成数值数据（正态分布、均匀分布）
- 生成文本数据（随机句子）
- 生成图像数据（模拟灰度图）
- 实时进度条显示
- 数据验证和统计

用法：
    python synthetic_data_generator.py
    python synthetic_data_generator.py --samples 5000 --types all
"""

import argparse
import json
import random
import time
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Any


class SyntheticDataGenerator:
    """合成数据生成器"""
    
    def __init__(self, seed: int = 42):
        """初始化生成器"""
        random.seed(seed)
        np.random.seed(seed)
        
        # 预定义的文本模板
        self.text_templates = [
            "样本编号 {id} 的实验结果",
            "第 {id} 次测试数据记录",
            "实验批次 {batch} 的观测值",
            "传感器 {sensor} 在时间 {time} 的读数",
            "算法 {algo} 在数据集 {dataset} 上的表现"
        ]
        
        self.algorithms = ["SVM", "RandomForest", "NeuralNetwork", "KNN", "XGBoost"]
        self.datasets = ["MNIST", "CIFAR10", "ImageNet", "COCO", "Synthetic"]
        self.sensors = ["温度", "压力", "湿度", "加速度", "陀螺仪"]
    
    def generate_numeric_data(self, num_samples: int) -> List[Dict]:
        """生成数值数据"""
        data = []
        
        # 使用进度条
        pbar = tqdm(total=num_samples, desc="生成数值数据", unit="样本")
        
        for i in range(num_samples):
            # 生成正态分布数据
            normal_data = np.random.normal(0, 1, 10)
            
            # 生成均匀分布数据
            uniform_data = np.random.uniform(-5, 5, 5)
            
            sample = {
                "sample_id": i + 1,
                "normal_features": normal_data.tolist(),
                "uniform_features": uniform_data.tolist(),
                "target_value": np.random.normal(0, 0.5),
                "timestamp": time.time() + i
            }
            
            data.append(sample)
            pbar.update(1)
            time.sleep(0.001)  # 模拟处理时间
        
        pbar.close()
        return data
    
    def generate_text_data(self, num_samples: int) -> List[Dict]:
        """生成文本数据"""
        data = []
        
        pbar = tqdm(total=num_samples, desc="生成文本数据", unit="样本")
        
        for i in range(num_samples):
            template = random.choice(self.text_templates)
            
            # 填充模板
            text = template.format(
                id=i+1,
                batch=random.randint(1, 10),
                sensor=random.choice(self.sensors),
                time=time.strftime("%Y-%m-%d %H:%M:%S"),
                algo=random.choice(self.algorithms),
                dataset=random.choice(self.datasets)
            )
            
            # 添加一些随机词汇
            words = ["优秀", "良好", "一般", "较差", "异常"]
            text += f"，评估结果为：{random.choice(words)}"
            
            sample = {
                "text_id": i + 1,
                "content": text,
                "length": len(text),
                "category": random.choice(["实验记录", "观测报告", "测试结果"]),
                "confidence": round(random.uniform(0.7, 0.99), 3)
            }
            
            data.append(sample)
            pbar.update(1)
            time.sleep(0.002)  # 文本生成稍慢
        
        pbar.close()
        return data
    
    def generate_image_data(self, num_samples: int) -> List[Dict]:
        """模拟生成图像数据（灰度图）"""
        data = []
        
        pbar = tqdm(total=num_samples, desc="生成图像数据", unit="图像")
        
        for i in range(num_samples):
            # 模拟生成 28x28 的灰度图像素
            height, width = 28, 28
            pixels = np.random.randint(0, 256, (height, width))
            
            # 计算一些图像统计信息
            mean_intensity = np.mean(pixels)
            std_intensity = np.std(pixels)
            
            sample = {
                "image_id": i + 1,
                "dimensions": f"{height}x{width}",
                "mean_intensity": float(mean_intensity),
                "std_intensity": float(std_intensity),
                "pixel_count": height * width,
                "simulated_pixels": "28x28灰度矩阵（模拟）"
            }
            
            data.append(sample)
            pbar.update(1)
            time.sleep(0.005)  # 图像生成最慢
        
        pbar.close()
        return data
    
    def generate_all_data(self, num_samples: int) -> Dict[str, List]:
        """生成所有类型的数据"""
        print(f"🚀 开始生成 {num_samples} 个样本的合成数据...")
        print("-" * 50)
        
        start_time = time.time()
        
        # 生成不同类型的数据
        numeric_data = self.generate_numeric_data(num_samples)
        text_data = self.generate_text_data(num_samples)
        image_data = self.generate_image_data(num_samples)
        
        end_time = time.time()
        
        # 统计信息
        total_samples = len(numeric_data) + len(text_data) + len(image_data)
        elapsed_time = end_time - start_time
        
        print("-" * 50)
        print(f"✅ 数据生成完成！")
        print(f"📊 总样本数: {total_samples}")
        print(f"⏱️  耗时: {elapsed_time:.2f} 秒")
        print(f"📈 平均速度: {total_samples/elapsed_time:.1f} 样本/秒")
        
        return {
            "numeric": numeric_data,
            "text": text_data,
            "image": image_data,
            "metadata": {
                "total_samples": total_samples,
                "generation_time": elapsed_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
    
    def save_data(self, data: Dict, output_dir: str = "synthetic_data"):
        """保存生成的数据"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n💾 保存数据到 {output_path}...")
        
        # 保存不同类型的数据
        with tqdm(total=3, desc="保存数据文件") as pbar:
            # 保存数值数据
            with open(output_path / "numeric_data.json", "w", encoding="utf-8") as f:
                json.dump(data["numeric"], f, indent=2, ensure_ascii=False)
            pbar.update(1)
            time.sleep(0.1)
            
            # 保存文本数据
            with open(output_path / "text_data.json", "w", encoding="utf-8") as f:
                json.dump(data["text"], f, indent=2, ensure_ascii=False)
            pbar.update(1)
            time.sleep(0.1)
            
            # 保存图像数据
            with open(output_path / "image_data.json", "w", encoding="utf-8") as f:
                json.dump(data["image"], f, indent=2, ensure_ascii=False)
            pbar.update(1)
            time.sleep(0.1)
        
        # 保存元数据
        with open(output_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(data["metadata"], f, indent=2, ensure_ascii=False)
        
        print(f"✅ 数据保存完成！")
        print(f"📁 输出目录: {output_path.absolute()}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="合成数据生成器")
    parser.add_argument("--samples", type=int, default=1000, 
                       help="每种数据类型生成的样本数（默认: 1000）")
    parser.add_argument("--output", type=str, default="synthetic_data",
                       help="输出目录（默认: synthetic_data）")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子（默认: 42）")
    
    args = parser.parse_args()
    
    # 创建生成器
    generator = SyntheticDataGenerator(seed=args.seed)
    
    # 生成数据
    data = generator.generate_all_data(args.samples)
    
    # 保存数据
    generator.save_data(data, args.output)
    
    # 显示数据统计
    print("\n📊 数据统计:")
    print(f"   数值数据: {len(data['numeric'])} 个样本")
    print(f"   文本数据: {len(data['text'])} 个样本")
    print(f"   图像数据: {len(data['image'])} 个样本")
    print(f"   总数据量: {data['metadata']['total_samples']} 个样本")


if __name__ == "__main__":
    main()
