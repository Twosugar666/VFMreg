#!/usr/bin/env python3
"""
脑磁配准算法演示脚本
====================

展示基于单目相机的端到端脑磁配准系统核心算法流程

主要功能：
1. 头部轮廓分割演示
2. NeRF隐式配准优化
3. VFMReg端到端配准
4. 性能对比分析
5. 实时演示界面

使用说明：
    python brain_registration_demo.py --mode demo      # 基础演示
    python brain_registration_demo.py --mode realtime  # 实时演示
    python brain_registration_demo.py --mode benchmark  # 性能测试
"""

import argparse
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# 如果没有matplotlib，使用纯文本输出
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib不可用，将使用文本输出模式")


class HeadSegmentationDemo:
    """头部轮廓分割演示类"""
    
    def __init__(self):
        self.model_name = "YOLOv8n-seg + Qwen2.5-VL LoRA"
        self.edge_enhance = True
        self.multi_scale_loss = True
    
    def simulate_segmentation(self, image_path: str) -> Dict:
        """模拟头部分割过程"""
        print(f"🔍 开始头部轮廓分割: {image_path}")
        
        # 使用模拟数据（不依赖cv2读取真实图像）
        height, width = 480, 640
        image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # 生成模拟分割掩码（椭圆形状模拟头部）
        center_x, center_y = width // 2, height // 2
        radius_x, radius_y = width // 3, height // 3
        
        mask = np.zeros((height, width), dtype=np.uint8)
        y, x = np.ogrid[:height, :width]
        mask_ellipse = ((x - center_x) ** 2 / radius_x ** 2 + 
                        (y - center_y) ** 2 / radius_y ** 2) <= 1
        mask[mask_ellipse] = 255
        
        # 模拟边缘增强效果（使用numpy替代cv2）
        if self.edge_enhance:
            # 简单的边缘检测替代Canny
            edges = np.abs(np.gradient(mask.astype(float))[0]) + np.abs(np.gradient(mask.astype(float))[1])
            edges = (edges > 0).astype(np.uint8) * 255
            # 加权融合
            mask = np.clip(mask * 0.8 + edges * 0.2, 0, 255).astype(np.uint8)
        
        # 计算分割指标
        iou = np.random.uniform(0.85, 0.95)  # 模拟IoU
        accuracy = np.random.uniform(0.92, 0.98)  # 模拟准确率
        
        return {
            'image': image,
            'mask': mask,
            'iou': iou,
            'accuracy': accuracy,
            'processing_time': np.random.uniform(0.01, 0.03)  # 10-30ms
        }
    
    def visualize_segmentation(self, result: Dict):
        """可视化分割结果"""
        if HAS_MATPLOTLIB:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 原始图像（使用RGB格式）
            axes[0].imshow(result['image'])
            axes[0].set_title('原始图像')
            axes[0].axis('off')
            
            # 分割掩码
            axes[1].imshow(result['mask'], cmap='gray')
            axes[1].set_title('分割掩码')
            axes[1].axis('off')
            
            # 叠加效果
            overlay = result['image'].copy()
            overlay[result['mask'] > 0] = [0, 255, 0]  # 绿色叠加
            axes[2].imshow(overlay)
            axes[2].set_title('分割结果叠加')
            axes[2].axis('off')
            
            plt.suptitle(f'头部轮廓分割演示 (IoU: {result["iou"]:.3f}, '
                         f'准确率: {result["accuracy"]:.3f})')
            plt.tight_layout()
            plt.show()
        else:
            # 文本输出模式
            print(f"📊 头部轮廓分割结果:")
            print(f"   • IoU: {result['iou']:.3f}")
            print(f"   • 准确率: {result['accuracy']:.3f}")
            print(f"   • 处理时间: {result['processing_time']*1000:.1f}ms")
            print(f"   • 图像尺寸: {result['image'].shape}")
            print(f"   • 分割区域像素数: {np.sum(result['mask'] > 0)}")


class NeRFRegistrationDemo:
    """NeRF隐式配准演示类"""
    
    def __init__(self):
        self.method_name = "NeRF隐式配准"
        self.optimization_steps = 100
    
    def simulate_nerf_registration(self, images: List[np.ndarray]) -> Dict:
        """模拟NeRF配准过程"""
        print(f"🧠 开始NeRF隐式配准: {len(images)}张图像")
        
        # 模拟NeRF优化过程
        translation_errors = []
        rotation_errors = []
        
        # 模拟优化过程
        for step in range(self.optimization_steps):
            # 模拟误差下降过程
            t_error = max(0.1, 5.0 - step * 0.05)  # 平移误差从5mm降到0.1mm
            r_error = max(0.1, 5.0 - step * 0.05)  # 旋转误差从5°降到0.1°
            translation_errors.append(t_error)
            rotation_errors.append(r_error)
        
        final_translation = translation_errors[-1]
        final_rotation = rotation_errors[-1]
        
        return {
            'translation_errors': translation_errors,
            'rotation_errors': rotation_errors,
            'final_translation_error': final_translation,
            'final_rotation_error': final_rotation,
            'optimization_time': np.random.uniform(2.0, 5.0)  # 2-5秒
        }
    
    def visualize_optimization(self, result: Dict):
        """可视化优化过程"""
        if HAS_MATPLOTLIB:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # 平移误差收敛曲线
            axes[0].plot(result['translation_errors'])
            axes[0].set_xlabel('优化步数')
            axes[0].set_ylabel('平移误差 (mm)')
            axes[0].set_title('平移误差收敛曲线')
            axes[0].grid(True)
            
            # 旋转误差收敛曲线
            axes[1].plot(result['rotation_errors'])
            axes[1].set_xlabel('优化步数')
            axes[1].set_ylabel('旋转误差 (°)')
            axes[1].set_title('旋转误差收敛曲线')
            axes[1].grid(True)
            
            plt.suptitle(f'NeRF隐式配准优化过程 (最终误差: {result["final_translation_error"]:.1f}mm, '
                         f'{result["final_rotation_error"]:.1f}°)')
            plt.tight_layout()
            plt.show()
        else:
            # 文本输出模式
            print(f"📈 NeRF隐式配准优化结果:")
            print(f"   • 最终平移误差: {result['final_translation_error']:.1f}mm")
            print(f"   • 最终旋转误差: {result['final_rotation_error']:.1f}°")
            print(f"   • 优化时间: {result['optimization_time']:.1f}s")
            print(f"   • 优化步数: {len(result['translation_errors'])}")
            print(f"   • 初始误差: {result['translation_errors'][0]:.1f}mm → 最终误差: {result['translation_errors'][-1]:.1f}mm")


class VFMRegDemo:
    """VFMReg端到端配准演示类"""
    
    def __init__(self):
        self.model_name = "VFMReg端到端框架"
        self.inference_time = 0.015  # 15ms
    
    def simulate_vfm_registration(self, image_sequence: List[np.ndarray]) -> Dict:
        """模拟VFMReg配准过程"""
        print(f"⚡ 开始VFMReg端到端配准: {len(image_sequence)}帧序列")
        
        # 模拟实时配准
        processing_times = []
        translation_results = []
        rotation_results = []
        
        for i, img in enumerate(image_sequence):
            # 模拟处理时间
            proc_time = np.random.normal(self.inference_time, 0.002)
            processing_times.append(proc_time)
            
            # 模拟配准结果
            t_error = np.random.normal(0.6, 0.1)  # 0.6mm误差
            r_error = np.random.normal(0.7, 0.1)  # 0.7°误差
            translation_results.append(t_error)
            rotation_results.append(r_error)
        
        return {
            'processing_times': processing_times,
            'translation_results': translation_results,
            'rotation_results': rotation_results,
            'avg_processing_time': np.mean(processing_times),
            'avg_translation_error': np.mean(translation_results),
            'avg_rotation_error': np.mean(rotation_results)
        }
    
    def visualize_realtime_performance(self, result: Dict):
        """可视化实时性能"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 处理时间分布
        axes[0, 0].hist(result['processing_times'], bins=20, alpha=0.7)
        axes[0, 0].axvline(result['avg_processing_time'], color='red', linestyle='--', label='平均值')
        axes[0, 0].set_xlabel('处理时间 (秒)')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].set_title('处理时间分布')
        axes[0, 0].legend()
        
        # 平移误差分布
        axes[0, 1].hist(result['translation_results'], bins=20, alpha=0.7, color='orange')
        axes[0, 1].axvline(result['avg_translation_error'], color='red', linestyle='--', label='平均值')
        axes[0, 1].set_xlabel('平移误差 (mm)')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].set_title('平移误差分布')
        axes[0, 1].legend()
        
        # 旋转误差分布
        axes[1, 0].hist(result['rotation_results'], bins=20, alpha=0.7, color='green')
        axes[1, 0].axvline(result['avg_rotation_error'], color='red', linestyle='--', label='平均值')
        axes[1, 0].set_xlabel('旋转误差 (°)')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].set_title('旋转误差分布')
        axes[1, 0].legend()
        
        # 实时性能曲线
        frames = range(len(result['processing_times']))
        axes[1, 1].plot(frames, result['processing_times'], label='处理时间')
        axes[1, 1].set_xlabel('帧序号')
        axes[1, 1].set_ylabel('时间 (秒)', color='blue')
        axes[1, 1].tick_params(axis='y', labelcolor='blue')
        
        ax2 = axes[1, 1].twinx()
        ax2.plot(frames, result['translation_results'], color='orange', label='平移误差')
        ax2.plot(frames, result['rotation_results'], color='green', label='旋转误差')
        ax2.set_ylabel('误差值', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        axes[1, 1].set_title('实时性能监控')
        fig.legend(loc='upper right')
        
        plt.suptitle(f'VFMReg端到端配准性能 (平均处理时间: {result["avg_processing_time"]*1000:.1f}ms)')
        plt.tight_layout()
        plt.show()


class PerformanceComparison:
    """性能对比分析类"""
    
    def __init__(self):
        self.methods = {
            '传统ICP': {'translation': 2.1, 'rotation': 2.3, 'time': 0.5},
            'NeRF隐式配准': {'translation': 1.2, 'rotation': 0.9, 'time': 3.0},
            'VFMReg(合成)': {'translation': 0.5, 'rotation': 0.6, 'time': 0.015},
            'VFMReg(真实)': {'translation': 0.6, 'rotation': 0.7, 'time': 0.02}
        }
    
    def create_comparison_chart(self):
        """创建性能对比图表"""
        methods = list(self.methods.keys())
        translation_errors = [self.methods[m]['translation'] for m in methods]
        rotation_errors = [self.methods[m]['rotation'] for m in methods]
        processing_times = [self.methods[m]['time'] * 1000 for m in methods]  # 转换为ms
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 平移误差对比
        bars1 = axes[0].bar(methods, translation_errors, color=['red', 'orange', 'green', 'blue'])
        axes[0].set_title('平移误差对比 (mm)')
        axes[0].set_ylabel('误差 (mm)')
        axes[0].tick_params(axis='x', rotation=45)
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom')
        
        # 旋转误差对比
        bars2 = axes[1].bar(methods, rotation_errors, color=['red', 'orange', 'green', 'blue'])
        axes[1].set_title('旋转误差对比 (°)')
        axes[1].set_ylabel('误差 (°)')
        axes[1].tick_params(axis='x', rotation=45)
        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom')
        
        # 处理时间对比
        bars3 = axes[2].bar(methods, processing_times, color=['red', 'orange', 'green', 'blue'])
        axes[2].set_title('处理时间对比 (ms)')
        axes[2].set_ylabel('时间 (ms)')
        axes[2].tick_params(axis='x', rotation=45)
        for bar in bars3:
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom')
        
        plt.suptitle('脑磁配准算法性能对比分析')
        plt.tight_layout()
        plt.show()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='脑磁配准算法演示')
    parser.add_argument('--mode', choices=['demo', 'realtime', 'benchmark'], 
                       default='demo', help='演示模式')
    parser.add_argument('--image_path', type=str, default='', 
                       help='测试图像路径')
    
    args = parser.parse_args()
    
    print("🧠 脑磁配准算法演示系统")
    print("=" * 50)
    
    # 初始化演示类
    seg_demo = HeadSegmentationDemo()
    nerf_demo = NeRFRegistrationDemo()
    vfm_demo = VFMRegDemo()
    perf_comp = PerformanceComparison()
    
    if args.mode == 'demo':
        print("📊 基础演示模式")
        
        # 头部轮廓分割演示
        print("\n1. 头部轮廓分割演示")
        seg_result = seg_demo.simulate_segmentation(args.image_path)
        seg_demo.visualize_segmentation(seg_result)
        
        # NeRF配准演示
        print("\n2. NeRF隐式配准演示")
        images = [np.random.randint(0, 255, (480, 640, 3)) for _ in range(5)]
        nerf_result = nerf_demo.simulate_nerf_registration(images)
        nerf_demo.visualize_optimization(nerf_result)
        
        # VFMReg演示
        print("\n3. VFMReg端到端配准演示")
        image_seq = [np.random.randint(0, 255, (480, 640, 3)) for _ in range(100)]
        vfm_result = vfm_demo.simulate_vfm_registration(image_seq)
        vfm_demo.visualize_realtime_performance(vfm_result)
        
        # 性能对比
        print("\n4. 性能对比分析")
        perf_comp.create_comparison_chart()
        
    elif args.mode == 'realtime':
        print("⚡ 实时演示模式")
        # 这里可以添加实时视频流处理代码
        print("实时演示功能需要摄像头支持，当前为模拟演示")
        
    elif args.mode == 'benchmark':
        print("📈 性能测试模式")
        # 这里可以添加性能基准测试代码
        print("性能测试功能需要真实数据，当前为模拟演示")
    
    print("\n✅ 演示完成！")
    print("\n💡 研究亮点总结：")
    print("• 头部轮廓分割：高精度、高鲁棒性")
    print("• NeRF隐式配准：无需显式三维重建")
    print("• VFMReg框架：端到端、实时处理")
    print("• 综合性能：亚毫米级精度 + 20ms延迟")


if __name__ == "__main__":
    main()
