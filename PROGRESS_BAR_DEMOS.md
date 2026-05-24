# Python 进度条演示总结

## 📊 两个进度条演示脚本

### 1. 合成数据生成器 (`synthetic_data_generator.py`)

**用途**: 模拟生成实验数据，展示基础进度条用法

**特色功能**:
- ✅ 三种数据类型生成（数值、文本、图像）
- ✅ 独立的进度条显示
- ✅ 实时统计信息
- ✅ 数据保存和验证

**运行示例**:
```bash
# 生成 200 个样本
python synthetic_data_generator.py --samples 200

# 生成 1000 个样本到指定目录
python synthetic_data_generator.py --samples 1000 --output my_data
```

**输出效果**:
```
🚀 开始生成 200 个样本的合成数据...
--------------------------------------------------
生成数值数据: 100%|████████████████████| 200/200 [00:00<00:00, 934.99样本/s]
生成文本数据: 100%|████████████████████| 200/200 [00:00<00:00, 481.52样本/s]
生成图像数据: 100%|████████████████████| 200/200 [00:01<00:00, 194.26图像/s]
--------------------------------------------------
✅ 数据生成完成！
📊 总样本数: 600
⏱️  耗时: 1.66 秒
📈 平均速度: 360.7 样本/秒
```

### 2. 高级进度条演示 (`advanced_progress_demo.py`)

**用途**: 展示 tqdm 库的各种高级用法

**特色功能**:
- 🔄 **嵌套进度条** - 主任务 + 子任务层次结构
- 🎨 **自定义样式** - 简约、详细统计等不同风格
- ⚡ **多线程进度** - 并发处理的进度显示
- 📈 **实时统计** - 动态更新损失、准确率等指标
- 🔧 **完整流程** - 模拟完整的数据处理 pipeline

**运行示例**:
```bash
python advanced_progress_demo.py
```

**演示内容**:
1. **基础进度条** - 动态更新描述和进度
2. **嵌套进度条** - 轮次 → 批次 → 样本的层次结构
3. **自定义样式** - 简约风格和详细统计两种样式
4. **多线程处理** - 并发任务的进度显示
5. **数据处理流程** - 加载 → 清洗 → 特征 → 训练 → 评估
6. **实时统计信息** - 训练损失和准确率的实时更新

## 🎯 进度条使用场景

### 数据预处理
```python
# 文件批量处理
for file in tqdm(files, desc="处理文件"):
    process_file(file)

# 数据清洗
for row in tqdm(data_rows, desc="清洗数据"):
    clean_row(row)
```

### 模型训练
```python
# 训练轮次
for epoch in trange(epochs, desc="训练轮次"):
    # 训练批次
    for batch in trange(batches, desc="批次训练", leave=False):
        train_batch(batch)
```

### API 调用
```python
# 批量 API 请求
results = []
for url in tqdm(urls, desc="API 调用"):
    result = call_api(url)
    results.append(result)
```

## 🔧 常用进度条参数

### 基础参数
- `desc`: 进度条描述文字
- `total`: 总任务数
- `unit`: 单位（样本、文件、图像等）
- `leave`: 完成后是否保留进度条

### 样式参数
- `bar_format`: 自定义进度条格式
- `ncols`: 进度条宽度
- `colour`: 颜色设置

### 统计参数
- `set_postfix()`: 动态更新后缀信息
- `set_description()`: 动态更新描述
- `format_dict`: 获取进度条统计信息

## 💡 最佳实践

1. **合理设置单位**: 使用有意义的单位（样本、文件、请求等）
2. **动态更新信息**: 实时显示关键指标（损失、准确率、速度等）
3. **嵌套层次清晰**: 主任务和子任务要有明确的层次关系
4. **异常处理**: 确保进度条在异常情况下也能正常关闭
5. **性能考虑**: 避免过于频繁的进度条更新影响性能

## 📁 生成的文件

运行脚本后会生成：
- `synthetic_data/` - 合成数据目录
  - `numeric_data.json` - 数值数据
  - `text_data.json` - 文本数据  
  - `image_data.json` - 图像数据
  - `metadata.json` - 元数据信息

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install tqdm numpy

# 2. 运行基础演示
python synthetic_data_generator.py

# 3. 运行高级演示
python advanced_progress_demo.py

# 4. 集成到你的项目中
from tqdm import tqdm

for item in tqdm(items, desc="处理中"):
    process_item(item)
```

这两个脚本展示了 Python 进度条的完整用法，从基础到高级，可以直接集成到你的数据处理和机器学习项目中。
