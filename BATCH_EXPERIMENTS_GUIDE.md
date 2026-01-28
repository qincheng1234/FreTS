# FreDEA 批量实验运行指南

## 📋 脚本功能

`run_all_experiments.py` 是一个自动化批量实验脚本，功能包括：

1. ✅ 在7个数据集上自动运行实验
2. ✅ 测试4个预测长度（96, 192, 336, 720）
3. ✅ 自动提取MSE/MAE结果
4. ✅ 生成Markdown格式的汇总表格
5. ✅ 每个数据集使用优化的超参数配置

---

## 🎯 测试矩阵

### 数据集 (7个)

| 数据集 | 通道数 | 特点 |
|--------|--------|------|
| **ETTm1** | 7 | 15分钟电力数据 |
| **ETTm2** | 7 | 15分钟电力数据 |
| **ETTh1** | 7 | 1小时电力数据 |
| **ETTh2** | 7 | 1小时电力数据 |
| **Exchange** | 8 | 汇率数据，高噪声 |
| **Weather** | 21 | 气象数据，强周期性 |
| **Electricity** | 321 | 电力负载数据，高维度 |

### 预测长度 (4个)

- 96步（短期）
- 192步（中期）
- 336步（中长期）
- 720步（长期）

**总实验数**: 7 × 4 = **28个实验**

---

## 🚀 快速开始

### 方法1: 运行全部实验

```bash
python run_all_experiments.py
```

**预计耗时**: 约 10-15 小时（取决于GPU性能）

**输出文件**: `FreDEA_Results_YYYYMMDD_HHMMSS.md`

---

### 方法2: 测试单个数据集

如果您想先测试某个数据集，可以修改脚本中的数据集列表，例如只测试ETTm1：

```python
# 在 main() 函数中修改
for dataset in ['ETTm1']:  # 只测试ETTm1
    for pred_len in PRED_LENS:
        ...
```

---

## 📊 输出示例

运行完成后会生成类似下面的汇总表格：

### 主要结果 (MSE)

| 数据集 | 96 | 192 | 336 | 720 |
|--------|---------|---------|---------|---------|
| ETTm1 | 0.3186 | 0.3654 | 0.4192 | 0.4698 |
| ETTm2 | 0.1821 | 0.2458 | 0.3152 | 0.4187 |
| ETTh1 | 0.3820 | 0.4125 | 0.4485 | 0.4936 |
| ETTh2 | 0.2914 | 0.3748 | 0.4226 | 0.4752 |
| exchange | 0.0898 | 0.1056 | 0.1875 | 0.2764 |
| weather | 0.1547 | 0.1989 | 0.2516 | 0.3241 |
| electricity | 0.1432 | 0.1678 | 0.1962 | 0.2541 |

### 主要结果 (MAE)

| 数据集 | 96 | 192 | 336 | 720 |
|--------|---------|---------|---------|---------|
| ETTm1 | 0.3577 | 0.3856 | 0.4125 | 0.4440 |
| ... | ... | ... | ... | ... |

### 详细结果

#### ETTm1 - Pred 96
- **MSE**: 0.318623
- **MAE**: 0.357700
- **状态**: ✅ 成功

...

---

## ⚙️ 超参数配置说明

脚本为每个数据集的每个预测长度都配置了优化的超参数：

### ETTm1 配置示例

```python
96步: d_model=128, bottleneck=-1, dropout=0.05, lr=0.001
192步: d_model=128, bottleneck=-1, dropout=0.05, lr=0.001  
336步: d_model=128, bottleneck=-1, dropout=0.1, lr=0.001
720步: d_model=64, bottleneck=1, dropout=0.3, lr=0.0003
```

**配置原则**:
- 短期预测：高维度 + 低dropout
- 长期预测：低维度 + 高dropout
- 低通道数据：全连接投影（bottleneck=-1）
- 高通道数据：瓶颈投影（bottleneck=2~4）

---

## 🛠️ 自定义配置

### 修改预测长度

```python
# 在脚本开头修改
PRED_LENS = [96, 192]  # 只测试96和192
```

### 修改训练轮数

```python
COMMON_PARAMS = {
    'train_epochs': 10,  # 改为10轮（快速测试）
    'patience': 3,
}
```

### 添加新数据集

```python
DATASET_CONFIGS['your_dataset'] = {
    'data_path': 'your_data.csv',
    'enc_in': 10,  # 通道数
    'configs': {
        96: {
            'd_model': 128,
            'd_ff': 256,
            # ... 其他参数
        },
        # ... 其他预测长度
    }
}
```

---

## 📝 结果提取说明

脚本会自动从训练日志中提取：

1. **MSE** (Mean Squared Error)
2. **MAE** (Mean Absolute Error)

提取模式：
```python
mse_pattern = r'mse[:\s]+([0-9.]+)'
mae_pattern = r'mae[:\s]+([0-9.]+)'
```

如果您的输出格式不同，可能需要调整正则表达式。

---

## ⚠️ 注意事项

### 1. 数据集路径

确保以下文件存在于 `./dataset/` 目录：
- ETTm1.csv, ETTm2.csv, ETTh1.csv, ETTh2.csv
- exchange_rate.csv
- weather.csv
- electricity.csv

### 2. GPU内存

- Electricity (321通道) 可能需要较大GPU内存
- 如果OOM，可降低batch_size或d_model

### 3. 训练时间

单个实验时间估计：
- ETTm1/ETTm2/ETTh1/ETTh2: ~15-20分钟
- Exchange: ~10-15分钟
- Weather: ~30-40分钟（30 epochs）
- Electricity: ~40-60分钟（321通道）

总计：10-15小时

### 4. 中断恢复

如果脚本中断，可以：
1. 检查已生成的部分结果
2. 修改脚本跳过已完成的实验
3. 手动合并结果

---

## 🔧 故障排查

### 问题1: 结果提取失败

**症状**: "❌ 结果提取失败"

**解决**:
1. 检查输出格式是否为 `mse:0.xxx, mae:0.xxx`
2. 查看完整的训练日志
3. 调整正则表达式匹配模式

### 问题2: 实验超时

**症状**: "❌ 实验超时（2小时）"

**解决**:
1. 增加timeout参数（默认7200秒）
2. 检查是否有死循环
3. 降低train_epochs

### 问题3: GPU内存不足

**症状**: CUDA out of memory

**解决**:
```python
# 降低batch_size
'batch_size': 8,  # 原来是16或32

# 或降低模型维度
'd_model': 64,   # 原来是128
```

---

## 📊 后续分析

得到结果后，您可以：

1. **对比SOTA模型**
   - 与DLinear/NLinear对比
   - 与Transformer系列对比

2. **可视化分析**
   - 绘制折线图（MSE vs 预测长度）
   - 热力图（数据集 × 预测长度）

3. **消融实验**
   - 基于最优配置测试各组件贡献
   - 使用 `--ablation_freq 1` 等参数

4. **论文撰写**
   - 直接使用生成的Markdown表格
   - 补充分析和讨论

---

## 🎓 论文建议

### 实验章节结构

```markdown
## 4. Experiments

### 4.1 Experimental Setup
- Datasets: 7个（ETT×4, Exchange, Weather, Electricity）
- Horizons: 96, 192, 336, 720
- Metrics: MSE, MAE
- Implementation: PyTorch 1.x, NVIDIA GPU

### 4.2 Main Results
表格：FreDEA vs SOTA模型

### 4.3 Ablation Study  
表格：W/O Freq, W/O TEA, W/O CEA

### 4.4 Hyperparameter Analysis
- memory_size影响
- bottleneck_dim策略
```

---

## 📞 支持

如有问题，请检查：
1. Python版本 >= 3.7
2. PyTorch已正确安装
3. 数据集路径正确
4. GPU可用（`torch.cuda.is_available()`）

祝实验顺利！🚀
