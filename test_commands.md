# FreDEA 综合测试命令

## 修复后的架构特性
- ✅ TEA: 只在时间维度 T 操作（参数量减少 98%）
- ✅ ChannelMLP: 只在通道维度 N 操作（简化 MoE 结构）
- ✅ Embedding: 简单参数向量（实验证明最优）

---

## ETTm1 数据集

### 预测长度 96
```bash
python run_longExp.py --data ETTm1 --data_path ETTm1.csv --model FreDEA --model_id ETTm1_96_96_Fixed --enc_in 7 --dec_in 7 --c_out 7 --seq_len 96 --pred_len 96 --d_model 128 --d_ff 256 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --rev_affine 1 --train_epochs 20 --batch_size 32 --learning_rate 0.001 --patience 5 --dropout 0.05 --itr 1
```

### 预测长度 192
```bash
python run_longExp.py --data ETTm1 --data_path ETTm1.csv --model FreDEA --model_id ETTm1_96_192_Fixed --enc_in 7 --dec_in 7 --c_out 7 --seq_len 96 --pred_len 192 --d_model 128 --d_ff 256 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --rev_affine 1 --train_epochs 20 --batch_size 32 --learning_rate 0.001 --patience 5 --dropout 0.05 --itr 1
```

### 预测长度 336
```bash
python run_longExp.py --data ETTm1 --data_path ETTm1.csv --model FreDEA --model_id ETTm1_96_336_Fixed --enc_in 7 --dec_in 7 --c_out 7 --seq_len 96 --pred_len 336 --d_model 128 --d_ff 256 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --rev_affine 1 --train_epochs 20 --batch_size 32 --learning_rate 0.001 --patience 5 --dropout 0.1 --itr 1
```

### 预测长度 720
```bash
python run_longExp.py --data ETTm1 --data_path ETTm1.csv --model FreDEA --model_id ETTm1_96_720_Fixed --enc_in 7 --dec_in 7 --c_out 7 --seq_len 96 --pred_len 720 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim 1 --rev_affine 1 --train_epochs 20 --batch_size 128 --learning_rate 0.0003 --patience 5 --dropout 0.3 --itr 1
```

---

## Weather 数据集

### 预测长度 96
```bash
python run_longExp.py --data_path weather.csv --model_id Weather_96_96_Fixed --model FreDEA --data weather --features M --seq_len 96 --pred_len 96 --enc_in 21 --d_model 128 --d_ff 256 --e_layers 2 --dropout 0.1 --rev_affine 1 --memory_size 128 --batch_size 32 --learning_rate 0.0005 --lradj 3 --train_epochs 30 --patience 5 --itr 1 --num_workers 4 --loss mse --bottleneck_dim 4
```

### 预测长度 192
```bash
python run_longExp.py --data_path weather.csv --model_id Weather_96_192_Fixed --model FreDEA --data weather --features M --seq_len 96 --pred_len 192 --enc_in 21 --d_model 128 --d_ff 256 --e_layers 2 --dropout 0.1 --rev_affine 1 --memory_size 128 --batch_size 32 --learning_rate 0.0005 --lradj 3 --train_epochs 30 --patience 5 --itr 1 --num_workers 4 --loss mse --bottleneck_dim 4
```

### 预测长度 336
```bash
python run_longExp.py --data_path weather.csv --model_id Weather_96_336_Fixed --model FreDEA --data weather --features M --seq_len 96 --pred_len 336 --enc_in 21 --d_model 64 --d_ff 256 --e_layers 2 --dropout 0.1 --rev_affine 1 --memory_size 128 --batch_size 32 --learning_rate 0.0005 --lradj 3 --train_epochs 30 --patience 5 --itr 1 --num_workers 4 --loss mse --bottleneck_dim 4
```

### 预测长度 720
```bash
python run_longExp.py --data_path weather.csv --model_id Weather_96_720_Fixed --model FreDEA --data weather --features M --seq_len 96 --pred_len 720 --enc_in 21 --d_model 128 --d_ff 256 --e_layers 2 --dropout 0.1 --rev_affine 1 --memory_size 128 --batch_size 32 --learning_rate 0.0005 --lradj 3 --train_epochs 30 --patience 5 --itr 1 --num_workers 4 --loss mse --bottleneck_dim 4
```

---

## 超参数说明

### ETTm1
- **短期预测 (96, 192)**: 
  - d_model=128, bottleneck_dim=-1 (全连接)
  - dropout=0.05 (小dropout)
- **长期预测 (336, 720)**: 
  - d_model 逐渐减小, bottleneck_dim=1 (瓶颈)
  - dropout 增大到 0.3

### Weather
- **所有长度**: d_model=128/64, bottleneck_dim=4
- **learning_rate**: 0.0005 (较高)
- **lradj**: 3 (指数衰减)

---

## 运行方法

### Linux/Mac
```bash
chmod +x test_FreDEA_comprehensive.sh
./test_FreDEA_comprehensive.sh
```

### Windows (PowerShell)
逐个复制命令运行，或使用以下脚本创建 `.bat` 文件

### 单个测试
直接复制对应的命令到终端运行
