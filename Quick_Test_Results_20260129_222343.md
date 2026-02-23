# FreDEA 快速测试结果

生成时间: 2026-01-29 22:23:43

## ETTh2_96
```bash
python -u run_longExp.py --data ETTh2 --data_path ETTh2.csv --model FreDEA --model_id ETTh2_96_96 --seq_len 96 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --dropout 0.15 --batch_size 32 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 8 --itr 1 --num_workers 4 --fusion_init 3.0
```
**结果**: mse:0.2935276925563812, mae:0.3408585786819458

## ETTh2_192
```bash
python -u run_longExp.py --data ETTh2 --data_path ETTh2.csv --model FreDEA --model_id ETTh2_96_192 --seq_len 96 --pred_len 192 --enc_in 7 --dec_in 7 --c_out 7 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --dropout 0.15 --batch_size 32 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 8 --itr 1 --num_workers 4 --fusion_init 3.0
```
**结果**: mse:0.3775820732116699, mae:0.39522436261177063

## ETTh2_336
```bash
python -u run_longExp.py --data ETTh2 --data_path ETTh2.csv --model FreDEA --model_id ETTh2_96_336 --seq_len 96 --pred_len 336 --enc_in 7 --dec_in 7 --c_out 7 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --dropout 0.25 --batch_size 32 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 8 --itr 1 --num_workers 4 --fusion_init 3.0
```
**结果**: mse:0.4319886267185211, mae:0.4393880367279053

## ETTh2_720
```bash
python -u run_longExp.py --data ETTh2 --data_path ETTh2.csv --model FreDEA --model_id ETTh2_96_720 --seq_len 96 --pred_len 720 --enc_in 7 --dec_in 7 --c_out 7 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim 1 --dropout 0.5 --batch_size 32 --learning_rate 0.0003 --lradj type1 --train_epochs 20 --patience 8 --itr 1 --num_workers 4 --fusion_init 3.0
```
**结果**: mse:0.43475934863090515, mae:0.4475705623626709
