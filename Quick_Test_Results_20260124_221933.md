# FreDEA 快速测试结果

生成时间: 2026-01-24 22:19:33

## ETTh1_96
```bash
python -u run_longExp.py --data ETTh1 --data_path ETTh1.csv --model FreDEA --model_id ETTh1_96_96 --seq_len 96 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --dropout 0.1 --batch_size 32 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.3900655210018158, mae:0.4076296389102936

## ETTh1_192
```bash
python -u run_longExp.py --data ETTh1 --data_path ETTh1.csv --model FreDEA --model_id ETTh1_96_192 --seq_len 96 --pred_len 192 --enc_in 7 --dec_in 7 --c_out 7 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --dropout 0.1 --batch_size 32 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.43655499815940857, mae:0.4330449402332306

## ETTh1_336
```bash
python -u run_longExp.py --data ETTh1 --data_path ETTh1.csv --model FreDEA --model_id ETTh1_96_336 --seq_len 96 --pred_len 336 --enc_in 7 --dec_in 7 --c_out 7 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --dropout 0.2 --batch_size 32 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.4792359173297882, mae:0.45320138335227966

## ETTh1_720
```bash
python -u run_longExp.py --data ETTh1 --data_path ETTh1.csv --model FreDEA --model_id ETTh1_96_720 --seq_len 96 --pred_len 720 --enc_in 7 --dec_in 7 --c_out 7 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim 1 --dropout 0.4 --batch_size 32 --learning_rate 0.0003 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.4916341304779053, mae:0.47454872727394104

## ETTh2_96
```bash
python -u run_longExp.py --data ETTh2 --data_path ETTh2.csv --model FreDEA --model_id ETTh2_96_96 --seq_len 96 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --dropout 0.1 --batch_size 32 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.3144701421260834, mae:0.3609086871147156

## ETTh2_192
```bash
python -u run_longExp.py --data ETTh2 --data_path ETTh2.csv --model FreDEA --model_id ETTh2_96_192 --seq_len 96 --pred_len 192 --enc_in 7 --dec_in 7 --c_out 7 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --dropout 0.1 --batch_size 32 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.38750651478767395, mae:0.40536314249038696

## ETTh2_336
```bash
python -u run_longExp.py --data ETTh2 --data_path ETTh2.csv --model FreDEA --model_id ETTh2_96_336 --seq_len 96 --pred_len 336 --enc_in 7 --dec_in 7 --c_out 7 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --dropout 0.2 --batch_size 32 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.44869565963745117, mae:0.44827020168304443

## ETTh2_720
```bash
python -u run_longExp.py --data ETTh2 --data_path ETTh2.csv --model FreDEA --model_id ETTh2_96_720 --seq_len 96 --pred_len 720 --enc_in 7 --dec_in 7 --c_out 7 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim 1 --dropout 0.4 --batch_size 32 --learning_rate 0.0003 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.4440619945526123, mae:0.4544481337070465

## exchange_96
```bash
python -u run_longExp.py --data exchange --data_path exchange_rate.csv --model FreDEA --model_id exchange_96_96 --seq_len 96 --pred_len 96 --enc_in 8 --dec_in 8 --c_out 8 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 32 --bottleneck_dim -1 --dropout 0.5 --batch_size 32 --learning_rate 0.0001 --lradj 3 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.08798740059137344, mae:0.2072516530752182

## exchange_192
```bash
python -u run_longExp.py --data exchange --data_path exchange_rate.csv --model FreDEA --model_id exchange_96_192 --seq_len 96 --pred_len 192 --enc_in 8 --dec_in 8 --c_out 8 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 32 --bottleneck_dim -1 --dropout 0.5 --batch_size 32 --learning_rate 0.0001 --lradj 3 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.18777866661548615, mae:0.3080890476703644

## exchange_336
```bash
python -u run_longExp.py --data exchange --data_path exchange_rate.csv --model FreDEA --model_id exchange_96_336 --seq_len 96 --pred_len 336 --enc_in 8 --dec_in 8 --c_out 8 --d_model 32 --d_ff 64 --e_layers 1 --memory_size 32 --bottleneck_dim 1 --dropout 0.6 --batch_size 32 --learning_rate 0.0001 --lradj 3 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.3402845859527588, mae:0.4226170480251312

## exchange_720
```bash
python -u run_longExp.py --data exchange --data_path exchange_rate.csv --model FreDEA --model_id exchange_96_720 --seq_len 96 --pred_len 720 --enc_in 8 --dec_in 8 --c_out 8 --d_model 32 --d_ff 64 --e_layers 1 --memory_size 32 --bottleneck_dim 1 --dropout 0.6 --batch_size 32 --learning_rate 0.0001 --lradj 3 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.8558242917060852, mae:0.6969922184944153

## electricity_336
```bash
python -u run_longExp.py --data electricity --data_path electricity.csv --model FreDEA --model_id electricity_96_336 --seq_len 96 --pred_len 336 --enc_in 321 --dec_in 321 --c_out 321 --d_model 64 --d_ff 128 --e_layers 2 --memory_size 128 --bottleneck_dim 4 --dropout 0.3 --batch_size 64 --learning_rate 0.0003 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.21510833501815796, mae:0.3020906448364258

## electricity_720
```bash
python -u run_longExp.py --data electricity --data_path electricity.csv --model FreDEA --model_id electricity_96_720 --seq_len 96 --pred_len 720 --enc_in 321 --dec_in 321 --c_out 321 --d_model 64 --d_ff 128 --e_layers 2 --memory_size 128 --bottleneck_dim 4 --dropout 0.3 --batch_size 64 --learning_rate 0.0003 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.24991373717784882, mae:0.3282195031642914
