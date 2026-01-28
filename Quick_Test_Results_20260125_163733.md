# FreDEA 快速测试结果

生成时间: 2026-01-25 16:37:33

## electricity_96
```bash
python -u run_longExp.py --data electricity --data_path electricity.csv --model FreDEA --model_id electricity_96_96 --seq_len 96 --pred_len 96 --enc_in 321 --dec_in 321 --c_out 321 --d_model 128 --d_ff 256 --e_layers 1 --memory_size 128 --bottleneck_dim 4 --dropout 0.05 --batch_size 16 --learning_rate 0.0005 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.15537694096565247, mae:0.2539386451244354

## electricity_192
```bash
python -u run_longExp.py --data electricity --data_path electricity.csv --model FreDEA --model_id electricity_96_192 --seq_len 96 --pred_len 192 --enc_in 321 --dec_in 321 --c_out 321 --d_model 128 --d_ff 256 --e_layers 1 --memory_size 128 --bottleneck_dim 4 --dropout 0.05 --batch_size 16 --learning_rate 0.0005 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.17092417180538177, mae:0.26565057039260864

## electricity_336
```bash
python -u run_longExp.py --data electricity --data_path electricity.csv --model FreDEA --model_id electricity_96_336 --seq_len 96 --pred_len 336 --enc_in 321 --dec_in 321 --c_out 321 --d_model 128 --d_ff 256 --e_layers 1 --memory_size 128 --bottleneck_dim 4 --dropout 0.1 --batch_size 16 --learning_rate 0.0005 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.19477945566177368, mae:0.2912290394306183

## electricity_720
```bash
python -u run_longExp.py --data electricity --data_path electricity.csv --model FreDEA --model_id electricity_96_720 --seq_len 96 --pred_len 720 --enc_in 321 --dec_in 321 --c_out 321 --d_model 128 --d_ff 256 --e_layers 1 --memory_size 128 --bottleneck_dim 4 --dropout 0.1 --batch_size 16 --learning_rate 0.0005 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.22678226232528687, mae:0.31881511211395264

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
python -u run_longExp.py --data exchange --data_path exchange_rate.csv --model FreDEA --model_id exchange_96_336 --seq_len 96 --pred_len 336 --enc_in 8 --dec_in 8 --c_out 8 --d_model 32 --d_ff 32 --e_layers 1 --memory_size 32 --bottleneck_dim 1 --dropout 0.5 --batch_size 32 --learning_rate 0.0005 --lradj 3 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.3447337746620178, mae:0.4251805543899536

## exchange_720
```bash
python -u run_longExp.py --data exchange --data_path exchange_rate.csv --model FreDEA --model_id exchange_96_720 --seq_len 96 --pred_len 720 --enc_in 8 --dec_in 8 --c_out 8 --d_model 32 --d_ff 32 --e_layers 1 --memory_size 32 --bottleneck_dim 1 --dropout 0.5 --batch_size 32 --learning_rate 0.0005 --lradj 3 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.8966139554977417, mae:0.7119269967079163

## ETTh2_96
```bash
python -u run_longExp.py --data ETTh2 --data_path ETTh2.csv --model FreDEA --model_id ETTh2_96_96 --seq_len 96 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 256 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --dropout 0.05 --batch_size 32 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.31601783633232117, mae:0.35889384150505066

## ETTh2_192
```bash
python -u run_longExp.py --data ETTh2 --data_path ETTh2.csv --model FreDEA --model_id ETTh2_96_192 --seq_len 96 --pred_len 192 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 256 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --dropout 0.05 --batch_size 32 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.3887501060962677, mae:0.40826624631881714

## ETTh2_336
```bash
python -u run_longExp.py --data ETTh2 --data_path ETTh2.csv --model FreDEA --model_id ETTh2_96_336 --seq_len 96 --pred_len 336 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 256 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --dropout 0.1 --batch_size 32 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.44973576068878174, mae:0.45027703046798706

## ETTh2_720
```bash
python -u run_longExp.py --data ETTh2 --data_path ETTh2.csv --model FreDEA --model_id ETTh2_96_720 --seq_len 96 --pred_len 720 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 256 --e_layers 1 --memory_size 64 --bottleneck_dim 1 --dropout 0.35 --batch_size 32 --learning_rate 0.0003 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.4496059715747833, mae:0.456871896982193

## ETTh1_720
```bash
python -u run_longExp.py --data ETTh1 --data_path ETTh1.csv --model FreDEA --model_id ETTh1_96_720 --seq_len 96 --pred_len 720 --enc_in 7 --dec_in 7 --c_out 7 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim 1 --dropout 0.4 --batch_size 32 --learning_rate 0.0003 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.4916341304779053, mae:0.47454872727394104
