# FreDEA 快速测试结果

生成时间: 2026-01-25 09:58:40

## electricity_96
```bash
python -u run_longExp.py --data electricity --data_path electricity.csv --model FreDEA --model_id electricity_96_96 --seq_len 96 --pred_len 96 --enc_in 321 --dec_in 321 --c_out 321 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim 2 --dropout 0.1 --batch_size 16 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.1550777554512024, mae:0.2537936866283417

## electricity_192
```bash
python -u run_longExp.py --data electricity --data_path electricity.csv --model FreDEA --model_id electricity_96_192 --seq_len 96 --pred_len 192 --enc_in 321 --dec_in 321 --c_out 321 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim 2 --dropout 0.1 --batch_size 16 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.17057783901691437, mae:0.2659030258655548

## electricity_336
```bash
python -u run_longExp.py --data electricity --data_path electricity.csv --model FreDEA --model_id electricity_96_336 --seq_len 96 --pred_len 336 --enc_in 321 --dec_in 321 --c_out 321 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim 2 --dropout 0.15 --batch_size 16 --learning_rate 0.0005 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.19308701157569885, mae:0.2869762182235718

## electricity_720
```bash
python -u run_longExp.py --data electricity --data_path electricity.csv --model FreDEA --model_id electricity_96_720 --seq_len 96 --pred_len 720 --enc_in 321 --dec_in 321 --c_out 321 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim 2 --dropout 0.15 --batch_size 16 --learning_rate 0.0005 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.22878305613994598, mae:0.31784868240356445

## exchange_96
```bash
python -u run_longExp.py --data exchange --data_path exchange_rate.csv --model FreDEA --model_id exchange_96_96 --seq_len 96 --pred_len 96 --enc_in 8 --dec_in 8 --c_out 8 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 32 --bottleneck_dim -1 --dropout 0.0 --batch_size 32 --learning_rate 0.0001 --lradj 3 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.09140665084123611, mae:0.21150332689285278

## exchange_192
```bash
python -u run_longExp.py --data exchange --data_path exchange_rate.csv --model FreDEA --model_id exchange_96_192 --seq_len 96 --pred_len 192 --enc_in 8 --dec_in 8 --c_out 8 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 32 --bottleneck_dim -1 --dropout 0.0 --batch_size 32 --learning_rate 0.0001 --lradj 3 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.18898621201515198, mae:0.3096015453338623

## exchange_336
```bash
python -u run_longExp.py --data exchange --data_path exchange_rate.csv --model FreDEA --model_id exchange_96_336 --seq_len 96 --pred_len 336 --enc_in 8 --dec_in 8 --c_out 8 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 32 --bottleneck_dim 2 --dropout 0.1 --batch_size 32 --learning_rate 0.0001 --lradj 3 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.35234978795051575, mae:0.42960357666015625

## exchange_720
```bash
python -u run_longExp.py --data exchange --data_path exchange_rate.csv --model FreDEA --model_id exchange_96_720 --seq_len 96 --pred_len 720 --enc_in 8 --dec_in 8 --c_out 8 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 32 --bottleneck_dim 2 --dropout 0.1 --batch_size 32 --learning_rate 0.0001 --lradj 3 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.8674256801605225, mae:0.7014104127883911

## ETTh1_96
```bash
python -u run_longExp.py --data ETTh1 --data_path ETTh1.csv --model FreDEA --model_id ETTh1_96_96 --seq_len 96 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --dropout 0.0 --batch_size 32 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.38633668422698975, mae:0.4042598605155945

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
python -u run_longExp.py --data ETTh1 --data_path ETTh1.csv --model FreDEA --model_id ETTh1_96_720 --seq_len 96 --pred_len 720 --enc_in 7 --dec_in 7 --c_out 7 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim 1 --dropout 0.25 --batch_size 32 --learning_rate 0.0003 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.491559237241745, mae:0.4745498299598694

## ETTh2_96
```bash
python -u run_longExp.py --data ETTh2 --data_path ETTh2.csv --model FreDEA --model_id ETTh2_96_96 --seq_len 96 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --dropout 0.0 --batch_size 32 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.31095990538597107, mae:0.35911622643470764

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
python -u run_longExp.py --data ETTh2 --data_path ETTh2.csv --model FreDEA --model_id ETTh2_96_720 --seq_len 96 --pred_len 720 --enc_in 7 --dec_in 7 --c_out 7 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim 1 --dropout 0.25 --batch_size 32 --learning_rate 0.0003 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.4463173449039459, mae:0.4556024372577667

## ETTm1_96
```bash
python -u run_longExp.py --data ETTm1 --data_path ETTm1.csv --model FreDEA --model_id ETTm1_96_96 --seq_len 96 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 256 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --dropout 0.05 --batch_size 32 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.31862396001815796, mae:0.35769930481910706

## ETTm1_192
```bash
python -u run_longExp.py --data ETTm1 --data_path ETTm1.csv --model FreDEA --model_id ETTm1_96_192 --seq_len 96 --pred_len 192 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 256 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --dropout 0.05 --batch_size 32 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.3676345646381378, mae:0.38409173488616943

## ETTm1_336
```bash
python -u run_longExp.py --data ETTm1 --data_path ETTm1.csv --model FreDEA --model_id ETTm1_96_336 --seq_len 96 --pred_len 336 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 256 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --dropout 0.1 --batch_size 32 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.400520384311676, mae:0.4078785181045532

## ETTm1_720
```bash
python -u run_longExp.py --data ETTm1 --data_path ETTm1.csv --model FreDEA --model_id ETTm1_96_720 --seq_len 96 --pred_len 720 --enc_in 7 --dec_in 7 --c_out 7 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim 2 --dropout 0.3 --batch_size 32 --learning_rate 0.0003 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.4659935235977173, mae:0.44497552514076233

## ETTm2_96
```bash
python -u run_longExp.py --data ETTm2 --data_path ETTm2.csv --model FreDEA --model_id ETTm2_96_96 --seq_len 96 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 256 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --dropout 0.05 --batch_size 32 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.17643395066261292, mae:0.26066771149635315

## ETTm2_192
```bash
python -u run_longExp.py --data ETTm2 --data_path ETTm2.csv --model FreDEA --model_id ETTm2_96_192 --seq_len 96 --pred_len 192 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 256 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --dropout 0.05 --batch_size 32 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.2443224936723709, mae:0.30623188614845276

## ETTm2_336
```bash
python -u run_longExp.py --data ETTm2 --data_path ETTm2.csv --model FreDEA --model_id ETTm2_96_336 --seq_len 96 --pred_len 336 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 256 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --dropout 0.1 --batch_size 32 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.3048055171966553, mae:0.3451597988605499

## ETTm2_720
```bash
python -u run_longExp.py --data ETTm2 --data_path ETTm2.csv --model FreDEA --model_id ETTm2_96_720 --seq_len 96 --pred_len 720 --enc_in 7 --dec_in 7 --c_out 7 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim 1 --dropout 0.3 --batch_size 32 --learning_rate 0.0003 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4
```
**结果**: mse:0.39966830611228943, mae:0.39913713932037354
