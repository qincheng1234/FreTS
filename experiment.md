================================================================================
运行实验: ETTm1 - Pred 96
================================================================================
命令: python -u run_longExp.py --data ETTm1 --data_path ETTm1.csv --model FreDEA --model_id ETTm1_96_96 --seq_len 96 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 256 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --dropout 0.05 --batch_size 32 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4

Args in experiment:
Namespace(is_training=1, train_only=False, model_id='ETTm1_96_96', model='FreDEA', data='ETTm1', root_path='./dataset/', data_path='ETTm1.csv', channel_independence=0, features='M', target='OT', freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=96, individual=False, embed_type=0, enc_in=7, dec_in=7, c_out=7, d_model=128, reg_lambda=0.01, wavelet='haar', n_bins=8, n_heads=8, e_layers=1, d_layers=1, d_ff=256, moving_avg=25, factor=1, distil=True, rev_affine=1, memory_size=64, bottleneck_dim=-1, ablation_freq=0, ablation_tea=0, ablation_cea=0, dropout=0.05, embed='timeF', activation='gelu', output_attention=False, do_predict=False, num_workers=4, itr=1, train_epochs=20, batch_size=32, patience=5, learning_rate=0.001, des='Exp', loss='mse', lradj='type1', use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=False, devices='0,1,2,3', test_flop=False)
Use GPU: cuda:0
>>>>>>>start training : ETTm1_96_96_FreDEA_ETTm1_ftM_sl96_ll48_pl96_dm128_nh8_el1_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0>>>>>>>>>>>>>>>>>>>>>>>>>>
train 34369
val 11425
test 11425

============================================================
TRAINING CONFIGURATION
============================================================
Model:           FreDEA
Dataset:         ETTm1
Seq/Pred Len:    96 -> 96
Batch Size:      32
Learning Rate:   0.001
Train Epochs:    20
Total Params:    126,580
d_model:         128
n_heads:         8
e_layers:        1
============================================================

[Epoch 01] Iter  100/1074 | loss=0.42747 | 0.021s/iter | ETA: 7.3min
[Epoch 01] Iter  200/1074 | loss=0.41836 | 0.010s/iter | ETA: 3.5min
[Epoch 01] Iter  300/1074 | loss=0.24213 | 0.014s/iter | ETA: 4.8min
[Epoch 01] Iter  400/1074 | loss=0.32480 | 0.014s/iter | ETA: 4.9min
[Epoch 01] Iter  500/1074 | loss=0.39104 | 0.016s/iter | ETA: 5.4min
[Epoch 01] Iter  600/1074 | loss=0.38968 | 0.016s/iter | ETA: 5.6min
[Epoch 01] Iter  700/1074 | loss=0.23209 | 0.016s/iter | ETA: 5.6min
[Epoch 01] Iter  800/1074 | loss=0.28548 | 0.016s/iter | ETA: 5.4min
[Epoch 01] Iter  900/1074 | loss=0.26134 | 0.016s/iter | ETA: 5.4min
[Epoch 01] Iter 1000/1074 | loss=0.23415 | 0.016s/iter | ETA: 5.3min

------------------------------------------------------------
[Epoch 01] Summary | Time: 15.9s
Train Loss: 0.313227
Vali  Loss: 0.399344
Test  Loss: 0.331327
Validation loss decreased (inf --> 0.399344).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.001
[Epoch 02] Iter  100/1074 | loss=0.28847 | 0.048s/iter | ETA: 16.4min
[Epoch 02] Iter  200/1074 | loss=0.31970 | 0.012s/iter | ETA: 4.0min
[Epoch 02] Iter  300/1074 | loss=0.23196 | 0.011s/iter | ETA: 3.8min
[Epoch 02] Iter  400/1074 | loss=0.25544 | 0.011s/iter | ETA: 3.8min
[Epoch 02] Iter  500/1074 | loss=0.26694 | 0.011s/iter | ETA: 3.8min
[Epoch 02] Iter  600/1074 | loss=0.27001 | 0.011s/iter | ETA: 3.7min
[Epoch 02] Iter  700/1074 | loss=0.27085 | 0.011s/iter | ETA: 3.7min
[Epoch 02] Iter  800/1074 | loss=0.39086 | 0.013s/iter | ETA: 4.3min
[Epoch 02] Iter  900/1074 | loss=0.24339 | 0.014s/iter | ETA: 4.5min
[Epoch 02] Iter 1000/1074 | loss=0.28558 | 0.015s/iter | ETA: 4.7min

------------------------------------------------------------
[Epoch 02] Summary | Time: 13.4s
Train Loss: 0.265659
Vali  Loss: 0.405715
Test  Loss: 0.327260
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 03] Iter  100/1074 | loss=0.19738 | 0.047s/iter | ETA: 15.1min
[Epoch 03] Iter  200/1074 | loss=0.29726 | 0.011s/iter | ETA: 3.5min
[Epoch 03] Iter  300/1074 | loss=0.26706 | 0.011s/iter | ETA: 3.5min
[Epoch 03] Iter  400/1074 | loss=0.24873 | 0.010s/iter | ETA: 3.3min
[Epoch 03] Iter  500/1074 | loss=0.28611 | 0.011s/iter | ETA: 3.6min
[Epoch 03] Iter  600/1074 | loss=0.30836 | 0.015s/iter | ETA: 4.8min
[Epoch 03] Iter  700/1074 | loss=0.25583 | 0.016s/iter | ETA: 4.8min
[Epoch 03] Iter  800/1074 | loss=0.30929 | 0.016s/iter | ETA: 4.9min
[Epoch 03] Iter  900/1074 | loss=0.30505 | 0.016s/iter | ETA: 4.8min
[Epoch 03] Iter 1000/1074 | loss=0.25822 | 0.016s/iter | ETA: 4.8min

------------------------------------------------------------
[Epoch 03] Summary | Time: 14.6s
Train Loss: 0.249325
Vali  Loss: 0.402923
Test  Loss: 0.321599
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 0.00025
[Epoch 04] Iter  100/1074 | loss=0.25761 | 0.046s/iter | ETA: 14.0min
[Epoch 04] Iter  200/1074 | loss=0.22735 | 0.010s/iter | ETA: 3.1min
[Epoch 04] Iter  300/1074 | loss=0.20757 | 0.011s/iter | ETA: 3.2min
[Epoch 04] Iter  400/1074 | loss=0.25379 | 0.013s/iter | ETA: 3.8min
[Epoch 04] Iter  500/1074 | loss=0.20931 | 0.013s/iter | ETA: 3.8min
[Epoch 04] Iter  600/1074 | loss=0.28361 | 0.013s/iter | ETA: 3.7min
[Epoch 04] Iter  700/1074 | loss=0.23060 | 0.013s/iter | ETA: 3.7min
[Epoch 04] Iter  800/1074 | loss=0.25347 | 0.013s/iter | ETA: 3.7min
[Epoch 04] Iter  900/1074 | loss=0.21350 | 0.014s/iter | ETA: 4.0min
[Epoch 04] Iter 1000/1074 | loss=0.25012 | 0.014s/iter | ETA: 4.1min

------------------------------------------------------------
[Epoch 04] Summary | Time: 13.4s
Train Loss: 0.241423
Vali  Loss: 0.401415
Test  Loss: 0.320287
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 0.000125
[Epoch 05] Iter  100/1074 | loss=0.18135 | 0.048s/iter | ETA: 13.5min
[Epoch 05] Iter  200/1074 | loss=0.21787 | 0.013s/iter | ETA: 3.6min
[Epoch 05] Iter  300/1074 | loss=0.25273 | 0.013s/iter | ETA: 3.7min
[Epoch 05] Iter  400/1074 | loss=0.23636 | 0.013s/iter | ETA: 3.7min
[Epoch 05] Iter  500/1074 | loss=0.21742 | 0.013s/iter | ETA: 3.5min
[Epoch 05] Iter  600/1074 | loss=0.25746 | 0.013s/iter | ETA: 3.5min
[Epoch 05] Iter  700/1074 | loss=0.26881 | 0.013s/iter | ETA: 3.6min
[Epoch 05] Iter  800/1074 | loss=0.23218 | 0.013s/iter | ETA: 3.6min
[Epoch 05] Iter  900/1074 | loss=0.22839 | 0.013s/iter | ETA: 3.5min
[Epoch 05] Iter 1000/1074 | loss=0.20534 | 0.013s/iter | ETA: 3.5min

------------------------------------------------------------
[Epoch 05] Summary | Time: 14.0s
Train Loss: 0.237229
Vali  Loss: 0.396105
Test  Loss: 0.315846
Validation loss decreased (0.399344 --> 0.396105).  Saving model ...
------------------------------------------------------------
Updating learning rate to 6.25e-05
[Epoch 06] Iter  100/1074 | loss=0.24702 | 0.048s/iter | ETA: 12.8min
[Epoch 06] Iter  200/1074 | loss=0.20368 | 0.014s/iter | ETA: 3.6min
[Epoch 06] Iter  300/1074 | loss=0.26079 | 0.014s/iter | ETA: 3.7min
[Epoch 06] Iter  400/1074 | loss=0.24867 | 0.016s/iter | ETA: 4.1min
[Epoch 06] Iter  500/1074 | loss=0.22795 | 0.015s/iter | ETA: 4.0min
[Epoch 06] Iter  600/1074 | loss=0.22110 | 0.016s/iter | ETA: 4.0min
[Epoch 06] Iter  700/1074 | loss=0.28451 | 0.016s/iter | ETA: 4.0min
[Epoch 06] Iter  800/1074 | loss=0.22947 | 0.015s/iter | ETA: 3.9min
[Epoch 06] Iter  900/1074 | loss=0.24625 | 0.015s/iter | ETA: 3.9min
[Epoch 06] Iter 1000/1074 | loss=0.18280 | 0.015s/iter | ETA: 3.9min

------------------------------------------------------------
[Epoch 06] Summary | Time: 16.1s
Train Loss: 0.235111
Vali  Loss: 0.398274
Test  Loss: 0.316482
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 3.125e-05
[Epoch 07] Iter  100/1074 | loss=0.23753 | 0.051s/iter | ETA: 12.7min
[Epoch 07] Iter  200/1074 | loss=0.24234 | 0.015s/iter | ETA: 3.6min
[Epoch 07] Iter  300/1074 | loss=0.23339 | 0.015s/iter | ETA: 3.6min
[Epoch 07] Iter  400/1074 | loss=0.27430 | 0.015s/iter | ETA: 3.6min
[Epoch 07] Iter  500/1074 | loss=0.24426 | 0.014s/iter | ETA: 3.5min
[Epoch 07] Iter  600/1074 | loss=0.24151 | 0.014s/iter | ETA: 3.5min
[Epoch 07] Iter  700/1074 | loss=0.22664 | 0.014s/iter | ETA: 3.5min
[Epoch 07] Iter  800/1074 | loss=0.21680 | 0.015s/iter | ETA: 3.6min
[Epoch 07] Iter  900/1074 | loss=0.24626 | 0.015s/iter | ETA: 3.6min
[Epoch 07] Iter 1000/1074 | loss=0.29114 | 0.015s/iter | ETA: 3.5min

------------------------------------------------------------
[Epoch 07] Summary | Time: 15.9s
Train Loss: 0.233811
Vali  Loss: 0.396442
Test  Loss: 0.315899
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 1.5625e-05
[Epoch 08] Iter  100/1074 | loss=0.22499 | 0.048s/iter | ETA: 11.1min
[Epoch 08] Iter  200/1074 | loss=0.24452 | 0.012s/iter | ETA: 2.7min
[Epoch 08] Iter  300/1074 | loss=0.24960 | 0.013s/iter | ETA: 3.0min
[Epoch 08] Iter  400/1074 | loss=0.18303 | 0.013s/iter | ETA: 3.0min
[Epoch 08] Iter  500/1074 | loss=0.19837 | 0.013s/iter | ETA: 3.0min
[Epoch 08] Iter  600/1074 | loss=0.22937 | 0.013s/iter | ETA: 3.0min
[Epoch 08] Iter  700/1074 | loss=0.22279 | 0.015s/iter | ETA: 3.2min
[Epoch 08] Iter  800/1074 | loss=0.22239 | 0.015s/iter | ETA: 3.3min
[Epoch 08] Iter  900/1074 | loss=0.19315 | 0.015s/iter | ETA: 3.2min
[Epoch 08] Iter 1000/1074 | loss=0.20036 | 0.015s/iter | ETA: 3.2min

------------------------------------------------------------
[Epoch 08] Summary | Time: 14.7s
Train Loss: 0.233213
Vali  Loss: 0.395912
Test  Loss: 0.315983
Validation loss decreased (0.396105 --> 0.395912).  Saving model ...
------------------------------------------------------------
Updating learning rate to 7.8125e-06
[Epoch 09] Iter  100/1074 | loss=0.22470 | 0.048s/iter | ETA: 10.2min
[Epoch 09] Iter  200/1074 | loss=0.21778 | 0.010s/iter | ETA: 2.1min
[Epoch 09] Iter  300/1074 | loss=0.19208 | 0.010s/iter | ETA: 2.1min
[Epoch 09] Iter  400/1074 | loss=0.21081 | 0.010s/iter | ETA: 2.0min
[Epoch 09] Iter  500/1074 | loss=0.26145 | 0.010s/iter | ETA: 2.1min
[Epoch 09] Iter  600/1074 | loss=0.22496 | 0.010s/iter | ETA: 2.1min
[Epoch 09] Iter  700/1074 | loss=0.24352 | 0.010s/iter | ETA: 2.0min
[Epoch 09] Iter  800/1074 | loss=0.32531 | 0.010s/iter | ETA: 2.0min
[Epoch 09] Iter  900/1074 | loss=0.22611 | 0.010s/iter | ETA: 2.0min
[Epoch 09] Iter 1000/1074 | loss=0.25645 | 0.010s/iter | ETA: 2.0min

------------------------------------------------------------
[Epoch 09] Summary | Time: 10.9s
Train Loss: 0.232900
Vali  Loss: 0.396060
Test  Loss: 0.315847
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 3.90625e-06
[Epoch 10] Iter  100/1074 | loss=0.29133 | 0.043s/iter | ETA: 8.4min
[Epoch 10] Iter  200/1074 | loss=0.28958 | 0.013s/iter | ETA: 2.5min
[Epoch 10] Iter  300/1074 | loss=0.22002 | 0.013s/iter | ETA: 2.5min
[Epoch 10] Iter  400/1074 | loss=0.19322 | 0.013s/iter | ETA: 2.5min
[Epoch 10] Iter  500/1074 | loss=0.25894 | 0.013s/iter | ETA: 2.4min
[Epoch 10] Iter  600/1074 | loss=0.22961 | 0.013s/iter | ETA: 2.4min
[Epoch 10] Iter  700/1074 | loss=0.20500 | 0.013s/iter | ETA: 2.4min
[Epoch 10] Iter  800/1074 | loss=0.27780 | 0.013s/iter | ETA: 2.4min
[Epoch 10] Iter  900/1074 | loss=0.26180 | 0.013s/iter | ETA: 2.4min
[Epoch 10] Iter 1000/1074 | loss=0.20744 | 0.013s/iter | ETA: 2.4min

------------------------------------------------------------
[Epoch 10] Summary | Time: 13.9s
Train Loss: 0.232746
Vali  Loss: 0.395814
Test  Loss: 0.315955
Validation loss decreased (0.395912 --> 0.395814).  Saving model ...
------------------------------------------------------------
Updating learning rate to 1.953125e-06
[Epoch 11] Iter  100/1074 | loss=0.20214 | 0.048s/iter | ETA: 8.5min
[Epoch 11] Iter  200/1074 | loss=0.29900 | 0.012s/iter | ETA: 2.1min
[Epoch 11] Iter  300/1074 | loss=0.26499 | 0.012s/iter | ETA: 2.1min
[Epoch 11] Iter  400/1074 | loss=0.25809 | 0.012s/iter | ETA: 2.0min
[Epoch 11] Iter  500/1074 | loss=0.21532 | 0.012s/iter | ETA: 2.0min
[Epoch 11] Iter  600/1074 | loss=0.22765 | 0.012s/iter | ETA: 2.0min
[Epoch 11] Iter  700/1074 | loss=0.28901 | 0.012s/iter | ETA: 1.9min
[Epoch 11] Iter  800/1074 | loss=0.22398 | 0.012s/iter | ETA: 1.9min
[Epoch 11] Iter  900/1074 | loss=0.29257 | 0.012s/iter | ETA: 1.9min
[Epoch 11] Iter 1000/1074 | loss=0.21409 | 0.012s/iter | ETA: 1.9min

------------------------------------------------------------
[Epoch 11] Summary | Time: 12.7s
Train Loss: 0.232680
Vali  Loss: 0.396132
Test  Loss: 0.316092
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 9.765625e-07
[Epoch 12] Iter  100/1074 | loss=0.20191 | 0.045s/iter | ETA: 7.1min
[Epoch 12] Iter  200/1074 | loss=0.20811 | 0.012s/iter | ETA: 1.9min
[Epoch 12] Iter  300/1074 | loss=0.26398 | 0.013s/iter | ETA: 2.1min
[Epoch 12] Iter  400/1074 | loss=0.24443 | 0.014s/iter | ETA: 2.1min
[Epoch 12] Iter  500/1074 | loss=0.22154 | 0.015s/iter | ETA: 2.3min
[Epoch 12] Iter  600/1074 | loss=0.29407 | 0.012s/iter | ETA: 1.7min
[Epoch 12] Iter  700/1074 | loss=0.18257 | 0.011s/iter | ETA: 1.6min
[Epoch 12] Iter  800/1074 | loss=0.21664 | 0.010s/iter | ETA: 1.5min
[Epoch 12] Iter  900/1074 | loss=0.18799 | 0.011s/iter | ETA: 1.5min
[Epoch 12] Iter 1000/1074 | loss=0.32338 | 0.011s/iter | ETA: 1.6min

------------------------------------------------------------
[Epoch 12] Summary | Time: 13.0s
Train Loss: 0.232626
Vali  Loss: 0.395894
Test  Loss: 0.315904
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 4.8828125e-07
[Epoch 13] Iter  100/1074 | loss=0.28008 | 0.044s/iter | ETA: 6.3min
[Epoch 13] Iter  200/1074 | loss=0.18941 | 0.010s/iter | ETA: 1.5min
[Epoch 13] Iter  300/1074 | loss=0.22042 | 0.010s/iter | ETA: 1.4min
[Epoch 13] Iter  400/1074 | loss=0.26900 | 0.010s/iter | ETA: 1.4min
[Epoch 13] Iter  500/1074 | loss=0.20503 | 0.010s/iter | ETA: 1.4min
[Epoch 13] Iter  600/1074 | loss=0.22004 | 0.010s/iter | ETA: 1.4min
[Epoch 13] Iter  700/1074 | loss=0.19968 | 0.010s/iter | ETA: 1.4min
[Epoch 13] Iter  800/1074 | loss=0.22839 | 0.011s/iter | ETA: 1.4min
[Epoch 13] Iter  900/1074 | loss=0.21030 | 0.010s/iter | ETA: 1.3min
[Epoch 13] Iter 1000/1074 | loss=0.21315 | 0.011s/iter | ETA: 1.4min

------------------------------------------------------------
[Epoch 13] Summary | Time: 11.5s
Train Loss: 0.232519
Vali  Loss: 0.395861
Test  Loss: 0.315914
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 2.44140625e-07
[Epoch 14] Iter  100/1074 | loss=0.25974 | 0.045s/iter | ETA: 5.6min
[Epoch 14] Iter  200/1074 | loss=0.22053 | 0.013s/iter | ETA: 1.6min
[Epoch 14] Iter  300/1074 | loss=0.20414 | 0.013s/iter | ETA: 1.6min
[Epoch 14] Iter  400/1074 | loss=0.20464 | 0.013s/iter | ETA: 1.5min
[Epoch 14] Iter  500/1074 | loss=0.22475 | 0.013s/iter | ETA: 1.5min
[Epoch 14] Iter  600/1074 | loss=0.23255 | 0.014s/iter | ETA: 1.6min
[Epoch 14] Iter  700/1074 | loss=0.28389 | 0.014s/iter | ETA: 1.6min
[Epoch 14] Iter  800/1074 | loss=0.19387 | 0.014s/iter | ETA: 1.6min
[Epoch 14] Iter  900/1074 | loss=0.19908 | 0.014s/iter | ETA: 1.6min
[Epoch 14] Iter 1000/1074 | loss=0.27617 | 0.014s/iter | ETA: 1.6min

------------------------------------------------------------
[Epoch 14] Summary | Time: 14.9s
Train Loss: 0.232635
Vali  Loss: 0.395941
Test  Loss: 0.315944
EarlyStopping counter: 4 out of 5
------------------------------------------------------------
Updating learning rate to 1.220703125e-07
[Epoch 15] Iter  100/1074 | loss=0.22303 | 0.049s/iter | ETA: 5.2min
[Epoch 15] Iter  200/1074 | loss=0.22701 | 0.011s/iter | ETA: 1.1min
[Epoch 15] Iter  300/1074 | loss=0.20850 | 0.012s/iter | ETA: 1.2min
[Epoch 15] Iter  400/1074 | loss=0.18878 | 0.012s/iter | ETA: 1.2min
[Epoch 15] Iter  500/1074 | loss=0.22348 | 0.012s/iter | ETA: 1.2min
[Epoch 15] Iter  600/1074 | loss=0.29258 | 0.012s/iter | ETA: 1.2min
[Epoch 15] Iter  700/1074 | loss=0.24043 | 0.012s/iter | ETA: 1.1min
[Epoch 15] Iter  800/1074 | loss=0.24111 | 0.012s/iter | ETA: 1.1min
[Epoch 15] Iter  900/1074 | loss=0.23360 | 0.012s/iter | ETA: 1.1min
[Epoch 15] Iter 1000/1074 | loss=0.25801 | 0.014s/iter | ETA: 1.3min

------------------------------------------------------------
[Epoch 15] Summary | Time: 13.5s
Train Loss: 0.232666
Vali  Loss: 0.395848
Test  Loss: 0.315937
EarlyStopping counter: 5 out of 5
------------------------------------------------------------

************************************************************
EARLY STOPPING TRIGGERED
************************************************************
>>>>>>>testing : ETTm1_96_96_FreDEA_ETTm1_ftM_sl96_ll48_pl96_dm128_nh8_el1_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 11425
mse:0.31654396653175354, mae:0.3584076762199402, rmse:0.5626224279403687
✅ 实验完成: MSE=0.316544, MAE=0.358408

================================================================================
运行实验: ETTm1 - Pred 192
================================================================================
命令: python -u run_longExp.py --data ETTm1 --data_path ETTm1.csv --model FreDEA --model_id ETTm1_96_192 --seq_len 96 --pred_len 192 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 256 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --dropout 0.05 --batch_size 32 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4

Args in experiment:
Namespace(is_training=1, train_only=False, model_id='ETTm1_96_192', model='FreDEA', data='ETTm1', root_path='./dataset/', data_path='ETTm1.csv', channel_independence=0, features='M', target='OT', freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=192, individual=False, embed_type=0, enc_in=7, dec_in=7, c_out=7, d_model=128, reg_lambda=0.01, wavelet='haar', n_bins=8, n_heads=8, e_layers=1, d_layers=1, d_ff=256, moving_avg=25, factor=1, distil=True, rev_affine=1, memory_size=64, bottleneck_dim=-1, ablation_freq=0, ablation_tea=0, ablation_cea=0, dropout=0.05, embed='timeF', activation='gelu', output_attention=False, do_predict=False, num_workers=4, itr=1, train_epochs=20, batch_size=32, patience=5, learning_rate=0.001, des='Exp', loss='mse', lradj='type1', use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=False, devices='0,1,2,3', test_flop=False)
Use GPU: cuda:0
>>>>>>>start training : ETTm1_96_192_FreDEA_ETTm1_ftM_sl96_ll48_pl192_dm128_nh8_el1_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0>>>>>>>>>>>>>>>>>>>>>>>>>>
train 34273
val 11329
test 11329

============================================================
TRAINING CONFIGURATION
============================================================
Model:           FreDEA
Dataset:         ETTm1
Seq/Pred Len:    96 -> 192
Batch Size:      32
Learning Rate:   0.001
Train Epochs:    20
Total Params:    138,964
d_model:         128
n_heads:         8
e_layers:        1
============================================================

[Epoch 01] Iter  100/1071 | loss=0.34216 | 0.021s/iter | ETA: 7.6min
[Epoch 01] Iter  200/1071 | loss=0.39936 | 0.013s/iter | ETA: 4.7min
[Epoch 01] Iter  300/1071 | loss=0.41447 | 0.015s/iter | ETA: 5.1min
[Epoch 01] Iter  400/1071 | loss=0.34521 | 0.015s/iter | ETA: 5.2min
[Epoch 01] Iter  500/1071 | loss=0.30984 | 0.015s/iter | ETA: 5.1min
[Epoch 01] Iter  600/1071 | loss=0.34696 | 0.014s/iter | ETA: 5.0min
[Epoch 01] Iter  700/1071 | loss=0.32792 | 0.014s/iter | ETA: 5.0min
[Epoch 01] Iter  800/1071 | loss=0.30546 | 0.014s/iter | ETA: 5.0min
[Epoch 01] Iter  900/1071 | loss=0.32104 | 0.015s/iter | ETA: 5.0min
[Epoch 01] Iter 1000/1071 | loss=0.29100 | 0.015s/iter | ETA: 5.0min

------------------------------------------------------------
[Epoch 01] Summary | Time: 15.6s
Train Loss: 0.359480
Vali  Loss: 0.520162
Test  Loss: 0.374523
Validation loss decreased (inf --> 0.520162).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.001
[Epoch 02] Iter  100/1071 | loss=0.36311 | 0.050s/iter | ETA: 17.0min
[Epoch 02] Iter  200/1071 | loss=0.31447 | 0.012s/iter | ETA: 3.9min
[Epoch 02] Iter  300/1071 | loss=0.31107 | 0.012s/iter | ETA: 4.1min
[Epoch 02] Iter  400/1071 | loss=0.37487 | 0.012s/iter | ETA: 4.1min
[Epoch 02] Iter  500/1071 | loss=0.25581 | 0.012s/iter | ETA: 4.1min
[Epoch 02] Iter  600/1071 | loss=0.26881 | 0.012s/iter | ETA: 4.0min
[Epoch 02] Iter  700/1071 | loss=0.30645 | 0.012s/iter | ETA: 4.0min
[Epoch 02] Iter  800/1071 | loss=0.37203 | 0.012s/iter | ETA: 4.0min
[Epoch 02] Iter  900/1071 | loss=0.32261 | 0.012s/iter | ETA: 3.9min
[Epoch 02] Iter 1000/1071 | loss=0.26632 | 0.012s/iter | ETA: 3.9min

------------------------------------------------------------
[Epoch 02] Summary | Time: 13.2s
Train Loss: 0.318345
Vali  Loss: 0.525114
Test  Loss: 0.365985
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 03] Iter  100/1071 | loss=0.25875 | 0.045s/iter | ETA: 14.4min
[Epoch 03] Iter  200/1071 | loss=0.30096 | 0.009s/iter | ETA: 2.9min
[Epoch 03] Iter  300/1071 | loss=0.26983 | 0.009s/iter | ETA: 2.8min
[Epoch 03] Iter  400/1071 | loss=0.27087 | 0.009s/iter | ETA: 2.8min
[Epoch 03] Iter  500/1071 | loss=0.28907 | 0.009s/iter | ETA: 2.8min
[Epoch 03] Iter  600/1071 | loss=0.26136 | 0.009s/iter | ETA: 2.8min
[Epoch 03] Iter  700/1071 | loss=0.24103 | 0.009s/iter | ETA: 2.7min
[Epoch 03] Iter  800/1071 | loss=0.28848 | 0.009s/iter | ETA: 2.8min
[Epoch 03] Iter  900/1071 | loss=0.26453 | 0.009s/iter | ETA: 2.7min
[Epoch 03] Iter 1000/1071 | loss=0.29734 | 0.009s/iter | ETA: 2.7min

------------------------------------------------------------
[Epoch 03] Summary | Time: 9.7s
Train Loss: 0.304113
Vali  Loss: 0.510939
Test  Loss: 0.371307
Validation loss decreased (0.520162 --> 0.510939).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.00025
[Epoch 04] Iter  100/1071 | loss=0.32482 | 0.045s/iter | ETA: 13.6min
[Epoch 04] Iter  200/1071 | loss=0.33124 | 0.012s/iter | ETA: 3.7min
[Epoch 04] Iter  300/1071 | loss=0.27843 | 0.014s/iter | ETA: 4.2min
[Epoch 04] Iter  400/1071 | loss=0.27271 | 0.014s/iter | ETA: 4.1min
[Epoch 04] Iter  500/1071 | loss=0.26389 | 0.015s/iter | ETA: 4.3min
[Epoch 04] Iter  600/1071 | loss=0.27502 | 0.013s/iter | ETA: 3.9min
[Epoch 04] Iter  700/1071 | loss=0.31156 | 0.012s/iter | ETA: 3.4min
[Epoch 04] Iter  800/1071 | loss=0.31970 | 0.010s/iter | ETA: 3.0min
[Epoch 04] Iter  900/1071 | loss=0.30202 | 0.011s/iter | ETA: 3.1min
[Epoch 04] Iter 1000/1071 | loss=0.33570 | 0.013s/iter | ETA: 3.7min

------------------------------------------------------------
[Epoch 04] Summary | Time: 13.5s
Train Loss: 0.297018
Vali  Loss: 0.511192
Test  Loss: 0.369227
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 0.000125
[Epoch 05] Iter  100/1071 | loss=0.27691 | 0.049s/iter | ETA: 13.9min
[Epoch 05] Iter  200/1071 | loss=0.31493 | 0.011s/iter | ETA: 3.1min
[Epoch 05] Iter  300/1071 | loss=0.30993 | 0.011s/iter | ETA: 3.1min
[Epoch 05] Iter  400/1071 | loss=0.31047 | 0.011s/iter | ETA: 3.1min
[Epoch 05] Iter  500/1071 | loss=0.32741 | 0.011s/iter | ETA: 3.2min
[Epoch 05] Iter  600/1071 | loss=0.27674 | 0.011s/iter | ETA: 3.1min
[Epoch 05] Iter  700/1071 | loss=0.31436 | 0.013s/iter | ETA: 3.5min
[Epoch 05] Iter  800/1071 | loss=0.29943 | 0.011s/iter | ETA: 3.1min
[Epoch 05] Iter  900/1071 | loss=0.29190 | 0.009s/iter | ETA: 2.4min
[Epoch 05] Iter 1000/1071 | loss=0.31438 | 0.009s/iter | ETA: 2.4min

------------------------------------------------------------
[Epoch 05] Summary | Time: 11.8s
Train Loss: 0.292644
Vali  Loss: 0.517363
Test  Loss: 0.365072
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 6.25e-05
[Epoch 06] Iter  100/1071 | loss=0.34235 | 0.048s/iter | ETA: 12.7min
[Epoch 06] Iter  200/1071 | loss=0.34775 | 0.015s/iter | ETA: 3.9min
[Epoch 06] Iter  300/1071 | loss=0.26000 | 0.015s/iter | ETA: 3.9min
[Epoch 06] Iter  400/1071 | loss=0.29083 | 0.015s/iter | ETA: 3.8min
[Epoch 06] Iter  500/1071 | loss=0.32984 | 0.015s/iter | ETA: 3.8min
[Epoch 06] Iter  600/1071 | loss=0.35369 | 0.015s/iter | ETA: 3.8min
[Epoch 06] Iter  700/1071 | loss=0.27227 | 0.012s/iter | ETA: 3.2min
[Epoch 06] Iter  800/1071 | loss=0.20578 | 0.015s/iter | ETA: 3.7min
[Epoch 06] Iter  900/1071 | loss=0.22817 | 0.013s/iter | ETA: 3.4min
[Epoch 06] Iter 1000/1071 | loss=0.32998 | 0.012s/iter | ETA: 3.0min

------------------------------------------------------------
[Epoch 06] Summary | Time: 15.0s
Train Loss: 0.290099
Vali  Loss: 0.511787
Test  Loss: 0.367068
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 3.125e-05
[Epoch 07] Iter  100/1071 | loss=0.24038 | 0.050s/iter | ETA: 12.5min
[Epoch 07] Iter  200/1071 | loss=0.32079 | 0.013s/iter | ETA: 3.2min
[Epoch 07] Iter  300/1071 | loss=0.26298 | 0.013s/iter | ETA: 3.2min
[Epoch 07] Iter  400/1071 | loss=0.26445 | 0.013s/iter | ETA: 3.2min
[Epoch 07] Iter  500/1071 | loss=0.35951 | 0.012s/iter | ETA: 2.8min
[Epoch 07] Iter  600/1071 | loss=0.27364 | 0.013s/iter | ETA: 3.0min
[Epoch 07] Iter  700/1071 | loss=0.25993 | 0.012s/iter | ETA: 2.9min
[Epoch 07] Iter  800/1071 | loss=0.32435 | 0.012s/iter | ETA: 2.8min
[Epoch 07] Iter  900/1071 | loss=0.20592 | 0.012s/iter | ETA: 2.9min
[Epoch 07] Iter 1000/1071 | loss=0.26866 | 0.012s/iter | ETA: 2.9min

------------------------------------------------------------
[Epoch 07] Summary | Time: 13.5s
Train Loss: 0.289073
Vali  Loss: 0.513605
Test  Loss: 0.365671
EarlyStopping counter: 4 out of 5
------------------------------------------------------------
Updating learning rate to 1.5625e-05
[Epoch 08] Iter  100/1071 | loss=0.29344 | 0.048s/iter | ETA: 11.2min
[Epoch 08] Iter  200/1071 | loss=0.36080 | 0.012s/iter | ETA: 2.8min
[Epoch 08] Iter  300/1071 | loss=0.30833 | 0.014s/iter | ETA: 3.3min
[Epoch 08] Iter  400/1071 | loss=0.26025 | 0.015s/iter | ETA: 3.3min
[Epoch 08] Iter  500/1071 | loss=0.23848 | 0.015s/iter | ETA: 3.3min
[Epoch 08] Iter  600/1071 | loss=0.31243 | 0.015s/iter | ETA: 3.3min
[Epoch 08] Iter  700/1071 | loss=0.27820 | 0.014s/iter | ETA: 3.0min
[Epoch 08] Iter  800/1071 | loss=0.28914 | 0.014s/iter | ETA: 3.0min
[Epoch 08] Iter  900/1071 | loss=0.29291 | 0.014s/iter | ETA: 3.0min
[Epoch 08] Iter 1000/1071 | loss=0.37848 | 0.013s/iter | ETA: 2.7min

------------------------------------------------------------
[Epoch 08] Summary | Time: 14.6s
Train Loss: 0.288433
Vali  Loss: 0.514460
Test  Loss: 0.367195
EarlyStopping counter: 5 out of 5
------------------------------------------------------------

************************************************************
EARLY STOPPING TRIGGERED
************************************************************
>>>>>>>testing : ETTm1_96_192_FreDEA_ETTm1_ftM_sl96_ll48_pl192_dm128_nh8_el1_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 11329
mse:0.3718568682670593, mae:0.3900059461593628, rmse:0.6098006963729858
✅ 实验完成: MSE=0.371857, MAE=0.390006

================================================================================
运行实验: ETTm1 - Pred 336
================================================================================
命令: python -u run_longExp.py --data ETTm1 --data_path ETTm1.csv --model FreDEA --model_id ETTm1_96_336 --seq_len 96 --pred_len 336 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 256 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --dropout 0.1 --batch_size 32 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4

Args in experiment:
Namespace(is_training=1, train_only=False, model_id='ETTm1_96_336', model='FreDEA', data='ETTm1', root_path='./dataset/', data_path='ETTm1.csv', channel_independence=0, features='M', target='OT', freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=336, individual=False, embed_type=0, enc_in=7, dec_in=7, c_out=7, d_model=128, reg_lambda=0.01, wavelet='haar', n_bins=8, n_heads=8, e_layers=1, d_layers=1, d_ff=256, moving_avg=25, factor=1, distil=True, rev_affine=1, memory_size=64, bottleneck_dim=-1, ablation_freq=0, ablation_tea=0, ablation_cea=0, dropout=0.1, embed='timeF', activation='gelu', output_attention=False, do_predict=False, num_workers=4, itr=1, train_epochs=20, batch_size=32, patience=5, learning_rate=0.001, des='Exp', loss='mse', lradj='type1', use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=False, devices='0,1,2,3', test_flop=False)
Use GPU: cuda:0
>>>>>>>start training : ETTm1_96_336_FreDEA_ETTm1_ftM_sl96_ll48_pl336_dm128_nh8_el1_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0>>>>>>>>>>>>>>>>>>>>>>>>>>
train 34129
val 11185
test 11185

============================================================
TRAINING CONFIGURATION
============================================================
Model:           FreDEA
Dataset:         ETTm1
Seq/Pred Len:    96 -> 336
Batch Size:      32
Learning Rate:   0.001
Train Epochs:    20
Total Params:    157,540
d_model:         128
n_heads:         8
e_layers:        1
============================================================

[Epoch 01] Iter  100/1066 | loss=0.48179 | 0.024s/iter | ETA: 8.4min
[Epoch 01] Iter  200/1066 | loss=0.42423 | 0.015s/iter | ETA: 5.3min
[Epoch 01] Iter  300/1066 | loss=0.49264 | 0.015s/iter | ETA: 5.3min
[Epoch 01] Iter  400/1066 | loss=0.38269 | 0.012s/iter | ETA: 4.2min
[Epoch 01] Iter  500/1066 | loss=0.39014 | 0.013s/iter | ETA: 4.4min
[Epoch 01] Iter  600/1066 | loss=0.31728 | 0.012s/iter | ETA: 4.1min
[Epoch 01] Iter  700/1066 | loss=0.33565 | 0.013s/iter | ETA: 4.3min
[Epoch 01] Iter  800/1066 | loss=0.29727 | 0.013s/iter | ETA: 4.4min
[Epoch 01] Iter  900/1066 | loss=0.37615 | 0.013s/iter | ETA: 4.3min
[Epoch 01] Iter 1000/1066 | loss=0.39719 | 0.013s/iter | ETA: 4.3min

------------------------------------------------------------
[Epoch 01] Summary | Time: 14.5s
Train Loss: 0.404625
Vali  Loss: 0.683043
Test  Loss: 0.408411
Validation loss decreased (inf --> 0.683043).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.001
[Epoch 02] Iter  100/1066 | loss=0.47234 | 0.045s/iter | ETA: 15.2min
[Epoch 02] Iter  200/1066 | loss=0.38887 | 0.009s/iter | ETA: 3.0min
[Epoch 02] Iter  300/1066 | loss=0.38070 | 0.009s/iter | ETA: 2.9min
[Epoch 02] Iter  400/1066 | loss=0.35492 | 0.007s/iter | ETA: 2.4min
[Epoch 02] Iter  500/1066 | loss=0.35350 | 0.009s/iter | ETA: 3.0min
[Epoch 02] Iter  600/1066 | loss=0.30856 | 0.009s/iter | ETA: 3.0min
[Epoch 02] Iter  700/1066 | loss=0.37953 | 0.009s/iter | ETA: 3.0min
[Epoch 02] Iter  800/1066 | loss=0.33268 | 0.010s/iter | ETA: 3.1min
[Epoch 02] Iter  900/1066 | loss=0.43380 | 0.011s/iter | ETA: 3.4min
[Epoch 02] Iter 1000/1066 | loss=0.39429 | 0.012s/iter | ETA: 3.7min

------------------------------------------------------------
[Epoch 02] Summary | Time: 10.2s
Train Loss: 0.369895
Vali  Loss: 0.654832
Test  Loss: 0.402854
Validation loss decreased (0.683043 --> 0.654832).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 03] Iter  100/1066 | loss=0.36069 | 0.046s/iter | ETA: 14.6min
[Epoch 03] Iter  200/1066 | loss=0.31432 | 0.011s/iter | ETA: 3.5min
[Epoch 03] Iter  300/1066 | loss=0.46797 | 0.011s/iter | ETA: 3.6min
[Epoch 03] Iter  400/1066 | loss=0.33124 | 0.012s/iter | ETA: 3.9min
[Epoch 03] Iter  500/1066 | loss=0.31846 | 0.013s/iter | ETA: 4.2min
[Epoch 03] Iter  600/1066 | loss=0.42392 | 0.013s/iter | ETA: 4.1min
[Epoch 03] Iter  700/1066 | loss=0.34496 | 0.013s/iter | ETA: 4.1min
[Epoch 03] Iter  800/1066 | loss=0.34990 | 0.013s/iter | ETA: 3.9min
[Epoch 03] Iter  900/1066 | loss=0.39156 | 0.006s/iter | ETA: 1.9min
[Epoch 03] Iter 1000/1066 | loss=0.38142 | 0.009s/iter | ETA: 2.6min

------------------------------------------------------------
[Epoch 03] Summary | Time: 11.9s
Train Loss: 0.356546
Vali  Loss: 0.661034
Test  Loss: 0.396522
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 0.00025
[Epoch 04] Iter  100/1066 | loss=0.42172 | 0.043s/iter | ETA: 12.9min
[Epoch 04] Iter  200/1066 | loss=0.29892 | 0.012s/iter | ETA: 3.5min
[Epoch 04] Iter  300/1066 | loss=0.46413 | 0.014s/iter | ETA: 4.2min
[Epoch 04] Iter  400/1066 | loss=0.28229 | 0.014s/iter | ETA: 4.1min
[Epoch 04] Iter  500/1066 | loss=0.34031 | 0.012s/iter | ETA: 3.5min
[Epoch 04] Iter  600/1066 | loss=0.31007 | 0.013s/iter | ETA: 3.9min
[Epoch 04] Iter  700/1066 | loss=0.30221 | 0.013s/iter | ETA: 3.8min
[Epoch 04] Iter  800/1066 | loss=0.35732 | 0.013s/iter | ETA: 3.8min
[Epoch 04] Iter  900/1066 | loss=0.34806 | 0.013s/iter | ETA: 3.8min
[Epoch 04] Iter 1000/1066 | loss=0.33821 | 0.013s/iter | ETA: 3.7min

------------------------------------------------------------
[Epoch 04] Summary | Time: 13.5s
Train Loss: 0.349874
Vali  Loss: 0.655197
Test  Loss: 0.400129
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 0.000125
[Epoch 05] Iter  100/1066 | loss=0.37738 | 0.050s/iter | ETA: 14.0min
[Epoch 05] Iter  200/1066 | loss=0.40482 | 0.011s/iter | ETA: 3.0min
[Epoch 05] Iter  300/1066 | loss=0.42514 | 0.010s/iter | ETA: 2.9min
[Epoch 05] Iter  400/1066 | loss=0.31432 | 0.011s/iter | ETA: 3.2min
[Epoch 05] Iter  500/1066 | loss=0.31060 | 0.012s/iter | ETA: 3.4min
[Epoch 05] Iter  600/1066 | loss=0.31468 | 0.013s/iter | ETA: 3.6min
[Epoch 05] Iter  700/1066 | loss=0.36965 | 0.013s/iter | ETA: 3.6min
[Epoch 05] Iter  800/1066 | loss=0.32975 | 0.013s/iter | ETA: 3.6min
[Epoch 05] Iter  900/1066 | loss=0.42602 | 0.013s/iter | ETA: 3.6min
[Epoch 05] Iter 1000/1066 | loss=0.34069 | 0.014s/iter | ETA: 3.6min

------------------------------------------------------------
[Epoch 05] Summary | Time: 13.4s
Train Loss: 0.345867
Vali  Loss: 0.655855
Test  Loss: 0.396921
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 6.25e-05
[Epoch 06] Iter  100/1066 | loss=0.37215 | 0.048s/iter | ETA: 12.8min
[Epoch 06] Iter  200/1066 | loss=0.40131 | 0.009s/iter | ETA: 2.4min
[Epoch 06] Iter  300/1066 | loss=0.32902 | 0.009s/iter | ETA: 2.4min
[Epoch 06] Iter  400/1066 | loss=0.34456 | 0.013s/iter | ETA: 3.3min
[Epoch 06] Iter  500/1066 | loss=0.38567 | 0.011s/iter | ETA: 2.8min
[Epoch 06] Iter  600/1066 | loss=0.28890 | 0.012s/iter | ETA: 3.2min
[Epoch 06] Iter  700/1066 | loss=0.29763 | 0.011s/iter | ETA: 2.8min
[Epoch 06] Iter  800/1066 | loss=0.31792 | 0.010s/iter | ETA: 2.6min
[Epoch 06] Iter  900/1066 | loss=0.30686 | 0.010s/iter | ETA: 2.5min
[Epoch 06] Iter 1000/1066 | loss=0.33502 | 0.010s/iter | ETA: 2.5min

------------------------------------------------------------
[Epoch 06] Summary | Time: 11.4s
Train Loss: 0.344025
Vali  Loss: 0.655645
Test  Loss: 0.398239
EarlyStopping counter: 4 out of 5
------------------------------------------------------------
Updating learning rate to 3.125e-05
[Epoch 07] Iter  100/1066 | loss=0.42404 | 0.044s/iter | ETA: 11.0min
[Epoch 07] Iter  200/1066 | loss=0.33714 | 0.009s/iter | ETA: 2.3min
[Epoch 07] Iter  300/1066 | loss=0.31412 | 0.007s/iter | ETA: 1.8min
[Epoch 07] Iter  400/1066 | loss=0.36092 | 0.006s/iter | ETA: 1.4min
[Epoch 07] Iter  500/1066 | loss=0.31114 | 0.006s/iter | ETA: 1.4min
[Epoch 07] Iter  600/1066 | loss=0.40533 | 0.006s/iter | ETA: 1.4min
[Epoch 07] Iter  700/1066 | loss=0.34039 | 0.006s/iter | ETA: 1.4min
[Epoch 07] Iter  800/1066 | loss=0.35697 | 0.007s/iter | ETA: 1.6min
[Epoch 07] Iter  900/1066 | loss=0.30527 | 0.009s/iter | ETA: 2.2min
[Epoch 07] Iter 1000/1066 | loss=0.32193 | 0.009s/iter | ETA: 2.1min

------------------------------------------------------------
[Epoch 07] Summary | Time: 8.2s
Train Loss: 0.343034
Vali  Loss: 0.654015
Test  Loss: 0.399512
Validation loss decreased (0.654832 --> 0.654015).  Saving model ...
------------------------------------------------------------
Updating learning rate to 1.5625e-05
[Epoch 08] Iter  100/1066 | loss=0.32364 | 0.044s/iter | ETA: 10.1min
[Epoch 08] Iter  200/1066 | loss=0.41003 | 0.009s/iter | ETA: 2.1min
[Epoch 08] Iter  300/1066 | loss=0.46987 | 0.009s/iter | ETA: 2.1min
[Epoch 08] Iter  400/1066 | loss=0.35113 | 0.009s/iter | ETA: 2.0min
[Epoch 08] Iter  500/1066 | loss=0.39727 | 0.009s/iter | ETA: 2.0min
[Epoch 08] Iter  600/1066 | loss=0.37264 | 0.009s/iter | ETA: 2.0min
[Epoch 08] Iter  700/1066 | loss=0.38630 | 0.009s/iter | ETA: 2.0min
[Epoch 08] Iter  800/1066 | loss=0.43156 | 0.006s/iter | ETA: 1.4min
[Epoch 08] Iter  900/1066 | loss=0.31163 | 0.009s/iter | ETA: 1.9min
[Epoch 08] Iter 1000/1066 | loss=0.30278 | 0.009s/iter | ETA: 1.9min

------------------------------------------------------------
[Epoch 08] Summary | Time: 9.5s
Train Loss: 0.342564
Vali  Loss: 0.654107
Test  Loss: 0.399515
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 7.8125e-06
[Epoch 09] Iter  100/1066 | loss=0.36513 | 0.047s/iter | ETA: 10.0min
[Epoch 09] Iter  200/1066 | loss=0.28099 | 0.013s/iter | ETA: 2.7min
[Epoch 09] Iter  300/1066 | loss=0.41700 | 0.014s/iter | ETA: 3.0min
[Epoch 09] Iter  400/1066 | loss=0.31973 | 0.015s/iter | ETA: 3.0min
[Epoch 09] Iter  500/1066 | loss=0.33321 | 0.015s/iter | ETA: 3.1min
[Epoch 09] Iter  600/1066 | loss=0.30466 | 0.015s/iter | ETA: 3.0min
[Epoch 09] Iter  700/1066 | loss=0.36720 | 0.015s/iter | ETA: 3.0min
[Epoch 09] Iter  800/1066 | loss=0.34959 | 0.012s/iter | ETA: 2.5min
[Epoch 09] Iter  900/1066 | loss=0.41362 | 0.012s/iter | ETA: 2.4min
[Epoch 09] Iter 1000/1066 | loss=0.36730 | 0.012s/iter | ETA: 2.4min

------------------------------------------------------------
[Epoch 09] Summary | Time: 14.4s
Train Loss: 0.342083
Vali  Loss: 0.653125
Test  Loss: 0.399867
Validation loss decreased (0.654015 --> 0.653125).  Saving model ...
------------------------------------------------------------
Updating learning rate to 3.90625e-06
[Epoch 10] Iter  100/1066 | loss=0.31530 | 0.050s/iter | ETA: 9.6min
[Epoch 10] Iter  200/1066 | loss=0.44389 | 0.012s/iter | ETA: 2.3min
[Epoch 10] Iter  300/1066 | loss=0.35728 | 0.011s/iter | ETA: 2.0min
[Epoch 10] Iter  400/1066 | loss=0.43937 | 0.009s/iter | ETA: 1.7min
[Epoch 10] Iter  500/1066 | loss=0.35489 | 0.009s/iter | ETA: 1.7min
[Epoch 10] Iter  600/1066 | loss=0.31940 | 0.010s/iter | ETA: 1.8min
[Epoch 10] Iter  700/1066 | loss=0.47759 | 0.012s/iter | ETA: 2.3min
[Epoch 10] Iter  800/1066 | loss=0.38236 | 0.014s/iter | ETA: 2.5min
[Epoch 10] Iter  900/1066 | loss=0.32398 | 0.014s/iter | ETA: 2.5min
[Epoch 10] Iter 1000/1066 | loss=0.41459 | 0.014s/iter | ETA: 2.5min

------------------------------------------------------------
[Epoch 10] Summary | Time: 12.7s
Train Loss: 0.341927
Vali  Loss: 0.652702
Test  Loss: 0.399879
Validation loss decreased (0.653125 --> 0.652702).  Saving model ...
------------------------------------------------------------
Updating learning rate to 1.953125e-06
[Epoch 11] Iter  100/1066 | loss=0.42356 | 0.051s/iter | ETA: 8.9min
[Epoch 11] Iter  200/1066 | loss=0.35258 | 0.014s/iter | ETA: 2.4min
[Epoch 11] Iter  300/1066 | loss=0.33834 | 0.013s/iter | ETA: 2.3min
[Epoch 11] Iter  400/1066 | loss=0.35832 | 0.013s/iter | ETA: 2.2min
[Epoch 11] Iter  500/1066 | loss=0.26686 | 0.014s/iter | ETA: 2.4min
[Epoch 11] Iter  600/1066 | loss=0.32778 | 0.015s/iter | ETA: 2.5min
[Epoch 11] Iter  700/1066 | loss=0.38588 | 0.015s/iter | ETA: 2.4min
[Epoch 11] Iter  800/1066 | loss=0.32647 | 0.015s/iter | ETA: 2.5min
[Epoch 11] Iter  900/1066 | loss=0.31550 | 0.015s/iter | ETA: 2.4min
[Epoch 11] Iter 1000/1066 | loss=0.28736 | 0.015s/iter | ETA: 2.4min

------------------------------------------------------------
[Epoch 11] Summary | Time: 15.2s
Train Loss: 0.342018
Vali  Loss: 0.653023
Test  Loss: 0.399624
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 9.765625e-07
[Epoch 12] Iter  100/1066 | loss=0.32484 | 0.051s/iter | ETA: 8.0min
[Epoch 12] Iter  200/1066 | loss=0.33014 | 0.009s/iter | ETA: 1.5min
[Epoch 12] Iter  300/1066 | loss=0.43648 | 0.009s/iter | ETA: 1.4min
[Epoch 12] Iter  400/1066 | loss=0.29346 | 0.009s/iter | ETA: 1.4min
[Epoch 12] Iter  500/1066 | loss=0.28908 | 0.009s/iter | ETA: 1.4min
[Epoch 12] Iter  600/1066 | loss=0.31927 | 0.009s/iter | ETA: 1.4min
[Epoch 12] Iter  700/1066 | loss=0.52700 | 0.009s/iter | ETA: 1.4min
[Epoch 12] Iter  800/1066 | loss=0.36461 | 0.009s/iter | ETA: 1.3min
[Epoch 12] Iter  900/1066 | loss=0.38286 | 0.009s/iter | ETA: 1.3min
[Epoch 12] Iter 1000/1066 | loss=0.32025 | 0.013s/iter | ETA: 1.8min

------------------------------------------------------------
[Epoch 12] Summary | Time: 10.9s
Train Loss: 0.341904
Vali  Loss: 0.653073
Test  Loss: 0.399764
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 4.8828125e-07
[Epoch 13] Iter  100/1066 | loss=0.31000 | 0.047s/iter | ETA: 6.6min
[Epoch 13] Iter  200/1066 | loss=0.33677 | 0.009s/iter | ETA: 1.3min
[Epoch 13] Iter  300/1066 | loss=0.38166 | 0.011s/iter | ETA: 1.5min
[Epoch 13] Iter  400/1066 | loss=0.32725 | 0.012s/iter | ETA: 1.6min
[Epoch 13] Iter  500/1066 | loss=0.26797 | 0.013s/iter | ETA: 1.7min
[Epoch 13] Iter  600/1066 | loss=0.32953 | 0.014s/iter | ETA: 1.8min
[Epoch 13] Iter  700/1066 | loss=0.32781 | 0.016s/iter | ETA: 2.0min
[Epoch 13] Iter  800/1066 | loss=0.37335 | 0.015s/iter | ETA: 1.9min
[Epoch 13] Iter  900/1066 | loss=0.31437 | 0.012s/iter | ETA: 1.5min
[Epoch 13] Iter 1000/1066 | loss=0.29631 | 0.014s/iter | ETA: 1.7min

------------------------------------------------------------
[Epoch 13] Summary | Time: 13.5s
Train Loss: 0.342095
Vali  Loss: 0.653561
Test  Loss: 0.399673
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 2.44140625e-07
[Epoch 14] Iter  100/1066 | loss=0.34953 | 0.051s/iter | ETA: 6.3min
[Epoch 14] Iter  200/1066 | loss=0.33869 | 0.009s/iter | ETA: 1.1min
[Epoch 14] Iter  300/1066 | loss=0.24971 | 0.013s/iter | ETA: 1.6min
[Epoch 14] Iter  400/1066 | loss=0.36397 | 0.014s/iter | ETA: 1.7min
[Epoch 14] Iter  500/1066 | loss=0.29339 | 0.014s/iter | ETA: 1.7min
[Epoch 14] Iter  600/1066 | loss=0.29978 | 0.015s/iter | ETA: 1.7min
[Epoch 14] Iter  700/1066 | loss=0.34331 | 0.010s/iter | ETA: 1.1min
[Epoch 14] Iter  800/1066 | loss=0.32367 | 0.011s/iter | ETA: 1.2min
[Epoch 14] Iter  900/1066 | loss=0.29856 | 0.014s/iter | ETA: 1.5min
[Epoch 14] Iter 1000/1066 | loss=0.36082 | 0.014s/iter | ETA: 1.5min

------------------------------------------------------------
[Epoch 14] Summary | Time: 13.5s
Train Loss: 0.341923
Vali  Loss: 0.653084
Test  Loss: 0.399718
EarlyStopping counter: 4 out of 5
------------------------------------------------------------
Updating learning rate to 1.220703125e-07
[Epoch 15] Iter  100/1066 | loss=0.31999 | 0.052s/iter | ETA: 5.5min
[Epoch 15] Iter  200/1066 | loss=0.41859 | 0.013s/iter | ETA: 1.3min
[Epoch 15] Iter  300/1066 | loss=0.48518 | 0.013s/iter | ETA: 1.3min
[Epoch 15] Iter  400/1066 | loss=0.36665 | 0.013s/iter | ETA: 1.3min
[Epoch 15] Iter  500/1066 | loss=0.36725 | 0.015s/iter | ETA: 1.5min
[Epoch 15] Iter  600/1066 | loss=0.34694 | 0.015s/iter | ETA: 1.4min
[Epoch 15] Iter  700/1066 | loss=0.29076 | 0.015s/iter | ETA: 1.4min
[Epoch 15] Iter  800/1066 | loss=0.32965 | 0.015s/iter | ETA: 1.4min
[Epoch 15] Iter  900/1066 | loss=0.31055 | 0.016s/iter | ETA: 1.4min
[Epoch 15] Iter 1000/1066 | loss=0.31453 | 0.014s/iter | ETA: 1.2min

------------------------------------------------------------
[Epoch 15] Summary | Time: 15.1s
Train Loss: 0.341991
Vali  Loss: 0.653483
Test  Loss: 0.399725
EarlyStopping counter: 5 out of 5
------------------------------------------------------------

************************************************************
EARLY STOPPING TRIGGERED
************************************************************
>>>>>>>testing : ETTm1_96_336_FreDEA_ETTm1_ftM_sl96_ll48_pl336_dm128_nh8_el1_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 11185
mse:0.3997076153755188, mae:0.41009780764579773, rmse:0.6322243213653564
✅ 实验完成: MSE=0.399708, MAE=0.410098

================================================================================
运行实验: ETTm1 - Pred 720
================================================================================
命令: python -u run_longExp.py --data ETTm1 --data_path ETTm1.csv --model FreDEA --model_id ETTm1_96_720 --seq_len 96 --pred_len 720 --enc_in 7 --dec_in 7 --c_out 7 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim 2 --dropout 0.3 --batch_size 32 --learning_rate 0.0003 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4

Args in experiment:
Namespace(is_training=1, train_only=False, model_id='ETTm1_96_720', model='FreDEA', data='ETTm1', root_path='./dataset/', data_path='ETTm1.csv', channel_independence=0, features='M', target='OT', freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=720, individual=False, embed_type=0, enc_in=7, dec_in=7, c_out=7, d_model=64, reg_lambda=0.01, wavelet='haar', n_bins=8, n_heads=8, e_layers=1, d_layers=1, d_ff=128, moving_avg=25, factor=1, distil=True, rev_affine=1, memory_size=64, bottleneck_dim=2, ablation_freq=0, ablation_tea=0, ablation_cea=0, dropout=0.3, embed='timeF', activation='gelu', output_attention=False, do_predict=False, num_workers=4, itr=1, train_epochs=20, batch_size=32, patience=5, learning_rate=0.0003, des='Exp', loss='mse', lradj='type1', use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=False, devices='0,1,2,3', test_flop=False)
Use GPU: cuda:0
>>>>>>>start training : ETTm1_96_720_FreDEA_ETTm1_ftM_sl96_ll48_pl720_dm64_nh8_el1_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0>>>>>>>>>>>>>>>>>>>>>>>>>>
train 33745
val 10801
test 10801

============================================================
TRAINING CONFIGURATION
============================================================
Model:           FreDEA
Dataset:         ETTm1
Seq/Pred Len:    96 -> 720
Batch Size:      32
Learning Rate:   0.0003
Train Epochs:    20
Total Params:    92,516
d_model:         64
n_heads:         8
e_layers:        1
============================================================

[Epoch 01] Iter  100/1054 | loss=0.57978 | 0.021s/iter | ETA: 7.3min
[Epoch 01] Iter  200/1054 | loss=0.57114 | 0.011s/iter | ETA: 3.7min
[Epoch 01] Iter  300/1054 | loss=0.51271 | 0.012s/iter | ETA: 4.0min
[Epoch 01] Iter  400/1054 | loss=0.51583 | 0.011s/iter | ETA: 3.9min
[Epoch 01] Iter  500/1054 | loss=0.55148 | 0.011s/iter | ETA: 3.9min
[Epoch 01] Iter  600/1054 | loss=0.47136 | 0.011s/iter | ETA: 3.9min
[Epoch 01] Iter  700/1054 | loss=0.46331 | 0.011s/iter | ETA: 3.8min
[Epoch 01] Iter  800/1054 | loss=0.54628 | 0.011s/iter | ETA: 3.8min
[Epoch 01] Iter  900/1054 | loss=0.51056 | 0.013s/iter | ETA: 4.2min
[Epoch 01] Iter 1000/1054 | loss=0.49513 | 0.014s/iter | ETA: 4.8min

------------------------------------------------------------
[Epoch 01] Summary | Time: 12.9s
Train Loss: 0.520479
Vali  Loss: 1.004888
Test  Loss: 0.482409
Validation loss decreased (inf --> 1.004888).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0003
[Epoch 02] Iter  100/1054 | loss=0.50205 | 0.048s/iter | ETA: 16.1min
[Epoch 02] Iter  200/1054 | loss=0.44891 | 0.009s/iter | ETA: 3.1min
[Epoch 02] Iter  300/1054 | loss=0.40503 | 0.009s/iter | ETA: 3.0min
[Epoch 02] Iter  400/1054 | loss=0.46508 | 0.009s/iter | ETA: 3.0min
[Epoch 02] Iter  500/1054 | loss=0.42655 | 0.009s/iter | ETA: 3.1min
[Epoch 02] Iter  600/1054 | loss=0.47466 | 0.010s/iter | ETA: 3.1min
[Epoch 02] Iter  700/1054 | loss=0.48724 | 0.009s/iter | ETA: 3.0min
[Epoch 02] Iter  800/1054 | loss=0.46751 | 0.009s/iter | ETA: 2.9min
[Epoch 02] Iter  900/1054 | loss=0.45699 | 0.009s/iter | ETA: 2.9min
[Epoch 02] Iter 1000/1054 | loss=0.51248 | 0.009s/iter | ETA: 2.9min

------------------------------------------------------------
[Epoch 02] Summary | Time: 9.8s
Train Loss: 0.477783
Vali  Loss: 0.980989
Test  Loss: 0.472375
Validation loss decreased (1.004888 --> 0.980989).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.00015
[Epoch 03] Iter  100/1054 | loss=0.44255 | 0.046s/iter | ETA: 14.4min
[Epoch 03] Iter  200/1054 | loss=0.48227 | 0.009s/iter | ETA: 2.9min
[Epoch 03] Iter  300/1054 | loss=0.43745 | 0.009s/iter | ETA: 2.9min
[Epoch 03] Iter  400/1054 | loss=0.48878 | 0.009s/iter | ETA: 2.8min
[Epoch 03] Iter  500/1054 | loss=0.40788 | 0.009s/iter | ETA: 2.8min
[Epoch 03] Iter  600/1054 | loss=0.44864 | 0.009s/iter | ETA: 2.9min
[Epoch 03] Iter  700/1054 | loss=0.53254 | 0.009s/iter | ETA: 2.9min
[Epoch 03] Iter  800/1054 | loss=0.44188 | 0.009s/iter | ETA: 2.8min
[Epoch 03] Iter  900/1054 | loss=0.43642 | 0.009s/iter | ETA: 2.8min
[Epoch 03] Iter 1000/1054 | loss=0.44312 | 0.010s/iter | ETA: 3.1min

------------------------------------------------------------
[Epoch 03] Summary | Time: 10.3s
Train Loss: 0.469096
Vali  Loss: 0.978696
Test  Loss: 0.468022
Validation loss decreased (0.980989 --> 0.978696).  Saving model ...
------------------------------------------------------------
Updating learning rate to 7.5e-05
[Epoch 04] Iter  100/1054 | loss=0.43552 | 0.050s/iter | ETA: 14.8min
[Epoch 04] Iter  200/1054 | loss=0.54477 | 0.009s/iter | ETA: 2.7min
[Epoch 04] Iter  300/1054 | loss=0.36289 | 0.012s/iter | ETA: 3.6min
[Epoch 04] Iter  400/1054 | loss=0.57702 | 0.014s/iter | ETA: 4.0min
[Epoch 04] Iter  500/1054 | loss=0.40773 | 0.014s/iter | ETA: 4.0min
[Epoch 04] Iter  600/1054 | loss=0.48957 | 0.013s/iter | ETA: 3.8min
[Epoch 04] Iter  700/1054 | loss=0.39323 | 0.014s/iter | ETA: 3.9min
[Epoch 04] Iter  800/1054 | loss=0.45191 | 0.014s/iter | ETA: 3.9min
[Epoch 04] Iter  900/1054 | loss=0.45264 | 0.014s/iter | ETA: 3.9min
[Epoch 04] Iter 1000/1054 | loss=0.42297 | 0.014s/iter | ETA: 3.9min

------------------------------------------------------------
[Epoch 04] Summary | Time: 13.4s
Train Loss: 0.466131
Vali  Loss: 0.977005
Test  Loss: 0.466184
Validation loss decreased (0.978696 --> 0.977005).  Saving model ...
------------------------------------------------------------
Updating learning rate to 3.75e-05
[Epoch 05] Iter  100/1054 | loss=0.48558 | 0.050s/iter | ETA: 14.1min
[Epoch 05] Iter  200/1054 | loss=0.48863 | 0.012s/iter | ETA: 3.4min
[Epoch 05] Iter  300/1054 | loss=0.50646 | 0.013s/iter | ETA: 3.5min
[Epoch 05] Iter  400/1054 | loss=0.39154 | 0.013s/iter | ETA: 3.6min
[Epoch 05] Iter  500/1054 | loss=0.54206 | 0.013s/iter | ETA: 3.5min
[Epoch 05] Iter  600/1054 | loss=0.48728 | 0.013s/iter | ETA: 3.5min
[Epoch 05] Iter  700/1054 | loss=0.43960 | 0.013s/iter | ETA: 3.5min
[Epoch 05] Iter  800/1054 | loss=0.42183 | 0.013s/iter | ETA: 3.5min
[Epoch 05] Iter  900/1054 | loss=0.44511 | 0.013s/iter | ETA: 3.4min
[Epoch 05] Iter 1000/1054 | loss=0.44595 | 0.013s/iter | ETA: 3.4min

------------------------------------------------------------
[Epoch 05] Summary | Time: 13.6s
Train Loss: 0.464559
Vali  Loss: 0.976535
Test  Loss: 0.465008
Validation loss decreased (0.977005 --> 0.976535).  Saving model ...
------------------------------------------------------------
Updating learning rate to 1.875e-05
[Epoch 06] Iter  100/1054 | loss=0.44639 | 0.051s/iter | ETA: 13.5min
[Epoch 06] Iter  200/1054 | loss=0.55822 | 0.012s/iter | ETA: 3.1min
[Epoch 06] Iter  300/1054 | loss=0.52954 | 0.012s/iter | ETA: 3.0min
[Epoch 06] Iter  400/1054 | loss=0.43151 | 0.012s/iter | ETA: 3.1min
[Epoch 06] Iter  500/1054 | loss=0.60462 | 0.011s/iter | ETA: 2.8min
[Epoch 06] Iter  600/1054 | loss=0.46436 | 0.011s/iter | ETA: 2.9min
[Epoch 06] Iter  700/1054 | loss=0.46622 | 0.012s/iter | ETA: 3.0min
[Epoch 06] Iter  800/1054 | loss=0.43656 | 0.012s/iter | ETA: 2.9min
[Epoch 06] Iter  900/1054 | loss=0.43934 | 0.012s/iter | ETA: 2.9min
[Epoch 06] Iter 1000/1054 | loss=0.47232 | 0.012s/iter | ETA: 2.9min

------------------------------------------------------------
[Epoch 06] Summary | Time: 12.5s
Train Loss: 0.463976
Vali  Loss: 0.978104
Test  Loss: 0.464415
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 9.375e-06
[Epoch 07] Iter  100/1054 | loss=0.44244 | 0.050s/iter | ETA: 12.3min
[Epoch 07] Iter  200/1054 | loss=0.54472 | 0.014s/iter | ETA: 3.3min
[Epoch 07] Iter  300/1054 | loss=0.45566 | 0.013s/iter | ETA: 3.2min
[Epoch 07] Iter  400/1054 | loss=0.47022 | 0.013s/iter | ETA: 3.1min
[Epoch 07] Iter  500/1054 | loss=0.49912 | 0.013s/iter | ETA: 3.0min
[Epoch 07] Iter  600/1054 | loss=0.42260 | 0.012s/iter | ETA: 2.8min
[Epoch 07] Iter  700/1054 | loss=0.46986 | 0.014s/iter | ETA: 3.4min
[Epoch 07] Iter  800/1054 | loss=0.39638 | 0.014s/iter | ETA: 3.3min
[Epoch 07] Iter  900/1054 | loss=0.48607 | 0.015s/iter | ETA: 3.4min
[Epoch 07] Iter 1000/1054 | loss=0.40530 | 0.012s/iter | ETA: 2.7min

------------------------------------------------------------
[Epoch 07] Summary | Time: 13.7s
Train Loss: 0.463677
Vali  Loss: 0.977684
Test  Loss: 0.464481
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 4.6875e-06
[Epoch 08] Iter  100/1054 | loss=0.46388 | 0.051s/iter | ETA: 11.5min
[Epoch 08] Iter  200/1054 | loss=0.49204 | 0.013s/iter | ETA: 3.0min
[Epoch 08] Iter  300/1054 | loss=0.44873 | 0.013s/iter | ETA: 2.9min
[Epoch 08] Iter  400/1054 | loss=0.50913 | 0.013s/iter | ETA: 2.9min
[Epoch 08] Iter  500/1054 | loss=0.48565 | 0.013s/iter | ETA: 2.9min
[Epoch 08] Iter  600/1054 | loss=0.57445 | 0.013s/iter | ETA: 2.9min
[Epoch 08] Iter  700/1054 | loss=0.43818 | 0.015s/iter | ETA: 3.2min
[Epoch 08] Iter  800/1054 | loss=0.39553 | 0.015s/iter | ETA: 3.3min
[Epoch 08] Iter  900/1054 | loss=0.49623 | 0.012s/iter | ETA: 2.5min
[Epoch 08] Iter 1000/1054 | loss=0.37944 | 0.013s/iter | ETA: 2.7min

------------------------------------------------------------
[Epoch 08] Summary | Time: 14.1s
Train Loss: 0.463544
Vali  Loss: 0.976513
Test  Loss: 0.464770
Validation loss decreased (0.976535 --> 0.976513).  Saving model ...
------------------------------------------------------------
Updating learning rate to 2.34375e-06
[Epoch 09] Iter  100/1054 | loss=0.49033 | 0.055s/iter | ETA: 11.4min
[Epoch 09] Iter  200/1054 | loss=0.46432 | 0.015s/iter | ETA: 3.0min
[Epoch 09] Iter  300/1054 | loss=0.41536 | 0.015s/iter | ETA: 3.0min
[Epoch 09] Iter  400/1054 | loss=0.48633 | 0.014s/iter | ETA: 2.9min
[Epoch 09] Iter  500/1054 | loss=0.42124 | 0.015s/iter | ETA: 3.0min
[Epoch 09] Iter  600/1054 | loss=0.43847 | 0.015s/iter | ETA: 3.1min
[Epoch 09] Iter  700/1054 | loss=0.53020 | 0.015s/iter | ETA: 3.1min
[Epoch 09] Iter  800/1054 | loss=0.50074 | 0.013s/iter | ETA: 2.5min
[Epoch 09] Iter  900/1054 | loss=0.48591 | 0.015s/iter | ETA: 2.9min
[Epoch 09] Iter 1000/1054 | loss=0.48945 | 0.015s/iter | ETA: 2.9min

------------------------------------------------------------
[Epoch 09] Summary | Time: 15.6s
Train Loss: 0.463122
Vali  Loss: 0.976703
Test  Loss: 0.464641
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 1.171875e-06
[Epoch 10] Iter  100/1054 | loss=0.50808 | 0.053s/iter | ETA: 10.2min
[Epoch 10] Iter  200/1054 | loss=0.39138 | 0.011s/iter | ETA: 2.1min
[Epoch 10] Iter  300/1054 | loss=0.47683 | 0.011s/iter | ETA: 2.0min
[Epoch 10] Iter  400/1054 | loss=0.42825 | 0.011s/iter | ETA: 2.0min
[Epoch 10] Iter  500/1054 | loss=0.56107 | 0.011s/iter | ETA: 2.0min
[Epoch 10] Iter  600/1054 | loss=0.49646 | 0.011s/iter | ETA: 2.1min
[Epoch 10] Iter  700/1054 | loss=0.45017 | 0.012s/iter | ETA: 2.1min
[Epoch 10] Iter  800/1054 | loss=0.47095 | 0.012s/iter | ETA: 2.1min
[Epoch 10] Iter  900/1054 | loss=0.37707 | 0.012s/iter | ETA: 2.1min
[Epoch 10] Iter 1000/1054 | loss=0.47999 | 0.011s/iter | ETA: 2.0min

------------------------------------------------------------
[Epoch 10] Summary | Time: 12.1s
Train Loss: 0.463286
Vali  Loss: 0.976606
Test  Loss: 0.464644
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 5.859375e-07
[Epoch 11] Iter  100/1054 | loss=0.39627 | 0.048s/iter | ETA: 8.4min
[Epoch 11] Iter  200/1054 | loss=0.51554 | 0.007s/iter | ETA: 1.2min
[Epoch 11] Iter  300/1054 | loss=0.38617 | 0.009s/iter | ETA: 1.6min
[Epoch 11] Iter  400/1054 | loss=0.42054 | 0.008s/iter | ETA: 1.3min
[Epoch 11] Iter  500/1054 | loss=0.49333 | 0.009s/iter | ETA: 1.6min
[Epoch 11] Iter  600/1054 | loss=0.53905 | 0.009s/iter | ETA: 1.5min
[Epoch 11] Iter  700/1054 | loss=0.58083 | 0.009s/iter | ETA: 1.5min
[Epoch 11] Iter  800/1054 | loss=0.36806 | 0.009s/iter | ETA: 1.5min
[Epoch 11] Iter  900/1054 | loss=0.47453 | 0.009s/iter | ETA: 1.5min
[Epoch 11] Iter 1000/1054 | loss=0.46871 | 0.010s/iter | ETA: 1.5min

------------------------------------------------------------
[Epoch 11] Summary | Time: 9.6s
Train Loss: 0.463111
Vali  Loss: 0.976118
Test  Loss: 0.464614
Validation loss decreased (0.976513 --> 0.976118).  Saving model ...
------------------------------------------------------------
Updating learning rate to 2.9296875e-07
[Epoch 12] Iter  100/1054 | loss=0.46842 | 0.049s/iter | ETA: 7.7min
[Epoch 12] Iter  200/1054 | loss=0.48707 | 0.015s/iter | ETA: 2.3min
[Epoch 12] Iter  300/1054 | loss=0.47056 | 0.016s/iter | ETA: 2.5min
[Epoch 12] Iter  400/1054 | loss=0.47430 | 0.016s/iter | ETA: 2.4min
[Epoch 12] Iter  500/1054 | loss=0.43700 | 0.016s/iter | ETA: 2.4min
[Epoch 12] Iter  600/1054 | loss=0.53127 | 0.015s/iter | ETA: 2.2min
[Epoch 12] Iter  700/1054 | loss=0.52635 | 0.014s/iter | ETA: 2.0min
[Epoch 12] Iter  800/1054 | loss=0.42900 | 0.012s/iter | ETA: 1.7min
[Epoch 12] Iter  900/1054 | loss=0.40890 | 0.012s/iter | ETA: 1.8min
[Epoch 12] Iter 1000/1054 | loss=0.51777 | 0.012s/iter | ETA: 1.7min

------------------------------------------------------------
[Epoch 12] Summary | Time: 14.7s
Train Loss: 0.463035
Vali  Loss: 0.976935
Test  Loss: 0.464588
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 1.46484375e-07
[Epoch 13] Iter  100/1054 | loss=0.45553 | 0.049s/iter | ETA: 6.8min
[Epoch 13] Iter  200/1054 | loss=0.45331 | 0.009s/iter | ETA: 1.3min
[Epoch 13] Iter  300/1054 | loss=0.52719 | 0.009s/iter | ETA: 1.3min
[Epoch 13] Iter  400/1054 | loss=0.46910 | 0.011s/iter | ETA: 1.5min
[Epoch 13] Iter  500/1054 | loss=0.50923 | 0.013s/iter | ETA: 1.7min
[Epoch 13] Iter  600/1054 | loss=0.44632 | 0.014s/iter | ETA: 1.8min
[Epoch 13] Iter  700/1054 | loss=0.40663 | 0.014s/iter | ETA: 1.9min
[Epoch 13] Iter  800/1054 | loss=0.45428 | 0.015s/iter | ETA: 1.9min
[Epoch 13] Iter  900/1054 | loss=0.53368 | 0.015s/iter | ETA: 1.9min
[Epoch 13] Iter 1000/1054 | loss=0.39834 | 0.015s/iter | ETA: 1.8min

------------------------------------------------------------
[Epoch 13] Summary | Time: 13.4s
Train Loss: 0.463301
Vali  Loss: 0.976712
Test  Loss: 0.464578
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 7.32421875e-08
[Epoch 14] Iter  100/1054 | loss=0.49409 | 0.052s/iter | ETA: 6.3min
[Epoch 14] Iter  200/1054 | loss=0.55113 | 0.011s/iter | ETA: 1.4min
[Epoch 14] Iter  300/1054 | loss=0.46611 | 0.011s/iter | ETA: 1.3min
[Epoch 14] Iter  400/1054 | loss=0.43253 | 0.011s/iter | ETA: 1.3min
[Epoch 14] Iter  500/1054 | loss=0.37212 | 0.011s/iter | ETA: 1.3min
[Epoch 14] Iter  600/1054 | loss=0.47819 | 0.011s/iter | ETA: 1.3min
[Epoch 14] Iter  700/1054 | loss=0.44956 | 0.011s/iter | ETA: 1.3min
[Epoch 14] Iter  800/1054 | loss=0.45535 | 0.011s/iter | ETA: 1.2min
[Epoch 14] Iter  900/1054 | loss=0.50383 | 0.011s/iter | ETA: 1.2min
[Epoch 14] Iter 1000/1054 | loss=0.45587 | 0.011s/iter | ETA: 1.2min

------------------------------------------------------------
[Epoch 14] Summary | Time: 12.0s
Train Loss: 0.463195
Vali  Loss: 0.977101
Test  Loss: 0.464572
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 3.662109375e-08
[Epoch 15] Iter  100/1054 | loss=0.47617 | 0.049s/iter | ETA: 5.0min
[Epoch 15] Iter  200/1054 | loss=0.43501 | 0.013s/iter | ETA: 1.3min
[Epoch 15] Iter  300/1054 | loss=0.45127 | 0.012s/iter | ETA: 1.2min
[Epoch 15] Iter  400/1054 | loss=0.49727 | 0.012s/iter | ETA: 1.2min
[Epoch 15] Iter  500/1054 | loss=0.47085 | 0.013s/iter | ETA: 1.2min
[Epoch 15] Iter  600/1054 | loss=0.47101 | 0.013s/iter | ETA: 1.2min
[Epoch 15] Iter  700/1054 | loss=0.44139 | 0.013s/iter | ETA: 1.2min
[Epoch 15] Iter  800/1054 | loss=0.41171 | 0.013s/iter | ETA: 1.2min
[Epoch 15] Iter  900/1054 | loss=0.45810 | 0.013s/iter | ETA: 1.2min
[Epoch 15] Iter 1000/1054 | loss=0.60218 | 0.013s/iter | ETA: 1.1min

------------------------------------------------------------
[Epoch 15] Summary | Time: 13.3s
Train Loss: 0.463294
Vali  Loss: 0.977256
Test  Loss: 0.464573
EarlyStopping counter: 4 out of 5
------------------------------------------------------------
Updating learning rate to 1.8310546875e-08
[Epoch 16] Iter  100/1054 | loss=0.53598 | 0.048s/iter | ETA: 4.1min
[Epoch 16] Iter  200/1054 | loss=0.49947 | 0.009s/iter | ETA: 0.8min
[Epoch 16] Iter  300/1054 | loss=0.45028 | 0.010s/iter | ETA: 0.8min
[Epoch 16] Iter  400/1054 | loss=0.41852 | 0.009s/iter | ETA: 0.8min
[Epoch 16] Iter  500/1054 | loss=0.36168 | 0.009s/iter | ETA: 0.7min
[Epoch 16] Iter  600/1054 | loss=0.44269 | 0.009s/iter | ETA: 0.7min
[Epoch 16] Iter  700/1054 | loss=0.56250 | 0.009s/iter | ETA: 0.7min
[Epoch 16] Iter  800/1054 | loss=0.37218 | 0.009s/iter | ETA: 0.7min
[Epoch 16] Iter  900/1054 | loss=0.47463 | 0.009s/iter | ETA: 0.7min
[Epoch 16] Iter 1000/1054 | loss=0.46681 | 0.009s/iter | ETA: 0.6min

------------------------------------------------------------
[Epoch 16] Summary | Time: 9.8s
Train Loss: 0.463201
Vali  Loss: 0.977348
Test  Loss: 0.464572
EarlyStopping counter: 5 out of 5
------------------------------------------------------------

************************************************************
EARLY STOPPING TRIGGERED
************************************************************
>>>>>>>testing : ETTm1_96_720_FreDEA_ETTm1_ftM_sl96_ll48_pl720_dm64_nh8_el1_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 10801
mse:0.4643148183822632, mae:0.44643425941467285, rmse:0.6814064979553223

================================================================================
运行实验: ETTm2 - Pred 96
================================================================================
命令: python -u run_longExp.py --data ETTm2 --data_path ETTm2.csv --model FreDEA --model_id ETTm2_96_96 --seq_len 96 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 256 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --dropout 0.05 --batch_size 32 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4

Args in experiment:
Namespace(is_training=1, train_only=False, model_id='ETTm2_96_96', model='FreDEA', data='ETTm2', root_path='./dataset/', data_path='ETTm2.csv', channel_independence=0, features='M', target='OT', freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=96, individual=False, embed_type=0, enc_in=7, dec_in=7, c_out=7, d_model=128, reg_lambda=0.01, wavelet='haar', n_bins=8, n_heads=8, e_layers=1, d_layers=1, d_ff=256, moving_avg=25, factor=1, distil=True, rev_affine=1, memory_size=64, bottleneck_dim=-1, ablation_freq=0, ablation_tea=0, ablation_cea=0, dropout=0.05, embed='timeF', activation='gelu', output_attention=False, do_predict=False, num_workers=4, itr=1, train_epochs=20, batch_size=32, patience=5, learning_rate=0.001, des='Exp', loss='mse', lradj='type1', use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=False, devices='0,1,2,3', test_flop=False)
Use GPU: cuda:0
>>>>>>>start training : ETTm2_96_96_FreDEA_ETTm2_ftM_sl96_ll48_pl96_dm128_nh8_el1_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0>>>>>>>>>>>>>>>>>>>>>>>>>>
train 34369
val 11425
test 11425

============================================================
TRAINING CONFIGURATION
============================================================
Model:           FreDEA
Dataset:         ETTm2
Seq/Pred Len:    96 -> 96
Batch Size:      32
Learning Rate:   0.001
Train Epochs:    20
Total Params:    126,580
d_model:         128
n_heads:         8
e_layers:        1
============================================================

[Epoch 01] Iter  100/1074 | loss=0.28078 | 0.020s/iter | ETA: 7.2min
[Epoch 01] Iter  200/1074 | loss=0.32644 | 0.009s/iter | ETA: 3.2min
[Epoch 01] Iter  300/1074 | loss=0.37294 | 0.014s/iter | ETA: 5.0min
[Epoch 01] Iter  400/1074 | loss=0.14795 | 0.014s/iter | ETA: 4.8min
[Epoch 01] Iter  500/1074 | loss=0.57403 | 0.014s/iter | ETA: 4.9min
[Epoch 01] Iter  600/1074 | loss=0.19569 | 0.014s/iter | ETA: 4.9min
[Epoch 01] Iter  700/1074 | loss=0.16571 | 0.014s/iter | ETA: 4.9min
[Epoch 01] Iter  800/1074 | loss=0.21136 | 0.014s/iter | ETA: 4.8min
[Epoch 01] Iter  900/1074 | loss=0.20948 | 0.014s/iter | ETA: 4.8min
[Epoch 01] Iter 1000/1074 | loss=0.19426 | 0.014s/iter | ETA: 4.7min

------------------------------------------------------------
[Epoch 01] Summary | Time: 14.6s
Train Loss: 0.236269
Vali  Loss: 0.130711
Test  Loss: 0.180008
Validation loss decreased (inf --> 0.130711).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.001
[Epoch 02] Iter  100/1074 | loss=0.16363 | 0.048s/iter | ETA: 16.4min
[Epoch 02] Iter  200/1074 | loss=0.21044 | 0.012s/iter | ETA: 4.2min
[Epoch 02] Iter  300/1074 | loss=0.12331 | 0.015s/iter | ETA: 4.9min
[Epoch 02] Iter  400/1074 | loss=0.32140 | 0.015s/iter | ETA: 4.9min
[Epoch 02] Iter  500/1074 | loss=0.21413 | 0.015s/iter | ETA: 4.9min
[Epoch 02] Iter  600/1074 | loss=0.21395 | 0.015s/iter | ETA: 5.1min
[Epoch 02] Iter  700/1074 | loss=0.18611 | 0.015s/iter | ETA: 4.8min
[Epoch 02] Iter  800/1074 | loss=0.18256 | 0.014s/iter | ETA: 4.5min
[Epoch 02] Iter  900/1074 | loss=0.25914 | 0.014s/iter | ETA: 4.4min
[Epoch 02] Iter 1000/1074 | loss=0.17884 | 0.014s/iter | ETA: 4.4min

------------------------------------------------------------
[Epoch 02] Summary | Time: 15.1s
Train Loss: 0.216341
Vali  Loss: 0.126279
Test  Loss: 0.174769
Validation loss decreased (0.130711 --> 0.126279).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 03] Iter  100/1074 | loss=0.16461 | 0.049s/iter | ETA: 15.8min
[Epoch 03] Iter  200/1074 | loss=0.10632 | 0.013s/iter | ETA: 4.2min
[Epoch 03] Iter  300/1074 | loss=0.11810 | 0.013s/iter | ETA: 4.2min
[Epoch 03] Iter  400/1074 | loss=0.22049 | 0.013s/iter | ETA: 4.1min
[Epoch 03] Iter  500/1074 | loss=0.12430 | 0.013s/iter | ETA: 4.1min
[Epoch 03] Iter  600/1074 | loss=0.12272 | 0.013s/iter | ETA: 4.1min
[Epoch 03] Iter  700/1074 | loss=0.17419 | 0.013s/iter | ETA: 4.0min
[Epoch 03] Iter  800/1074 | loss=0.73947 | 0.013s/iter | ETA: 4.0min
[Epoch 03] Iter  900/1074 | loss=0.09631 | 0.014s/iter | ETA: 4.3min
[Epoch 03] Iter 1000/1074 | loss=0.26157 | 0.009s/iter | ETA: 2.8min

------------------------------------------------------------
[Epoch 03] Summary | Time: 13.6s
Train Loss: 0.201929
Vali  Loss: 0.125938
Test  Loss: 0.175712
Validation loss decreased (0.126279 --> 0.125938).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.00025
[Epoch 04] Iter  100/1074 | loss=0.15843 | 0.043s/iter | ETA: 13.0min
[Epoch 04] Iter  200/1074 | loss=0.09828 | 0.013s/iter | ETA: 3.9min
[Epoch 04] Iter  300/1074 | loss=0.21284 | 0.014s/iter | ETA: 4.3min
[Epoch 04] Iter  400/1074 | loss=0.14344 | 0.014s/iter | ETA: 4.3min
[Epoch 04] Iter  500/1074 | loss=0.12879 | 0.014s/iter | ETA: 4.2min
[Epoch 04] Iter  600/1074 | loss=0.09960 | 0.014s/iter | ETA: 4.2min
[Epoch 04] Iter  700/1074 | loss=0.15187 | 0.014s/iter | ETA: 4.2min
[Epoch 04] Iter  800/1074 | loss=0.14631 | 0.014s/iter | ETA: 4.1min
[Epoch 04] Iter  900/1074 | loss=0.26324 | 0.014s/iter | ETA: 4.1min
[Epoch 04] Iter 1000/1074 | loss=0.12096 | 0.014s/iter | ETA: 4.1min

------------------------------------------------------------
[Epoch 04] Summary | Time: 14.9s
Train Loss: 0.192815
Vali  Loss: 0.125812
Test  Loss: 0.175968
Validation loss decreased (0.125938 --> 0.125812).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.000125
[Epoch 05] Iter  100/1074 | loss=0.19673 | 0.048s/iter | ETA: 13.7min
[Epoch 05] Iter  200/1074 | loss=0.44699 | 0.014s/iter | ETA: 3.9min
[Epoch 05] Iter  300/1074 | loss=0.13839 | 0.013s/iter | ETA: 3.8min
[Epoch 05] Iter  400/1074 | loss=0.18875 | 0.009s/iter | ETA: 2.5min
[Epoch 05] Iter  500/1074 | loss=0.28698 | 0.012s/iter | ETA: 3.4min
[Epoch 05] Iter  600/1074 | loss=0.24048 | 0.012s/iter | ETA: 3.3min
[Epoch 05] Iter  700/1074 | loss=0.10273 | 0.013s/iter | ETA: 3.6min
[Epoch 05] Iter  800/1074 | loss=0.10578 | 0.014s/iter | ETA: 3.8min
[Epoch 05] Iter  900/1074 | loss=0.27110 | 0.014s/iter | ETA: 3.7min
[Epoch 05] Iter 1000/1074 | loss=0.18012 | 0.013s/iter | ETA: 3.5min

------------------------------------------------------------
[Epoch 05] Summary | Time: 13.8s
Train Loss: 0.188128
Vali  Loss: 0.126235
Test  Loss: 0.177337
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 6.25e-05
[Epoch 06] Iter  100/1074 | loss=0.16041 | 0.046s/iter | ETA: 12.2min
[Epoch 06] Iter  200/1074 | loss=0.14809 | 0.011s/iter | ETA: 3.0min
[Epoch 06] Iter  300/1074 | loss=0.15698 | 0.014s/iter | ETA: 3.6min
[Epoch 06] Iter  400/1074 | loss=0.35432 | 0.014s/iter | ETA: 3.6min
[Epoch 06] Iter  500/1074 | loss=0.10339 | 0.014s/iter | ETA: 3.6min
[Epoch 06] Iter  600/1074 | loss=0.13503 | 0.014s/iter | ETA: 3.5min
[Epoch 06] Iter  700/1074 | loss=0.13895 | 0.014s/iter | ETA: 3.5min
[Epoch 06] Iter  800/1074 | loss=0.11943 | 0.014s/iter | ETA: 3.5min
[Epoch 06] Iter  900/1074 | loss=0.19896 | 0.014s/iter | ETA: 3.6min
[Epoch 06] Iter 1000/1074 | loss=0.14784 | 0.014s/iter | ETA: 3.6min

------------------------------------------------------------
[Epoch 06] Summary | Time: 14.5s
Train Loss: 0.185642
Vali  Loss: 0.126987
Test  Loss: 0.178072
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 3.125e-05
[Epoch 07] Iter  100/1074 | loss=0.41718 | 0.046s/iter | ETA: 11.5min
[Epoch 07] Iter  200/1074 | loss=0.49525 | 0.010s/iter | ETA: 2.5min
[Epoch 07] Iter  300/1074 | loss=0.17239 | 0.011s/iter | ETA: 2.7min
[Epoch 07] Iter  400/1074 | loss=0.13615 | 0.014s/iter | ETA: 3.5min
[Epoch 07] Iter  500/1074 | loss=0.15684 | 0.016s/iter | ETA: 3.8min
[Epoch 07] Iter  600/1074 | loss=0.19803 | 0.016s/iter | ETA: 3.7min
[Epoch 07] Iter  700/1074 | loss=0.15509 | 0.013s/iter | ETA: 3.1min
[Epoch 07] Iter  800/1074 | loss=0.12119 | 0.014s/iter | ETA: 3.3min
[Epoch 07] Iter  900/1074 | loss=0.13614 | 0.014s/iter | ETA: 3.3min
[Epoch 07] Iter 1000/1074 | loss=0.17017 | 0.014s/iter | ETA: 3.3min

------------------------------------------------------------
[Epoch 07] Summary | Time: 14.4s
Train Loss: 0.184204
Vali  Loss: 0.126873
Test  Loss: 0.178188
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 1.5625e-05
[Epoch 08] Iter  100/1074 | loss=0.15781 | 0.051s/iter | ETA: 11.8min
[Epoch 08] Iter  200/1074 | loss=0.23495 | 0.015s/iter | ETA: 3.4min
[Epoch 08] Iter  300/1074 | loss=0.12566 | 0.016s/iter | ETA: 3.5min
[Epoch 08] Iter  400/1074 | loss=0.20484 | 0.015s/iter | ETA: 3.5min
[Epoch 08] Iter  500/1074 | loss=0.12091 | 0.016s/iter | ETA: 3.5min
[Epoch 08] Iter  600/1074 | loss=0.16400 | 0.015s/iter | ETA: 3.4min
[Epoch 08] Iter  700/1074 | loss=0.15095 | 0.015s/iter | ETA: 3.4min
[Epoch 08] Iter  800/1074 | loss=0.10510 | 0.015s/iter | ETA: 3.4min
[Epoch 08] Iter  900/1074 | loss=0.18713 | 0.015s/iter | ETA: 3.2min
[Epoch 08] Iter 1000/1074 | loss=0.20209 | 0.015s/iter | ETA: 3.3min

------------------------------------------------------------
[Epoch 08] Summary | Time: 16.4s
Train Loss: 0.183767
Vali  Loss: 0.127045
Test  Loss: 0.178522
EarlyStopping counter: 4 out of 5
------------------------------------------------------------
Updating learning rate to 7.8125e-06
[Epoch 09] Iter  100/1074 | loss=0.16753 | 0.048s/iter | ETA: 10.3min
[Epoch 09] Iter  200/1074 | loss=0.17801 | 0.010s/iter | ETA: 2.2min
[Epoch 09] Iter  300/1074 | loss=0.12336 | 0.010s/iter | ETA: 2.2min
[Epoch 09] Iter  400/1074 | loss=0.41070 | 0.010s/iter | ETA: 2.2min
[Epoch 09] Iter  500/1074 | loss=0.21326 | 0.010s/iter | ETA: 2.2min
[Epoch 09] Iter  600/1074 | loss=0.09802 | 0.010s/iter | ETA: 2.1min
[Epoch 09] Iter  700/1074 | loss=0.14185 | 0.010s/iter | ETA: 2.1min
[Epoch 09] Iter  800/1074 | loss=0.23143 | 0.010s/iter | ETA: 2.1min
[Epoch 09] Iter  900/1074 | loss=0.13315 | 0.010s/iter | ETA: 2.1min
[Epoch 09] Iter 1000/1074 | loss=0.44166 | 0.011s/iter | ETA: 2.1min

------------------------------------------------------------
[Epoch 09] Summary | Time: 11.4s
Train Loss: 0.183256
Vali  Loss: 0.127160
Test  Loss: 0.178548
EarlyStopping counter: 5 out of 5
------------------------------------------------------------

************************************************************
EARLY STOPPING TRIGGERED
************************************************************
>>>>>>>testing : ETTm2_96_96_FreDEA_ETTm2_ftM_sl96_ll48_pl96_dm128_nh8_el1_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 11425
mse:0.17622697353363037, mae:0.25821366906166077, rmse:0.419793963432312
✅ 实验完成: MSE=0.176227, MAE=0.258214

================================================================================
运行实验: ETTm2 - Pred 192
================================================================================
命令: python -u run_longExp.py --data ETTm2 --data_path ETTm2.csv --model FreDEA --model_id ETTm2_96_192 --seq_len 96 --pred_len 192 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 256 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --dropout 0.05 --batch_size 32 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4

Args in experiment:
Namespace(is_training=1, train_only=False, model_id='ETTm2_96_192', model='FreDEA', data='ETTm2', root_path='./dataset/', data_path='ETTm2.csv', channel_independence=0, features='M', target='OT', freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=192, individual=False, embed_type=0, enc_in=7, dec_in=7, c_out=7, d_model=128, reg_lambda=0.01, wavelet='haar', n_bins=8, n_heads=8, e_layers=1, d_layers=1, d_ff=256, moving_avg=25, factor=1, distil=True, rev_affine=1, memory_size=64, bottleneck_dim=-1, ablation_freq=0, ablation_tea=0, ablation_cea=0, dropout=0.05, embed='timeF', activation='gelu', output_attention=False, do_predict=False, num_workers=4, itr=1, train_epochs=20, batch_size=32, patience=5, learning_rate=0.001, des='Exp', loss='mse', lradj='type1', use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=False, devices='0,1,2,3', test_flop=False)
Use GPU: cuda:0
>>>>>>>start training : ETTm2_96_192_FreDEA_ETTm2_ftM_sl96_ll48_pl192_dm128_nh8_el1_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0>>>>>>>>>>>>>>>>>>>>>>>>>>
train 34273
val 11329
test 11329

============================================================
TRAINING CONFIGURATION
============================================================
Model:           FreDEA
Dataset:         ETTm2
Seq/Pred Len:    96 -> 192
Batch Size:      32
Learning Rate:   0.001
Train Epochs:    20
Total Params:    138,964
d_model:         128
n_heads:         8
e_layers:        1
============================================================

[Epoch 01] Iter  100/1071 | loss=0.84862 | 0.022s/iter | ETA: 7.7min
[Epoch 01] Iter  200/1071 | loss=0.21789 | 0.014s/iter | ETA: 5.1min
[Epoch 01] Iter  300/1071 | loss=0.25256 | 0.015s/iter | ETA: 5.3min
[Epoch 01] Iter  400/1071 | loss=0.42755 | 0.016s/iter | ETA: 5.5min
[Epoch 01] Iter  500/1071 | loss=0.47457 | 0.016s/iter | ETA: 5.5min
[Epoch 01] Iter  600/1071 | loss=0.45773 | 0.016s/iter | ETA: 5.4min
[Epoch 01] Iter  700/1071 | loss=0.31980 | 0.015s/iter | ETA: 5.2min
[Epoch 01] Iter  800/1071 | loss=0.50293 | 0.012s/iter | ETA: 4.3min
[Epoch 01] Iter  900/1071 | loss=0.29918 | 0.012s/iter | ETA: 4.2min
[Epoch 01] Iter 1000/1071 | loss=0.72950 | 0.015s/iter | ETA: 5.0min

------------------------------------------------------------
[Epoch 01] Summary | Time: 15.8s
Train Loss: 0.333939
Vali  Loss: 0.175572
Test  Loss: 0.245480
Validation loss decreased (inf --> 0.175572).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.001
[Epoch 02] Iter  100/1071 | loss=0.20179 | 0.052s/iter | ETA: 17.5min
[Epoch 02] Iter  200/1071 | loss=0.32151 | 0.013s/iter | ETA: 4.4min
[Epoch 02] Iter  300/1071 | loss=0.67744 | 0.013s/iter | ETA: 4.4min
[Epoch 02] Iter  400/1071 | loss=0.24190 | 0.013s/iter | ETA: 4.3min
[Epoch 02] Iter  500/1071 | loss=0.13373 | 0.013s/iter | ETA: 4.3min
[Epoch 02] Iter  600/1071 | loss=0.26178 | 0.013s/iter | ETA: 4.3min
[Epoch 02] Iter  700/1071 | loss=0.44266 | 0.014s/iter | ETA: 4.6min
[Epoch 02] Iter  800/1071 | loss=0.26668 | 0.015s/iter | ETA: 4.7min
[Epoch 02] Iter  900/1071 | loss=0.16779 | 0.015s/iter | ETA: 4.7min
[Epoch 02] Iter 1000/1071 | loss=0.26822 | 0.014s/iter | ETA: 4.7min

------------------------------------------------------------
[Epoch 02] Summary | Time: 14.6s
Train Loss: 0.314037
Vali  Loss: 0.171896
Test  Loss: 0.244687
Validation loss decreased (0.175572 --> 0.171896).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 03] Iter  100/1071 | loss=0.41117 | 0.047s/iter | ETA: 15.0min
[Epoch 03] Iter  200/1071 | loss=0.17440 | 0.009s/iter | ETA: 2.9min
[Epoch 03] Iter  300/1071 | loss=0.31827 | 0.011s/iter | ETA: 3.4min
[Epoch 03] Iter  400/1071 | loss=0.55998 | 0.013s/iter | ETA: 4.2min
[Epoch 03] Iter  500/1071 | loss=0.51123 | 0.013s/iter | ETA: 4.1min
[Epoch 03] Iter  600/1071 | loss=0.27337 | 0.014s/iter | ETA: 4.3min
[Epoch 03] Iter  700/1071 | loss=0.22804 | 0.014s/iter | ETA: 4.4min
[Epoch 03] Iter  800/1071 | loss=0.18203 | 0.014s/iter | ETA: 4.4min
[Epoch 03] Iter  900/1071 | loss=0.21259 | 0.014s/iter | ETA: 4.4min
[Epoch 03] Iter 1000/1071 | loss=0.24740 | 0.012s/iter | ETA: 3.6min

------------------------------------------------------------
[Epoch 03] Summary | Time: 13.4s
Train Loss: 0.298918
Vali  Loss: 0.171448
Test  Loss: 0.243240
Validation loss decreased (0.171896 --> 0.171448).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.00025
[Epoch 04] Iter  100/1071 | loss=0.21018 | 0.045s/iter | ETA: 13.5min
[Epoch 04] Iter  200/1071 | loss=0.19264 | 0.009s/iter | ETA: 2.8min
[Epoch 04] Iter  300/1071 | loss=0.25622 | 0.009s/iter | ETA: 2.7min
[Epoch 04] Iter  400/1071 | loss=0.18087 | 0.009s/iter | ETA: 2.7min
[Epoch 04] Iter  500/1071 | loss=0.32893 | 0.009s/iter | ETA: 2.7min
[Epoch 04] Iter  600/1071 | loss=0.71103 | 0.008s/iter | ETA: 2.4min
[Epoch 04] Iter  700/1071 | loss=0.24562 | 0.011s/iter | ETA: 3.2min
[Epoch 04] Iter  800/1071 | loss=0.24409 | 0.012s/iter | ETA: 3.3min
[Epoch 04] Iter  900/1071 | loss=0.37792 | 0.014s/iter | ETA: 4.2min
[Epoch 04] Iter 1000/1071 | loss=0.35800 | 0.015s/iter | ETA: 4.4min

------------------------------------------------------------
[Epoch 04] Summary | Time: 11.7s
Train Loss: 0.287604
Vali  Loss: 0.173897
Test  Loss: 0.249662
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 0.000125
[Epoch 05] Iter  100/1071 | loss=0.24003 | 0.050s/iter | ETA: 14.2min
[Epoch 05] Iter  200/1071 | loss=0.34265 | 0.013s/iter | ETA: 3.7min
[Epoch 05] Iter  300/1071 | loss=0.30984 | 0.009s/iter | ETA: 2.6min
[Epoch 05] Iter  400/1071 | loss=0.26087 | 0.008s/iter | ETA: 2.2min
[Epoch 05] Iter  500/1071 | loss=0.20173 | 0.010s/iter | ETA: 2.7min
[Epoch 05] Iter  600/1071 | loss=0.32767 | 0.010s/iter | ETA: 2.9min
[Epoch 05] Iter  700/1071 | loss=0.26511 | 0.010s/iter | ETA: 2.9min
[Epoch 05] Iter  800/1071 | loss=0.51361 | 0.011s/iter | ETA: 3.1min
[Epoch 05] Iter  900/1071 | loss=0.23368 | 0.015s/iter | ETA: 4.0min
[Epoch 05] Iter 1000/1071 | loss=0.42818 | 0.016s/iter | ETA: 4.2min

------------------------------------------------------------
[Epoch 05] Summary | Time: 12.7s
Train Loss: 0.281264
Vali  Loss: 0.173268
Test  Loss: 0.246066
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 6.25e-05
[Epoch 06] Iter  100/1071 | loss=0.22635 | 0.050s/iter | ETA: 13.4min
[Epoch 06] Iter  200/1071 | loss=0.27542 | 0.013s/iter | ETA: 3.3min
[Epoch 06] Iter  300/1071 | loss=0.20994 | 0.013s/iter | ETA: 3.3min
[Epoch 06] Iter  400/1071 | loss=0.19472 | 0.013s/iter | ETA: 3.3min
[Epoch 06] Iter  500/1071 | loss=0.20154 | 0.012s/iter | ETA: 3.2min
[Epoch 06] Iter  600/1071 | loss=0.18016 | 0.013s/iter | ETA: 3.3min
[Epoch 06] Iter  700/1071 | loss=0.40917 | 0.012s/iter | ETA: 3.0min
[Epoch 06] Iter  800/1071 | loss=0.13756 | 0.011s/iter | ETA: 2.7min
[Epoch 06] Iter  900/1071 | loss=0.40095 | 0.013s/iter | ETA: 3.2min
[Epoch 06] Iter 1000/1071 | loss=0.36099 | 0.014s/iter | ETA: 3.6min

------------------------------------------------------------
[Epoch 06] Summary | Time: 13.6s
Train Loss: 0.278369
Vali  Loss: 0.171865
Test  Loss: 0.245646
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 3.125e-05
[Epoch 07] Iter  100/1071 | loss=0.43794 | 0.047s/iter | ETA: 11.5min
[Epoch 07] Iter  200/1071 | loss=0.18180 | 0.009s/iter | ETA: 2.1min
[Epoch 07] Iter  300/1071 | loss=0.14181 | 0.009s/iter | ETA: 2.1min
[Epoch 07] Iter  400/1071 | loss=0.29265 | 0.009s/iter | ETA: 2.1min
[Epoch 07] Iter  500/1071 | loss=0.35553 | 0.009s/iter | ETA: 2.1min
[Epoch 07] Iter  600/1071 | loss=0.24214 | 0.009s/iter | ETA: 2.1min
[Epoch 07] Iter  700/1071 | loss=0.27715 | 0.013s/iter | ETA: 3.0min
[Epoch 07] Iter  800/1071 | loss=0.28704 | 0.013s/iter | ETA: 3.1min
[Epoch 07] Iter  900/1071 | loss=0.35688 | 0.013s/iter | ETA: 3.1min
[Epoch 07] Iter 1000/1071 | loss=0.16406 | 0.014s/iter | ETA: 3.4min

------------------------------------------------------------
[Epoch 07] Summary | Time: 11.7s
Train Loss: 0.276539
Vali  Loss: 0.172879
Test  Loss: 0.246157
EarlyStopping counter: 4 out of 5
------------------------------------------------------------
Updating learning rate to 1.5625e-05
[Epoch 08] Iter  100/1071 | loss=0.23427 | 0.051s/iter | ETA: 11.8min
[Epoch 08] Iter  200/1071 | loss=0.23088 | 0.014s/iter | ETA: 3.3min
[Epoch 08] Iter  300/1071 | loss=0.15725 | 0.015s/iter | ETA: 3.3min
[Epoch 08] Iter  400/1071 | loss=0.17634 | 0.015s/iter | ETA: 3.4min
[Epoch 08] Iter  500/1071 | loss=0.25830 | 0.013s/iter | ETA: 3.0min
[Epoch 08] Iter  600/1071 | loss=0.21009 | 0.015s/iter | ETA: 3.2min
[Epoch 08] Iter  700/1071 | loss=0.24572 | 0.015s/iter | ETA: 3.2min
[Epoch 08] Iter  800/1071 | loss=0.24263 | 0.014s/iter | ETA: 3.0min
[Epoch 08] Iter  900/1071 | loss=0.39619 | 0.013s/iter | ETA: 2.8min
[Epoch 08] Iter 1000/1071 | loss=0.16614 | 0.014s/iter | ETA: 3.0min

------------------------------------------------------------
[Epoch 08] Summary | Time: 14.9s
Train Loss: 0.275512
Vali  Loss: 0.172381
Test  Loss: 0.245306
EarlyStopping counter: 5 out of 5
------------------------------------------------------------

************************************************************
EARLY STOPPING TRIGGERED
************************************************************
>>>>>>>testing : ETTm2_96_192_FreDEA_ETTm2_ftM_sl96_ll48_pl192_dm128_nh8_el1_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 11329
mse:0.24365505576133728, mae:0.30195605754852295, rmse:0.4936142861843109
✅ 实验完成: MSE=0.243655, MAE=0.301956

================================================================================
运行实验: ETTm2 - Pred 336
================================================================================
命令: python -u run_longExp.py --data ETTm2 --data_path ETTm2.csv --model FreDEA --model_id ETTm2_96_336 --seq_len 96 --pred_len 336 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 256 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --dropout 0.1 --batch_size 32 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4

Args in experiment:
Namespace(is_training=1, train_only=False, model_id='ETTm2_96_336', model='FreDEA', data='ETTm2', root_path='./dataset/', data_path='ETTm2.csv', channel_independence=0, features='M', target='OT', freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=336, individual=False, embed_type=0, enc_in=7, dec_in=7, c_out=7, d_model=128, reg_lambda=0.01, wavelet='haar', n_bins=8, n_heads=8, e_layers=1, d_layers=1, d_ff=256, moving_avg=25, factor=1, distil=True, rev_affine=1, memory_size=64, bottleneck_dim=-1, ablation_freq=0, ablation_tea=0, ablation_cea=0, dropout=0.1, embed='timeF', activation='gelu', output_attention=False, do_predict=False, num_workers=4, itr=1, train_epochs=20, batch_size=32, patience=5, learning_rate=0.001, des='Exp', loss='mse', lradj='type1', use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=False, devices='0,1,2,3', test_flop=False)
Use GPU: cuda:0
>>>>>>>start training : ETTm2_96_336_FreDEA_ETTm2_ftM_sl96_ll48_pl336_dm128_nh8_el1_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0>>>>>>>>>>>>>>>>>>>>>>>>>>
train 34129
val 11185
test 11185

============================================================
TRAINING CONFIGURATION
============================================================
Model:           FreDEA
Dataset:         ETTm2
Seq/Pred Len:    96 -> 336
Batch Size:      32
Learning Rate:   0.001
Train Epochs:    20
Total Params:    157,540
d_model:         128
n_heads:         8
e_layers:        1
============================================================

[Epoch 01] Iter  100/1066 | loss=0.67796 | 0.017s/iter | ETA: 6.2min
[Epoch 01] Iter  200/1066 | loss=0.58833 | 0.006s/iter | ETA: 2.1min
[Epoch 01] Iter  300/1066 | loss=0.40771 | 0.010s/iter | ETA: 3.6min
[Epoch 01] Iter  400/1066 | loss=0.30020 | 0.011s/iter | ETA: 3.7min
[Epoch 01] Iter  500/1066 | loss=0.27143 | 0.009s/iter | ETA: 3.3min
[Epoch 01] Iter  600/1066 | loss=0.33733 | 0.011s/iter | ETA: 3.8min
[Epoch 01] Iter  700/1066 | loss=0.26337 | 0.011s/iter | ETA: 3.7min
[Epoch 01] Iter  800/1066 | loss=0.46627 | 0.011s/iter | ETA: 3.7min
[Epoch 01] Iter  900/1066 | loss=0.23294 | 0.011s/iter | ETA: 3.6min
[Epoch 01] Iter 1000/1066 | loss=0.23754 | 0.011s/iter | ETA: 3.6min

------------------------------------------------------------
[Epoch 01] Summary | Time: 10.9s
Train Loss: 0.443882
Vali  Loss: 0.220029
Test  Loss: 0.303844
Validation loss decreased (inf --> 0.220029).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.001
[Epoch 02] Iter  100/1066 | loss=0.20267 | 0.046s/iter | ETA: 15.4min
[Epoch 02] Iter  200/1066 | loss=0.49593 | 0.011s/iter | ETA: 3.6min
[Epoch 02] Iter  300/1066 | loss=0.43418 | 0.011s/iter | ETA: 3.5min
[Epoch 02] Iter  400/1066 | loss=0.52806 | 0.011s/iter | ETA: 3.5min
[Epoch 02] Iter  500/1066 | loss=0.28531 | 0.011s/iter | ETA: 3.5min
[Epoch 02] Iter  600/1066 | loss=0.55478 | 0.011s/iter | ETA: 3.6min
[Epoch 02] Iter  700/1066 | loss=0.36631 | 0.011s/iter | ETA: 3.5min
[Epoch 02] Iter  800/1066 | loss=0.58578 | 0.011s/iter | ETA: 3.5min
[Epoch 02] Iter  900/1066 | loss=0.30464 | 0.010s/iter | ETA: 3.3min
[Epoch 02] Iter 1000/1066 | loss=0.43600 | 0.010s/iter | ETA: 3.3min

------------------------------------------------------------
[Epoch 02] Summary | Time: 11.5s
Train Loss: 0.421465
Vali  Loss: 0.218853
Test  Loss: 0.303192
Validation loss decreased (0.220029 --> 0.218853).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 03] Iter  100/1066 | loss=0.26210 | 0.049s/iter | ETA: 15.6min
[Epoch 03] Iter  200/1066 | loss=0.39683 | 0.012s/iter | ETA: 3.8min
[Epoch 03] Iter  300/1066 | loss=0.57782 | 0.012s/iter | ETA: 3.8min
[Epoch 03] Iter  400/1066 | loss=0.60982 | 0.012s/iter | ETA: 3.8min
[Epoch 03] Iter  500/1066 | loss=0.67683 | 0.012s/iter | ETA: 3.7min
[Epoch 03] Iter  600/1066 | loss=0.35603 | 0.012s/iter | ETA: 3.8min
[Epoch 03] Iter  700/1066 | loss=0.50258 | 0.012s/iter | ETA: 3.7min
[Epoch 03] Iter  800/1066 | loss=0.28479 | 0.011s/iter | ETA: 3.5min
[Epoch 03] Iter  900/1066 | loss=0.51609 | 0.010s/iter | ETA: 3.0min
[Epoch 03] Iter 1000/1066 | loss=0.25915 | 0.012s/iter | ETA: 3.6min

------------------------------------------------------------
[Epoch 03] Summary | Time: 12.7s
Train Loss: 0.403974
Vali  Loss: 0.219902
Test  Loss: 0.301874
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 0.00025
[Epoch 04] Iter  100/1066 | loss=0.27012 | 0.047s/iter | ETA: 14.2min
[Epoch 04] Iter  200/1066 | loss=0.36848 | 0.014s/iter | ETA: 4.1min
[Epoch 04] Iter  300/1066 | loss=0.33270 | 0.016s/iter | ETA: 4.6min
[Epoch 04] Iter  400/1066 | loss=0.60863 | 0.010s/iter | ETA: 3.0min
[Epoch 04] Iter  500/1066 | loss=0.36562 | 0.012s/iter | ETA: 3.6min
[Epoch 04] Iter  600/1066 | loss=0.21057 | 0.012s/iter | ETA: 3.6min
[Epoch 04] Iter  700/1066 | loss=0.22507 | 0.013s/iter | ETA: 3.8min
[Epoch 04] Iter  800/1066 | loss=0.34527 | 0.015s/iter | ETA: 4.2min
[Epoch 04] Iter  900/1066 | loss=0.25655 | 0.015s/iter | ETA: 4.2min
[Epoch 04] Iter 1000/1066 | loss=0.24133 | 0.015s/iter | ETA: 4.2min

------------------------------------------------------------
[Epoch 04] Summary | Time: 14.2s
Train Loss: 0.386504
Vali  Loss: 0.220065
Test  Loss: 0.302411
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 0.000125
[Epoch 05] Iter  100/1066 | loss=0.20482 | 0.048s/iter | ETA: 13.5min
[Epoch 05] Iter  200/1066 | loss=0.49700 | 0.013s/iter | ETA: 3.6min
[Epoch 05] Iter  300/1066 | loss=0.18381 | 0.011s/iter | ETA: 3.1min
[Epoch 05] Iter  400/1066 | loss=0.25615 | 0.012s/iter | ETA: 3.4min
[Epoch 05] Iter  500/1066 | loss=0.39182 | 0.012s/iter | ETA: 3.4min
[Epoch 05] Iter  600/1066 | loss=0.28553 | 0.011s/iter | ETA: 3.0min
[Epoch 05] Iter  700/1066 | loss=0.46985 | 0.012s/iter | ETA: 3.3min
[Epoch 05] Iter  800/1066 | loss=0.29048 | 0.012s/iter | ETA: 3.3min
[Epoch 05] Iter  900/1066 | loss=0.27654 | 0.012s/iter | ETA: 3.3min
[Epoch 05] Iter 1000/1066 | loss=0.28332 | 0.012s/iter | ETA: 3.2min

------------------------------------------------------------
[Epoch 05] Summary | Time: 12.7s
Train Loss: 0.373739
Vali  Loss: 0.219222
Test  Loss: 0.303154
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 6.25e-05
[Epoch 06] Iter  100/1066 | loss=0.53810 | 0.047s/iter | ETA: 12.6min
[Epoch 06] Iter  200/1066 | loss=0.28125 | 0.009s/iter | ETA: 2.4min
[Epoch 06] Iter  300/1066 | loss=0.35112 | 0.011s/iter | ETA: 3.0min
[Epoch 06] Iter  400/1066 | loss=0.19344 | 0.013s/iter | ETA: 3.5min
[Epoch 06] Iter  500/1066 | loss=0.21391 | 0.015s/iter | ETA: 3.7min
[Epoch 06] Iter  600/1066 | loss=0.82566 | 0.014s/iter | ETA: 3.6min
[Epoch 06] Iter  700/1066 | loss=0.56265 | 0.012s/iter | ETA: 3.2min
[Epoch 06] Iter  800/1066 | loss=0.50512 | 0.013s/iter | ETA: 3.3min
[Epoch 06] Iter  900/1066 | loss=0.34821 | 0.013s/iter | ETA: 3.2min
[Epoch 06] Iter 1000/1066 | loss=0.21332 | 0.013s/iter | ETA: 3.2min

------------------------------------------------------------
[Epoch 06] Summary | Time: 13.1s
Train Loss: 0.368551
Vali  Loss: 0.218302
Test  Loss: 0.301485
Validation loss decreased (0.218853 --> 0.218302).  Saving model ...
------------------------------------------------------------
Updating learning rate to 3.125e-05
[Epoch 07] Iter  100/1066 | loss=0.44273 | 0.048s/iter | ETA: 11.9min
[Epoch 07] Iter  200/1066 | loss=0.23343 | 0.013s/iter | ETA: 3.1min
[Epoch 07] Iter  300/1066 | loss=0.29034 | 0.015s/iter | ETA: 3.6min
[Epoch 07] Iter  400/1066 | loss=0.57419 | 0.014s/iter | ETA: 3.4min
[Epoch 07] Iter  500/1066 | loss=0.32571 | 0.014s/iter | ETA: 3.3min
[Epoch 07] Iter  600/1066 | loss=0.55491 | 0.014s/iter | ETA: 3.3min
[Epoch 07] Iter  700/1066 | loss=0.24379 | 0.014s/iter | ETA: 3.2min
[Epoch 07] Iter  800/1066 | loss=0.24253 | 0.014s/iter | ETA: 3.2min
[Epoch 07] Iter  900/1066 | loss=0.29105 | 0.014s/iter | ETA: 3.2min
[Epoch 07] Iter 1000/1066 | loss=0.27091 | 0.014s/iter | ETA: 3.2min

------------------------------------------------------------
[Epoch 07] Summary | Time: 14.7s
Train Loss: 0.366022
Vali  Loss: 0.219285
Test  Loss: 0.303480
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 1.5625e-05
[Epoch 08] Iter  100/1066 | loss=0.23428 | 0.049s/iter | ETA: 11.1min
[Epoch 08] Iter  200/1066 | loss=0.28366 | 0.009s/iter | ETA: 2.1min
[Epoch 08] Iter  300/1066 | loss=0.57318 | 0.009s/iter | ETA: 2.1min
[Epoch 08] Iter  400/1066 | loss=0.31317 | 0.009s/iter | ETA: 2.0min
[Epoch 08] Iter  500/1066 | loss=0.46428 | 0.009s/iter | ETA: 2.0min
[Epoch 08] Iter  600/1066 | loss=0.56990 | 0.012s/iter | ETA: 2.6min
[Epoch 08] Iter  700/1066 | loss=0.45801 | 0.013s/iter | ETA: 2.8min
[Epoch 08] Iter  800/1066 | loss=0.23204 | 0.012s/iter | ETA: 2.7min
[Epoch 08] Iter  900/1066 | loss=0.26468 | 0.014s/iter | ETA: 2.9min
[Epoch 08] Iter 1000/1066 | loss=0.21375 | 0.014s/iter | ETA: 2.9min

------------------------------------------------------------
[Epoch 08] Summary | Time: 12.1s
Train Loss: 0.364040
Vali  Loss: 0.219434
Test  Loss: 0.303317
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 7.8125e-06
[Epoch 09] Iter  100/1066 | loss=0.60698 | 0.049s/iter | ETA: 10.4min
[Epoch 09] Iter  200/1066 | loss=0.28001 | 0.009s/iter | ETA: 1.9min
[Epoch 09] Iter  300/1066 | loss=0.22003 | 0.009s/iter | ETA: 1.9min
[Epoch 09] Iter  400/1066 | loss=0.28633 | 0.009s/iter | ETA: 1.9min
[Epoch 09] Iter  500/1066 | loss=0.51241 | 0.009s/iter | ETA: 1.9min
[Epoch 09] Iter  600/1066 | loss=0.65697 | 0.011s/iter | ETA: 2.3min
[Epoch 09] Iter  700/1066 | loss=0.37488 | 0.013s/iter | ETA: 2.7min
[Epoch 09] Iter  800/1066 | loss=0.24699 | 0.013s/iter | ETA: 2.6min
[Epoch 09] Iter  900/1066 | loss=0.43470 | 0.013s/iter | ETA: 2.6min
[Epoch 09] Iter 1000/1066 | loss=0.27324 | 0.013s/iter | ETA: 2.6min

------------------------------------------------------------
[Epoch 09] Summary | Time: 12.0s
Train Loss: 0.363910
Vali  Loss: 0.219381
Test  Loss: 0.303193
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 3.90625e-06
[Epoch 10] Iter  100/1066 | loss=0.24906 | 0.051s/iter | ETA: 9.9min
[Epoch 10] Iter  200/1066 | loss=0.28435 | 0.011s/iter | ETA: 2.0min
[Epoch 10] Iter  300/1066 | loss=0.39811 | 0.010s/iter | ETA: 2.0min
[Epoch 10] Iter  400/1066 | loss=0.23030 | 0.012s/iter | ETA: 2.2min
[Epoch 10] Iter  500/1066 | loss=0.23403 | 0.011s/iter | ETA: 2.0min
[Epoch 10] Iter  600/1066 | loss=0.27578 | 0.012s/iter | ETA: 2.2min
[Epoch 10] Iter  700/1066 | loss=0.33742 | 0.012s/iter | ETA: 2.2min
[Epoch 10] Iter  800/1066 | loss=0.20742 | 0.012s/iter | ETA: 2.1min
[Epoch 10] Iter  900/1066 | loss=0.32826 | 0.012s/iter | ETA: 2.3min
[Epoch 10] Iter 1000/1066 | loss=0.24874 | 0.014s/iter | ETA: 2.5min

------------------------------------------------------------
[Epoch 10] Summary | Time: 12.9s
Train Loss: 0.363070
Vali  Loss: 0.219147
Test  Loss: 0.302990
EarlyStopping counter: 4 out of 5
------------------------------------------------------------
Updating learning rate to 1.953125e-06
[Epoch 11] Iter  100/1066 | loss=0.26401 | 0.048s/iter | ETA: 8.5min
[Epoch 11] Iter  200/1066 | loss=0.30235 | 0.009s/iter | ETA: 1.6min
[Epoch 11] Iter  300/1066 | loss=0.61265 | 0.009s/iter | ETA: 1.5min
[Epoch 11] Iter  400/1066 | loss=0.55368 | 0.011s/iter | ETA: 1.8min
[Epoch 11] Iter  500/1066 | loss=0.30277 | 0.015s/iter | ETA: 2.5min
[Epoch 11] Iter  600/1066 | loss=0.35041 | 0.015s/iter | ETA: 2.6min
[Epoch 11] Iter  700/1066 | loss=0.24617 | 0.015s/iter | ETA: 2.4min
[Epoch 11] Iter  800/1066 | loss=0.16014 | 0.015s/iter | ETA: 2.4min
[Epoch 11] Iter  900/1066 | loss=0.35732 | 0.015s/iter | ETA: 2.4min
[Epoch 11] Iter 1000/1066 | loss=0.31639 | 0.015s/iter | ETA: 2.4min

------------------------------------------------------------
[Epoch 11] Summary | Time: 13.8s
Train Loss: 0.362715
Vali  Loss: 0.219147
Test  Loss: 0.302821
EarlyStopping counter: 5 out of 5
------------------------------------------------------------

************************************************************
EARLY STOPPING TRIGGERED
************************************************************
>>>>>>>testing : ETTm2_96_336_FreDEA_ETTm2_ftM_sl96_ll48_pl336_dm128_nh8_el1_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 11185
mse:0.30170947313308716, mae:0.3388698101043701, rmse:0.5492808818817139
✅ 实验完成: MSE=0.301709, MAE=0.338870

================================================================================
运行实验: ETTm2 - Pred 720
================================================================================
命令: python -u run_longExp.py --data ETTm2 --data_path ETTm2.csv --model FreDEA --model_id ETTm2_96_720 --seq_len 96 --pred_len 720 --enc_in 7 --dec_in 7 --c_out 7 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim 1 --dropout 0.3 --batch_size 32 --learning_rate 0.0003 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4

Args in experiment:
Namespace(is_training=1, train_only=False, model_id='ETTm2_96_720', model='FreDEA', data='ETTm2', root_path='./dataset/', data_path='ETTm2.csv', channel_independence=0, features='M', target='OT', freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=720, individual=False, embed_type=0, enc_in=7, dec_in=7, c_out=7, d_model=64, reg_lambda=0.01, wavelet='haar', n_bins=8, n_heads=8, e_layers=1, d_layers=1, d_ff=128, moving_avg=25, factor=1, distil=True, rev_affine=1, memory_size=64, bottleneck_dim=1, ablation_freq=0, ablation_tea=0, ablation_cea=0, dropout=0.3, embed='timeF', activation='gelu', output_attention=False, do_predict=False, num_workers=4, itr=1, train_epochs=20, batch_size=32, patience=5, learning_rate=0.0003, des='Exp', loss='mse', lradj='type1', use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=False, devices='0,1,2,3', test_flop=False)
Use GPU: cuda:0
>>>>>>>start training : ETTm2_96_720_FreDEA_ETTm2_ftM_sl96_ll48_pl720_dm64_nh8_el1_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0>>>>>>>>>>>>>>>>>>>>>>>>>>
train 33745
val 10801
test 10801

============================================================
TRAINING CONFIGURATION
============================================================
Model:           FreDEA
Dataset:         ETTm2
Seq/Pred Len:    96 -> 720
Batch Size:      32
Learning Rate:   0.0003
Train Epochs:    20
Total Params:    92,516
d_model:         64
n_heads:         8
e_layers:        1
============================================================

[Epoch 01] Iter  100/1054 | loss=0.84904 | 0.021s/iter | ETA: 7.2min
[Epoch 01] Iter  200/1054 | loss=0.56879 | 0.011s/iter | ETA: 3.9min
[Epoch 01] Iter  300/1054 | loss=1.01499 | 0.014s/iter | ETA: 4.7min
[Epoch 01] Iter  400/1054 | loss=0.36444 | 0.014s/iter | ETA: 4.7min
[Epoch 01] Iter  500/1054 | loss=0.44350 | 0.014s/iter | ETA: 4.7min
[Epoch 01] Iter  600/1054 | loss=0.43340 | 0.013s/iter | ETA: 4.4min
[Epoch 01] Iter  700/1054 | loss=0.37669 | 0.015s/iter | ETA: 4.9min
[Epoch 01] Iter  800/1054 | loss=0.46477 | 0.014s/iter | ETA: 4.9min
[Epoch 01] Iter  900/1054 | loss=0.37623 | 0.014s/iter | ETA: 4.9min
[Epoch 01] Iter 1000/1054 | loss=0.34928 | 0.012s/iter | ETA: 4.1min

------------------------------------------------------------
[Epoch 01] Summary | Time: 14.3s
Train Loss: 0.601204
Vali  Loss: 0.297055
Test  Loss: 0.411567
Validation loss decreased (inf --> 0.297055).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0003
[Epoch 02] Iter  100/1054 | loss=0.30987 | 0.053s/iter | ETA: 17.5min
[Epoch 02] Iter  200/1054 | loss=0.52014 | 0.013s/iter | ETA: 4.3min
[Epoch 02] Iter  300/1054 | loss=0.98844 | 0.013s/iter | ETA: 4.3min
[Epoch 02] Iter  400/1054 | loss=0.42419 | 0.012s/iter | ETA: 4.0min
[Epoch 02] Iter  500/1054 | loss=0.31444 | 0.013s/iter | ETA: 4.1min
[Epoch 02] Iter  600/1054 | loss=0.36498 | 0.013s/iter | ETA: 4.4min
[Epoch 02] Iter  700/1054 | loss=0.39757 | 0.015s/iter | ETA: 4.8min
[Epoch 02] Iter  800/1054 | loss=0.95503 | 0.014s/iter | ETA: 4.6min
[Epoch 02] Iter  900/1054 | loss=0.88057 | 0.015s/iter | ETA: 4.8min
[Epoch 02] Iter 1000/1054 | loss=0.80007 | 0.015s/iter | ETA: 4.8min

------------------------------------------------------------
[Epoch 02] Summary | Time: 14.4s
Train Loss: 0.580115
Vali  Loss: 0.295517
Test  Loss: 0.404541
Validation loss decreased (0.297055 --> 0.295517).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.00015
[Epoch 03] Iter  100/1054 | loss=0.54424 | 0.049s/iter | ETA: 15.3min
[Epoch 03] Iter  200/1054 | loss=0.45321 | 0.012s/iter | ETA: 3.7min
[Epoch 03] Iter  300/1054 | loss=0.38720 | 0.012s/iter | ETA: 3.8min
[Epoch 03] Iter  400/1054 | loss=0.87959 | 0.011s/iter | ETA: 3.3min
[Epoch 03] Iter  500/1054 | loss=0.86923 | 0.012s/iter | ETA: 3.7min
[Epoch 03] Iter  600/1054 | loss=0.33210 | 0.012s/iter | ETA: 3.7min
[Epoch 03] Iter  700/1054 | loss=0.49168 | 0.012s/iter | ETA: 3.7min
[Epoch 03] Iter  800/1054 | loss=0.33927 | 0.012s/iter | ETA: 3.7min
[Epoch 03] Iter  900/1054 | loss=0.49085 | 0.012s/iter | ETA: 3.6min
[Epoch 03] Iter 1000/1054 | loss=0.41886 | 0.012s/iter | ETA: 3.6min

------------------------------------------------------------
[Epoch 03] Summary | Time: 12.7s
Train Loss: 0.572329
Vali  Loss: 0.293766
Test  Loss: 0.403017
Validation loss decreased (0.295517 --> 0.293766).  Saving model ...
------------------------------------------------------------
Updating learning rate to 7.5e-05
[Epoch 04] Iter  100/1054 | loss=0.67686 | 0.047s/iter | ETA: 14.1min
[Epoch 04] Iter  200/1054 | loss=0.61445 | 0.009s/iter | ETA: 2.7min
[Epoch 04] Iter  300/1054 | loss=0.81041 | 0.009s/iter | ETA: 2.7min
[Epoch 04] Iter  400/1054 | loss=0.66702 | 0.009s/iter | ETA: 2.7min
[Epoch 04] Iter  500/1054 | loss=0.50517 | 0.009s/iter | ETA: 2.7min
[Epoch 04] Iter  600/1054 | loss=0.72630 | 0.009s/iter | ETA: 2.6min
[Epoch 04] Iter  700/1054 | loss=0.83617 | 0.012s/iter | ETA: 3.4min
[Epoch 04] Iter  800/1054 | loss=0.50239 | 0.014s/iter | ETA: 4.0min
[Epoch 04] Iter  900/1054 | loss=0.53681 | 0.014s/iter | ETA: 4.0min
[Epoch 04] Iter 1000/1054 | loss=0.67487 | 0.014s/iter | ETA: 4.0min

------------------------------------------------------------
[Epoch 04] Summary | Time: 11.6s
Train Loss: 0.568651
Vali  Loss: 0.292638
Test  Loss: 0.402561
Validation loss decreased (0.293766 --> 0.292638).  Saving model ...
------------------------------------------------------------
Updating learning rate to 3.75e-05
[Epoch 05] Iter  100/1054 | loss=0.44224 | 0.046s/iter | ETA: 12.8min
[Epoch 05] Iter  200/1054 | loss=0.66650 | 0.011s/iter | ETA: 3.1min
[Epoch 05] Iter  300/1054 | loss=0.34489 | 0.011s/iter | ETA: 3.1min
[Epoch 05] Iter  400/1054 | loss=0.42554 | 0.011s/iter | ETA: 3.1min
[Epoch 05] Iter  500/1054 | loss=0.67763 | 0.013s/iter | ETA: 3.6min
[Epoch 05] Iter  600/1054 | loss=0.56998 | 0.014s/iter | ETA: 3.7min
[Epoch 05] Iter  700/1054 | loss=0.38361 | 0.012s/iter | ETA: 3.2min
[Epoch 05] Iter  800/1054 | loss=0.62856 | 0.011s/iter | ETA: 2.9min
[Epoch 05] Iter  900/1054 | loss=0.29257 | 0.011s/iter | ETA: 2.8min
[Epoch 05] Iter 1000/1054 | loss=0.61989 | 0.012s/iter | ETA: 3.1min

------------------------------------------------------------
[Epoch 05] Summary | Time: 12.4s
Train Loss: 0.566930
Vali  Loss: 0.293267
Test  Loss: 0.402142
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 1.875e-05
[Epoch 06] Iter  100/1054 | loss=0.40159 | 0.046s/iter | ETA: 12.0min
[Epoch 06] Iter  200/1054 | loss=0.62230 | 0.009s/iter | ETA: 2.4min
[Epoch 06] Iter  300/1054 | loss=0.83457 | 0.009s/iter | ETA: 2.3min
[Epoch 06] Iter  400/1054 | loss=0.51479 | 0.010s/iter | ETA: 2.6min
[Epoch 06] Iter  500/1054 | loss=0.29409 | 0.012s/iter | ETA: 3.2min
[Epoch 06] Iter  600/1054 | loss=0.68996 | 0.013s/iter | ETA: 3.3min
[Epoch 06] Iter  700/1054 | loss=0.48536 | 0.013s/iter | ETA: 3.3min
[Epoch 06] Iter  800/1054 | loss=0.73360 | 0.013s/iter | ETA: 3.2min
[Epoch 06] Iter  900/1054 | loss=0.34145 | 0.013s/iter | ETA: 3.2min
[Epoch 06] Iter 1000/1054 | loss=0.61440 | 0.013s/iter | ETA: 3.2min

------------------------------------------------------------
[Epoch 06] Summary | Time: 12.3s
Train Loss: 0.566283
Vali  Loss: 0.292387
Test  Loss: 0.401494
Validation loss decreased (0.292638 --> 0.292387).  Saving model ...
------------------------------------------------------------
Updating learning rate to 9.375e-06
[Epoch 07] Iter  100/1054 | loss=0.49098 | 0.046s/iter | ETA: 11.3min
[Epoch 07] Iter  200/1054 | loss=0.54752 | 0.009s/iter | ETA: 2.2min
[Epoch 07] Iter  300/1054 | loss=0.70557 | 0.009s/iter | ETA: 2.2min
[Epoch 07] Iter  400/1054 | loss=0.81117 | 0.012s/iter | ETA: 2.9min
[Epoch 07] Iter  500/1054 | loss=0.61295 | 0.014s/iter | ETA: 3.3min
[Epoch 07] Iter  600/1054 | loss=0.39655 | 0.014s/iter | ETA: 3.3min
[Epoch 07] Iter  700/1054 | loss=0.67646 | 0.014s/iter | ETA: 3.3min
[Epoch 07] Iter  800/1054 | loss=0.37862 | 0.012s/iter | ETA: 2.8min
[Epoch 07] Iter  900/1054 | loss=0.56418 | 0.012s/iter | ETA: 2.9min
[Epoch 07] Iter 1000/1054 | loss=0.29193 | 0.012s/iter | ETA: 2.7min

------------------------------------------------------------
[Epoch 07] Summary | Time: 12.5s
Train Loss: 0.566067
Vali  Loss: 0.292789
Test  Loss: 0.401734
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 4.6875e-06
[Epoch 08] Iter  100/1054 | loss=0.57354 | 0.052s/iter | ETA: 11.7min
[Epoch 08] Iter  200/1054 | loss=0.42526 | 0.012s/iter | ETA: 2.7min
[Epoch 08] Iter  300/1054 | loss=0.74064 | 0.012s/iter | ETA: 2.6min
[Epoch 08] Iter  400/1054 | loss=0.38153 | 0.012s/iter | ETA: 2.6min
[Epoch 08] Iter  500/1054 | loss=0.39359 | 0.012s/iter | ETA: 2.7min
[Epoch 08] Iter  600/1054 | loss=0.56604 | 0.011s/iter | ETA: 2.5min
[Epoch 08] Iter  700/1054 | loss=0.40640 | 0.011s/iter | ETA: 2.4min
[Epoch 08] Iter  800/1054 | loss=0.56462 | 0.014s/iter | ETA: 2.9min
[Epoch 08] Iter  900/1054 | loss=0.70612 | 0.014s/iter | ETA: 2.9min
[Epoch 08] Iter 1000/1054 | loss=0.56821 | 0.014s/iter | ETA: 2.9min

------------------------------------------------------------
[Epoch 08] Summary | Time: 13.1s
Train Loss: 0.564496
Vali  Loss: 0.292749
Test  Loss: 0.401753
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 2.34375e-06
[Epoch 09] Iter  100/1054 | loss=0.56148 | 0.049s/iter | ETA: 10.2min
[Epoch 09] Iter  200/1054 | loss=1.06082 | 0.009s/iter | ETA: 1.9min
[Epoch 09] Iter  300/1054 | loss=0.54036 | 0.009s/iter | ETA: 1.9min
[Epoch 09] Iter  400/1054 | loss=0.41079 | 0.009s/iter | ETA: 1.9min
[Epoch 09] Iter  500/1054 | loss=0.91328 | 0.013s/iter | ETA: 2.7min
[Epoch 09] Iter  600/1054 | loss=0.36692 | 0.014s/iter | ETA: 2.8min
[Epoch 09] Iter  700/1054 | loss=0.96345 | 0.014s/iter | ETA: 2.7min
[Epoch 09] Iter  800/1054 | loss=0.47932 | 0.014s/iter | ETA: 2.7min
[Epoch 09] Iter  900/1054 | loss=0.36621 | 0.014s/iter | ETA: 2.7min
[Epoch 09] Iter 1000/1054 | loss=0.44963 | 0.014s/iter | ETA: 2.7min

------------------------------------------------------------
[Epoch 09] Summary | Time: 12.8s
Train Loss: 0.565119
Vali  Loss: 0.292822
Test  Loss: 0.401764
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 1.171875e-06
[Epoch 10] Iter  100/1054 | loss=0.33549 | 0.052s/iter | ETA: 9.9min
[Epoch 10] Iter  200/1054 | loss=0.35816 | 0.014s/iter | ETA: 2.6min
[Epoch 10] Iter  300/1054 | loss=0.61799 | 0.014s/iter | ETA: 2.7min
[Epoch 10] Iter  400/1054 | loss=0.49540 | 0.015s/iter | ETA: 2.7min
[Epoch 10] Iter  500/1054 | loss=0.41126 | 0.015s/iter | ETA: 2.8min
[Epoch 10] Iter  600/1054 | loss=0.76708 | 0.015s/iter | ETA: 2.8min
[Epoch 10] Iter  700/1054 | loss=0.34575 | 0.015s/iter | ETA: 2.8min
[Epoch 10] Iter  800/1054 | loss=0.48951 | 0.016s/iter | ETA: 2.9min
[Epoch 10] Iter  900/1054 | loss=0.42278 | 0.016s/iter | ETA: 2.8min
[Epoch 10] Iter 1000/1054 | loss=0.35123 | 0.016s/iter | ETA: 2.8min

------------------------------------------------------------
[Epoch 10] Summary | Time: 15.8s
Train Loss: 0.565907
Vali  Loss: 0.292854
Test  Loss: 0.401763
EarlyStopping counter: 4 out of 5
------------------------------------------------------------
Updating learning rate to 5.859375e-07
[Epoch 11] Iter  100/1054 | loss=0.48769 | 0.050s/iter | ETA: 8.8min
[Epoch 11] Iter  200/1054 | loss=0.69289 | 0.009s/iter | ETA: 1.6min
[Epoch 11] Iter  300/1054 | loss=0.57736 | 0.008s/iter | ETA: 1.4min
[Epoch 11] Iter  400/1054 | loss=0.38239 | 0.009s/iter | ETA: 1.4min
[Epoch 11] Iter  500/1054 | loss=0.58447 | 0.009s/iter | ETA: 1.4min
[Epoch 11] Iter  600/1054 | loss=0.80584 | 0.008s/iter | ETA: 1.4min
[Epoch 11] Iter  700/1054 | loss=0.36374 | 0.009s/iter | ETA: 1.4min
[Epoch 11] Iter  800/1054 | loss=0.62115 | 0.009s/iter | ETA: 1.5min
[Epoch 11] Iter  900/1054 | loss=0.32739 | 0.009s/iter | ETA: 1.5min
[Epoch 11] Iter 1000/1054 | loss=0.46609 | 0.011s/iter | ETA: 1.7min

------------------------------------------------------------
[Epoch 11] Summary | Time: 9.9s
Train Loss: 0.564510
Vali  Loss: 0.292638
Test  Loss: 0.401760
EarlyStopping counter: 5 out of 5
------------------------------------------------------------

************************************************************
EARLY STOPPING TRIGGERED
************************************************************
>>>>>>>testing : ETTm2_96_720_FreDEA_ETTm2_ftM_sl96_ll48_pl720_dm64_nh8_el1_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 10801
mse:0.40098652243614197, mae:0.40008530020713806, rmse:0.633234977722168
✅ 实验完成: MSE=0.400987, MAE=0.400085

================================================================================
运行实验: ETTh1 - Pred 96
================================================================================
命令: python -u run_longExp.py --data ETTh1 --data_path ETTh1.csv --model FreDEA --model_id ETTh1_96_96 --seq_len 96 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --dropout 0.0 --batch_size 32 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4

Args in experiment:
Namespace(is_training=1, train_only=False, model_id='ETTh1_96_96', model='FreDEA', data='ETTh1', root_path='./dataset/', data_path='ETTh1.csv', channel_independence=0, features='M', target='OT', freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=96, individual=False, embed_type=0, enc_in=7, dec_in=7, c_out=7, d_model=64, reg_lambda=0.01, wavelet='haar', n_bins=8, n_heads=8, e_layers=1, d_layers=1, d_ff=128, moving_avg=25, factor=1, distil=True, rev_affine=1, memory_size=64, bottleneck_dim=-1, ablation_freq=0, ablation_tea=0, ablation_cea=0, dropout=0.0, embed='timeF', activation='gelu', output_attention=False, do_predict=False, num_workers=4, itr=1, train_epochs=20, batch_size=32, patience=5, learning_rate=0.001, des='Exp', loss='mse', lradj='type1', use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=False, devices='0,1,2,3', test_flop=False)
Use GPU: cuda:0
>>>>>>>start training : ETTh1_96_96_FreDEA_ETTh1_ftM_sl96_ll48_pl96_dm64_nh8_el1_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0>>>>>>>>>>>>>>>>>>>>>>>>>>
train 8449
val 2785
test 2785

============================================================
TRAINING CONFIGURATION
============================================================
Model:           FreDEA
Dataset:         ETTh1
Seq/Pred Len:    96 -> 96
Batch Size:      32
Learning Rate:   0.001
Train Epochs:    20
Total Params:    51,956
d_model:         64
n_heads:         8
e_layers:        1
============================================================

[Epoch 01] Iter  100/264 | loss=0.45608 | 0.020s/iter | ETA: 1.7min
[Epoch 01] Iter  200/264 | loss=0.29447 | 0.008s/iter | ETA: 0.7min

------------------------------------------------------------
[Epoch 01] Summary | Time: 2.7s
Train Loss: 0.424162
Vali  Loss: 0.749592
Test  Loss: 0.408897
Validation loss decreased (inf --> 0.749592).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.001
[Epoch 02] Iter  100/264 | loss=0.33719 | 0.024s/iter | ETA: 2.0min
[Epoch 02] Iter  200/264 | loss=0.37392 | 0.010s/iter | ETA: 0.8min

------------------------------------------------------------
[Epoch 02] Summary | Time: 2.8s
Train Loss: 0.363735
Vali  Loss: 0.706526
Test  Loss: 0.394414
Validation loss decreased (0.749592 --> 0.706526).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 03] Iter  100/264 | loss=0.34128 | 0.024s/iter | ETA: 1.9min
[Epoch 03] Iter  200/264 | loss=0.33567 | 0.008s/iter | ETA: 0.6min

------------------------------------------------------------
[Epoch 03] Summary | Time: 2.3s
Train Loss: 0.349951
Vali  Loss: 0.693692
Test  Loss: 0.386704
Validation loss decreased (0.706526 --> 0.693692).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.00025
[Epoch 04] Iter  100/264 | loss=0.38718 | 0.023s/iter | ETA: 1.7min
[Epoch 04] Iter  200/264 | loss=0.31921 | 0.009s/iter | ETA: 0.6min

------------------------------------------------------------
[Epoch 04] Summary | Time: 2.4s
Train Loss: 0.343543
Vali  Loss: 0.686259
Test  Loss: 0.392249
Validation loss decreased (0.693692 --> 0.686259).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.000125
[Epoch 05] Iter  100/264 | loss=0.32727 | 0.022s/iter | ETA: 1.5min
[Epoch 05] Iter  200/264 | loss=0.32219 | 0.009s/iter | ETA: 0.6min

------------------------------------------------------------
[Epoch 05] Summary | Time: 2.3s
Train Loss: 0.340054
Vali  Loss: 0.689128
Test  Loss: 0.387972
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 6.25e-05
[Epoch 06] Iter  100/264 | loss=0.38423 | 0.023s/iter | ETA: 1.5min
[Epoch 06] Iter  200/264 | loss=0.38924 | 0.008s/iter | ETA: 0.5min

------------------------------------------------------------
[Epoch 06] Summary | Time: 2.3s
Train Loss: 0.338253
Vali  Loss: 0.689588
Test  Loss: 0.386951
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 3.125e-05
[Epoch 07] Iter  100/264 | loss=0.34581 | 0.023s/iter | ETA: 1.4min
[Epoch 07] Iter  200/264 | loss=0.35120 | 0.009s/iter | ETA: 0.5min

------------------------------------------------------------
[Epoch 07] Summary | Time: 2.4s
Train Loss: 0.337322
Vali  Loss: 0.690264
Test  Loss: 0.386211
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 1.5625e-05
[Epoch 08] Iter  100/264 | loss=0.33601 | 0.023s/iter | ETA: 1.3min
[Epoch 08] Iter  200/264 | loss=0.27749 | 0.008s/iter | ETA: 0.4min

------------------------------------------------------------
[Epoch 08] Summary | Time: 2.2s
Train Loss: 0.336821
Vali  Loss: 0.688085
Test  Loss: 0.387023
EarlyStopping counter: 4 out of 5
------------------------------------------------------------
Updating learning rate to 7.8125e-06
[Epoch 09] Iter  100/264 | loss=0.31846 | 0.025s/iter | ETA: 1.3min
[Epoch 09] Iter  200/264 | loss=0.31587 | 0.012s/iter | ETA: 0.6min

------------------------------------------------------------
[Epoch 09] Summary | Time: 3.2s
Train Loss: 0.336487
Vali  Loss: 0.688896
Test  Loss: 0.386569
EarlyStopping counter: 5 out of 5
------------------------------------------------------------

************************************************************
EARLY STOPPING TRIGGERED
************************************************************
>>>>>>>testing : ETTh1_96_96_FreDEA_ETTh1_ftM_sl96_ll48_pl96_dm64_nh8_el1_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 2785
mse:0.3913840055465698, mae:0.4079279899597168, rmse:0.625606894493103
✅ 实验完成: MSE=0.391384, MAE=0.407928

================================================================================
运行实验: ETTh1 - Pred 192
================================================================================
命令: python -u run_longExp.py --data ETTh1 --data_path ETTh1.csv --model FreDEA --model_id ETTh1_96_192 --seq_len 96 --pred_len 192 --enc_in 7 --dec_in 7 --c_out 7 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --dropout 0.1 --batch_size 32 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4

Args in experiment:
Namespace(is_training=1, train_only=False, model_id='ETTh1_96_192', model='FreDEA', data='ETTh1', root_path='./dataset/', data_path='ETTh1.csv', channel_independence=0, features='M', target='OT', freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=192, individual=False, embed_type=0, enc_in=7, dec_in=7, c_out=7, d_model=64, reg_lambda=0.01, wavelet='haar', n_bins=8, n_heads=8, e_layers=1, d_layers=1, d_ff=128, moving_avg=25, factor=1, distil=True, rev_affine=1, memory_size=64, bottleneck_dim=-1, ablation_freq=0, ablation_tea=0, ablation_cea=0, dropout=0.1, embed='timeF', activation='gelu', output_attention=False, do_predict=False, num_workers=4, itr=1, train_epochs=20, batch_size=32, patience=5, learning_rate=0.001, des='Exp', loss='mse', lradj='type1', use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=False, devices='0,1,2,3', test_flop=False)
Use GPU: cuda:0
>>>>>>>start training : ETTh1_96_192_FreDEA_ETTh1_ftM_sl96_ll48_pl192_dm64_nh8_el1_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0>>>>>>>>>>>>>>>>>>>>>>>>>>
train 8353
val 2689
test 2689

============================================================
TRAINING CONFIGURATION
============================================================
Model:           FreDEA
Dataset:         ETTh1
Seq/Pred Len:    96 -> 192
Batch Size:      32
Learning Rate:   0.001
Train Epochs:    20
Total Params:    58,196
d_model:         64
n_heads:         8
e_layers:        1
============================================================

[Epoch 01] Iter  100/261 | loss=0.45770 | 0.019s/iter | ETA: 1.6min
[Epoch 01] Iter  200/261 | loss=0.46062 | 0.008s/iter | ETA: 0.7min

------------------------------------------------------------
[Epoch 01] Summary | Time: 2.7s
Train Loss: 0.509875
Vali  Loss: 1.036579
Test  Loss: 0.472156
Validation loss decreased (inf --> 1.036579).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.001
[Epoch 02] Iter  100/261 | loss=0.44956 | 0.025s/iter | ETA: 2.0min
[Epoch 02] Iter  200/261 | loss=0.42285 | 0.011s/iter | ETA: 0.8min

------------------------------------------------------------
[Epoch 02] Summary | Time: 2.8s
Train Loss: 0.444220
Vali  Loss: 1.014098
Test  Loss: 0.447720
Validation loss decreased (1.036579 --> 1.014098).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 03] Iter  100/261 | loss=0.42833 | 0.026s/iter | ETA: 2.0min
[Epoch 03] Iter  200/261 | loss=0.41264 | 0.009s/iter | ETA: 0.7min

------------------------------------------------------------
[Epoch 03] Summary | Time: 2.5s
Train Loss: 0.431829
Vali  Loss: 1.001107
Test  Loss: 0.442216
Validation loss decreased (1.014098 --> 1.001107).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.00025
[Epoch 04] Iter  100/261 | loss=0.43845 | 0.028s/iter | ETA: 2.0min
[Epoch 04] Iter  200/261 | loss=0.40449 | 0.013s/iter | ETA: 0.9min

------------------------------------------------------------
[Epoch 04] Summary | Time: 3.4s
Train Loss: 0.425539
Vali  Loss: 1.005666
Test  Loss: 0.437710
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 0.000125
[Epoch 05] Iter  100/261 | loss=0.42019 | 0.028s/iter | ETA: 1.9min
[Epoch 05] Iter  200/261 | loss=0.40697 | 0.010s/iter | ETA: 0.7min

------------------------------------------------------------
[Epoch 05] Summary | Time: 2.7s
Train Loss: 0.422414
Vali  Loss: 1.002931
Test  Loss: 0.437531
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 6.25e-05
[Epoch 06] Iter  100/261 | loss=0.36982 | 0.026s/iter | ETA: 1.6min
[Epoch 06] Iter  200/261 | loss=0.47999 | 0.012s/iter | ETA: 0.7min

------------------------------------------------------------
[Epoch 06] Summary | Time: 3.0s
Train Loss: 0.420472
Vali  Loss: 1.004790
Test  Loss: 0.437108
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 3.125e-05
[Epoch 07] Iter  100/261 | loss=0.39325 | 0.027s/iter | ETA: 1.6min
[Epoch 07] Iter  200/261 | loss=0.43040 | 0.009s/iter | ETA: 0.5min

------------------------------------------------------------
[Epoch 07] Summary | Time: 2.4s
Train Loss: 0.420009
Vali  Loss: 1.004812
Test  Loss: 0.436502
EarlyStopping counter: 4 out of 5
------------------------------------------------------------
Updating learning rate to 1.5625e-05
[Epoch 08] Iter  100/261 | loss=0.42207 | 0.026s/iter | ETA: 1.4min
[Epoch 08] Iter  200/261 | loss=0.44756 | 0.011s/iter | ETA: 0.6min

------------------------------------------------------------
[Epoch 08] Summary | Time: 3.1s
Train Loss: 0.419899
Vali  Loss: 1.003171
Test  Loss: 0.436783
EarlyStopping counter: 5 out of 5
------------------------------------------------------------

************************************************************
EARLY STOPPING TRIGGERED
************************************************************
>>>>>>>testing : ETTh1_96_192_FreDEA_ETTh1_ftM_sl96_ll48_pl192_dm64_nh8_el1_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 2689
mse:0.44172507524490356, mae:0.43985700607299805, rmse:0.664624035358429
✅ 实验完成: MSE=0.441725, MAE=0.439857

================================================================================
运行实验: ETTh1 - Pred 336
================================================================================
命令: python -u run_longExp.py --data ETTh1 --data_path ETTh1.csv --model FreDEA --model_id ETTh1_96_336 --seq_len 96 --pred_len 336 --enc_in 7 --dec_in 7 --c_out 7 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --dropout 0.2 --batch_size 32 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4

Args in experiment:
Namespace(is_training=1, train_only=False, model_id='ETTh1_96_336', model='FreDEA', data='ETTh1', root_path='./dataset/', data_path='ETTh1.csv', channel_independence=0, features='M', target='OT', freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=336, individual=False, embed_type=0, enc_in=7, dec_in=7, c_out=7, d_model=64, reg_lambda=0.01, wavelet='haar', n_bins=8, n_heads=8, e_layers=1, d_layers=1, d_ff=128, moving_avg=25, factor=1, distil=True, rev_affine=1, memory_size=64, bottleneck_dim=-1, ablation_freq=0, ablation_tea=0, ablation_cea=0, dropout=0.2, embed='timeF', activation='gelu', output_attention=False, do_predict=False, num_workers=4, itr=1, train_epochs=20, batch_size=32, patience=5, learning_rate=0.001, des='Exp', loss='mse', lradj='type1', use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=False, devices='0,1,2,3', test_flop=False)
Use GPU: cuda:0
>>>>>>>start training : ETTh1_96_336_FreDEA_ETTh1_ftM_sl96_ll48_pl336_dm64_nh8_el1_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0>>>>>>>>>>>>>>>>>>>>>>>>>>
train 8209
val 2545
test 2545

============================================================
TRAINING CONFIGURATION
============================================================
Model:           FreDEA
Dataset:         ETTh1
Seq/Pred Len:    96 -> 336
Batch Size:      32
Learning Rate:   0.001
Train Epochs:    20
Total Params:    67,556
d_model:         64
n_heads:         8
e_layers:        1
============================================================

[Epoch 01] Iter  100/256 | loss=0.51815 | 0.018s/iter | ETA: 1.5min
[Epoch 01] Iter  200/256 | loss=0.44436 | 0.007s/iter | ETA: 0.5min

------------------------------------------------------------
[Epoch 01] Summary | Time: 2.3s
Train Loss: 0.552261
Vali  Loss: 1.307247
Test  Loss: 0.491684
Validation loss decreased (inf --> 1.307247).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.001
[Epoch 02] Iter  100/256 | loss=0.49887 | 0.022s/iter | ETA: 1.7min
[Epoch 02] Iter  200/256 | loss=0.45286 | 0.008s/iter | ETA: 0.6min

------------------------------------------------------------
[Epoch 02] Summary | Time: 2.1s
Train Loss: 0.491104
Vali  Loss: 1.289066
Test  Loss: 0.475868
Validation loss decreased (1.307247 --> 1.289066).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 03] Iter  100/256 | loss=0.40926 | 0.023s/iter | ETA: 1.7min
[Epoch 03] Iter  200/256 | loss=0.53744 | 0.009s/iter | ETA: 0.6min

------------------------------------------------------------
[Epoch 03] Summary | Time: 2.3s
Train Loss: 0.477250
Vali  Loss: 1.300400
Test  Loss: 0.467662
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 0.00025
[Epoch 04] Iter  100/256 | loss=0.55700 | 0.024s/iter | ETA: 1.7min
[Epoch 04] Iter  200/256 | loss=0.46587 | 0.009s/iter | ETA: 0.6min

------------------------------------------------------------
[Epoch 04] Summary | Time: 2.4s
Train Loss: 0.470788
Vali  Loss: 1.302142
Test  Loss: 0.462556
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 0.000125
[Epoch 05] Iter  100/256 | loss=0.45013 | 0.021s/iter | ETA: 1.4min
[Epoch 05] Iter  200/256 | loss=0.46510 | 0.005s/iter | ETA: 0.3min

------------------------------------------------------------
[Epoch 05] Summary | Time: 1.5s
Train Loss: 0.467298
Vali  Loss: 1.308243
Test  Loss: 0.462961
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 6.25e-05
[Epoch 06] Iter  100/256 | loss=0.44452 | 0.018s/iter | ETA: 1.1min
[Epoch 06] Iter  200/256 | loss=0.45351 | 0.005s/iter | ETA: 0.3min

------------------------------------------------------------
[Epoch 06] Summary | Time: 1.5s
Train Loss: 0.465185
Vali  Loss: 1.309043
Test  Loss: 0.461592
EarlyStopping counter: 4 out of 5
------------------------------------------------------------
Updating learning rate to 3.125e-05
[Epoch 07] Iter  100/256 | loss=0.45654 | 0.025s/iter | ETA: 1.4min
[Epoch 07] Iter  200/256 | loss=0.51137 | 0.012s/iter | ETA: 0.7min

------------------------------------------------------------
[Epoch 07] Summary | Time: 3.3s
Train Loss: 0.464806
Vali  Loss: 1.307465
Test  Loss: 0.464372
EarlyStopping counter: 5 out of 5
------------------------------------------------------------

************************************************************
EARLY STOPPING TRIGGERED
************************************************************
>>>>>>>testing : ETTh1_96_336_FreDEA_ETTh1_ftM_sl96_ll48_pl336_dm64_nh8_el1_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 2545
mse:0.4747979938983917, mae:0.4517346918582916, rmse:0.689055860042572
✅ 实验完成: MSE=0.474798, MAE=0.451735

================================================================================
运行实验: ETTh1 - Pred 720
================================================================================
命令: python -u run_longExp.py --data ETTh1 --data_path ETTh1.csv --model FreDEA --model_id ETTh1_96_720 --seq_len 96 --pred_len 720 --enc_in 7 --dec_in 7 --c_out 7 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim 1 --dropout 0.4 --batch_size 32 --learning_rate 0.0003 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4

Args in experiment:
Namespace(is_training=1, train_only=False, model_id='ETTh1_96_720', model='FreDEA', data='ETTh1', root_path='./dataset/', data_path='ETTh1.csv', channel_independence=0, features='M', target='OT', freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=720, individual=False, embed_type=0, enc_in=7, dec_in=7, c_out=7, d_model=64, reg_lambda=0.01, wavelet='haar', n_bins=8, n_heads=8, e_layers=1, d_layers=1, d_ff=128, moving_avg=25, factor=1, distil=True, rev_affine=1, memory_size=64, bottleneck_dim=1, ablation_freq=0, ablation_tea=0, ablation_cea=0, dropout=0.4, embed='timeF', activation='gelu', output_attention=False, do_predict=False, num_workers=4, itr=1, train_epochs=20, batch_size=32, patience=5, learning_rate=0.0003, des='Exp', loss='mse', lradj='type1', use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=False, devices='0,1,2,3', test_flop=False)
Use GPU: cuda:0
>>>>>>>start training : ETTh1_96_720_FreDEA_ETTh1_ftM_sl96_ll48_pl720_dm64_nh8_el1_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0>>>>>>>>>>>>>>>>>>>>>>>>>>
train 7825
val 2161
test 2161

============================================================
TRAINING CONFIGURATION
============================================================
Model:           FreDEA
Dataset:         ETTh1
Seq/Pred Len:    96 -> 720
Batch Size:      32
Learning Rate:   0.0003
Train Epochs:    20
Total Params:    92,516
d_model:         64
n_heads:         8
e_layers:        1
============================================================

[Epoch 01] Iter  100/244 | loss=0.75360 | 0.021s/iter | ETA: 1.7min
[Epoch 01] Iter  200/244 | loss=0.79074 | 0.010s/iter | ETA: 0.8min

------------------------------------------------------------
[Epoch 01] Summary | Time: 3.1s
Train Loss: 0.761776
Vali  Loss: 1.924981
Test  Loss: 0.720577
Validation loss decreased (inf --> 1.924981).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0003
[Epoch 02] Iter  100/244 | loss=0.71026 | 0.024s/iter | ETA: 1.8min
[Epoch 02] Iter  200/244 | loss=0.69589 | 0.010s/iter | ETA: 0.7min

------------------------------------------------------------
[Epoch 02] Summary | Time: 2.5s
Train Loss: 0.709002
Vali  Loss: 1.657488
Test  Loss: 0.551527
Validation loss decreased (1.924981 --> 1.657488).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.00015
[Epoch 03] Iter  100/244 | loss=0.68348 | 0.027s/iter | ETA: 1.9min
[Epoch 03] Iter  200/244 | loss=0.71144 | 0.012s/iter | ETA: 0.8min

------------------------------------------------------------
[Epoch 03] Summary | Time: 3.1s
Train Loss: 0.667430
Vali  Loss: 1.633025
Test  Loss: 0.529872
Validation loss decreased (1.657488 --> 1.633025).  Saving model ...
------------------------------------------------------------
Updating learning rate to 7.5e-05
[Epoch 04] Iter  100/244 | loss=0.53135 | 0.028s/iter | ETA: 1.9min
[Epoch 04] Iter  200/244 | loss=0.55961 | 0.011s/iter | ETA: 0.7min

------------------------------------------------------------
[Epoch 04] Summary | Time: 2.9s
Train Loss: 0.659681
Vali  Loss: 1.619020
Test  Loss: 0.522111
Validation loss decreased (1.633025 --> 1.619020).  Saving model ...
------------------------------------------------------------
Updating learning rate to 3.75e-05
[Epoch 05] Iter  100/244 | loss=0.67254 | 0.024s/iter | ETA: 1.5min
[Epoch 05] Iter  200/244 | loss=0.65824 | 0.009s/iter | ETA: 0.6min

------------------------------------------------------------
[Epoch 05] Summary | Time: 2.4s
Train Loss: 0.656456
Vali  Loss: 1.619601
Test  Loss: 0.519647
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 1.875e-05
[Epoch 06] Iter  100/244 | loss=0.63617 | 0.024s/iter | ETA: 1.4min
[Epoch 06] Iter  200/244 | loss=0.64755 | 0.012s/iter | ETA: 0.7min

------------------------------------------------------------
[Epoch 06] Summary | Time: 2.9s
Train Loss: 0.654993
Vali  Loss: 1.614242
Test  Loss: 0.518540
Validation loss decreased (1.619020 --> 1.614242).  Saving model ...
------------------------------------------------------------
Updating learning rate to 9.375e-06
[Epoch 07] Iter  100/244 | loss=0.75703 | 0.026s/iter | ETA: 1.4min
[Epoch 07] Iter  200/244 | loss=0.68285 | 0.009s/iter | ETA: 0.5min

------------------------------------------------------------
[Epoch 07] Summary | Time: 2.4s
Train Loss: 0.654581
Vali  Loss: 1.614005
Test  Loss: 0.517751
Validation loss decreased (1.614242 --> 1.614005).  Saving model ...
------------------------------------------------------------
Updating learning rate to 4.6875e-06
[Epoch 08] Iter  100/244 | loss=0.77439 | 0.025s/iter | ETA: 1.3min
[Epoch 08] Iter  200/244 | loss=0.75024 | 0.011s/iter | ETA: 0.6min

------------------------------------------------------------
[Epoch 08] Summary | Time: 3.0s
Train Loss: 0.654135
Vali  Loss: 1.614994
Test  Loss: 0.517437
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 2.34375e-06
[Epoch 09] Iter  100/244 | loss=0.66834 | 0.026s/iter | ETA: 1.2min
[Epoch 09] Iter  200/244 | loss=0.64224 | 0.009s/iter | ETA: 0.4min

------------------------------------------------------------
[Epoch 09] Summary | Time: 2.3s
Train Loss: 0.653351
Vali  Loss: 1.614459
Test  Loss: 0.517244
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 1.171875e-06
[Epoch 10] Iter  100/244 | loss=0.61341 | 0.025s/iter | ETA: 1.1min
[Epoch 10] Iter  200/244 | loss=0.60257 | 0.012s/iter | ETA: 0.5min

------------------------------------------------------------
[Epoch 10] Summary | Time: 3.0s
Train Loss: 0.653694
Vali  Loss: 1.615261
Test  Loss: 0.517147
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 5.859375e-07
[Epoch 11] Iter  100/244 | loss=0.65333 | 0.028s/iter | ETA: 1.1min
[Epoch 11] Iter  200/244 | loss=0.54509 | 0.012s/iter | ETA: 0.4min

------------------------------------------------------------
[Epoch 11] Summary | Time: 3.0s
Train Loss: 0.654625
Vali  Loss: 1.615970
Test  Loss: 0.517113
EarlyStopping counter: 4 out of 5
------------------------------------------------------------
Updating learning rate to 2.9296875e-07
[Epoch 12] Iter  100/244 | loss=0.71260 | 0.027s/iter | ETA: 0.9min
[Epoch 12] Iter  200/244 | loss=0.69016 | 0.012s/iter | ETA: 0.4min

------------------------------------------------------------
[Epoch 12] Summary | Time: 3.1s
Train Loss: 0.653700
Vali  Loss: 1.615123
Test  Loss: 0.517086
EarlyStopping counter: 5 out of 5
------------------------------------------------------------

************************************************************
EARLY STOPPING TRIGGERED
************************************************************
>>>>>>>testing : ETTh1_96_720_FreDEA_ETTh1_ftM_sl96_ll48_pl720_dm64_nh8_el1_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 2161
mse:0.5165561437606812, mae:0.5000734329223633, rmse:0.718718409538269
✅ 实验完成: MSE=0.516556, MAE=0.500073

================================================================================
运行实验: ETTh2 - Pred 96
================================================================================
命令: python -u run_longExp.py --data ETTh2 --data_path ETTh2.csv --model FreDEA --model_id ETTh2_96_96 --seq_len 96 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --dropout 0.0 --batch_size 32 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4

Args in experiment:
Namespace(is_training=1, train_only=False, model_id='ETTh2_96_96', model='FreDEA', data='ETTh2', root_path='./dataset/', data_path='ETTh2.csv', channel_independence=0, features='M', target='OT', freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=96, individual=False, embed_type=0, enc_in=7, dec_in=7, c_out=7, d_model=64, reg_lambda=0.01, wavelet='haar', n_bins=8, n_heads=8, e_layers=1, d_layers=1, d_ff=128, moving_avg=25, factor=1, distil=True, rev_affine=1, memory_size=64, bottleneck_dim=-1, ablation_freq=0, ablation_tea=0, ablation_cea=0, dropout=0.0, embed='timeF', activation='gelu', output_attention=False, do_predict=False, num_workers=4, itr=1, train_epochs=20, batch_size=32, patience=5, learning_rate=0.001, des='Exp', loss='mse', lradj='type1', use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=False, devices='0,1,2,3', test_flop=False)
Use GPU: cuda:0
>>>>>>>start training : ETTh2_96_96_FreDEA_ETTh2_ftM_sl96_ll48_pl96_dm64_nh8_el1_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0>>>>>>>>>>>>>>>>>>>>>>>>>>
train 8449
val 2785
test 2785

============================================================
TRAINING CONFIGURATION
============================================================
Model:           FreDEA
Dataset:         ETTh2
Seq/Pred Len:    96 -> 96
Batch Size:      32
Learning Rate:   0.001
Train Epochs:    20
Total Params:    51,956
d_model:         64
n_heads:         8
e_layers:        1
============================================================

[Epoch 01] Iter  100/264 | loss=0.62140 | 0.019s/iter | ETA: 1.7min
[Epoch 01] Iter  200/264 | loss=0.76244 | 0.009s/iter | ETA: 0.7min

------------------------------------------------------------
[Epoch 01] Summary | Time: 2.8s
Train Loss: 0.462439
Vali  Loss: 0.233416
Test  Loss: 0.322074
Validation loss decreased (inf --> 0.233416).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.001
[Epoch 02] Iter  100/264 | loss=0.32465 | 0.023s/iter | ETA: 1.9min
[Epoch 02] Iter  200/264 | loss=0.26578 | 0.009s/iter | ETA: 0.7min

------------------------------------------------------------
[Epoch 02] Summary | Time: 2.5s
Train Loss: 0.405518
Vali  Loss: 0.230235
Test  Loss: 0.302085
Validation loss decreased (0.233416 --> 0.230235).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 03] Iter  100/264 | loss=0.48687 | 0.027s/iter | ETA: 2.1min
[Epoch 03] Iter  200/264 | loss=0.34720 | 0.013s/iter | ETA: 1.0min

------------------------------------------------------------
[Epoch 03] Summary | Time: 3.6s
Train Loss: 0.351950
Vali  Loss: 0.241482
Test  Loss: 0.300739
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 0.00025
[Epoch 04] Iter  100/264 | loss=0.33168 | 0.024s/iter | ETA: 1.8min
[Epoch 04] Iter  200/264 | loss=0.36124 | 0.007s/iter | ETA: 0.5min

------------------------------------------------------------
[Epoch 04] Summary | Time: 2.0s
Train Loss: 0.328575
Vali  Loss: 0.239526
Test  Loss: 0.299419
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 0.000125
[Epoch 05] Iter  100/264 | loss=0.29863 | 0.021s/iter | ETA: 1.5min
[Epoch 05] Iter  200/264 | loss=0.23696 | 0.005s/iter | ETA: 0.3min

------------------------------------------------------------
[Epoch 05] Summary | Time: 1.4s
Train Loss: 0.318208
Vali  Loss: 0.243791
Test  Loss: 0.302627
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 6.25e-05
[Epoch 06] Iter  100/264 | loss=0.27415 | 0.021s/iter | ETA: 1.4min
[Epoch 06] Iter  200/264 | loss=0.47421 | 0.008s/iter | ETA: 0.5min

------------------------------------------------------------
[Epoch 06] Summary | Time: 2.3s
Train Loss: 0.313774
Vali  Loss: 0.238499
Test  Loss: 0.301536
EarlyStopping counter: 4 out of 5
------------------------------------------------------------
Updating learning rate to 3.125e-05
[Epoch 07] Iter  100/264 | loss=0.22692 | 0.026s/iter | ETA: 1.5min
[Epoch 07] Iter  200/264 | loss=0.23001 | 0.011s/iter | ETA: 0.6min

------------------------------------------------------------
[Epoch 07] Summary | Time: 3.0s
Train Loss: 0.311608
Vali  Loss: 0.242881
Test  Loss: 0.301432
EarlyStopping counter: 5 out of 5
------------------------------------------------------------

************************************************************
EARLY STOPPING TRIGGERED
************************************************************
>>>>>>>testing : ETTh2_96_96_FreDEA_ETTh2_ftM_sl96_ll48_pl96_dm64_nh8_el1_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 2785
mse:0.3045145869255066, mae:0.35437247157096863, rmse:0.5518283843994141
✅ 实验完成: MSE=0.304515, MAE=0.354372

================================================================================
运行实验: ETTh2 - Pred 192
================================================================================
命令: python -u run_longExp.py --data ETTh2 --data_path ETTh2.csv --model FreDEA --model_id ETTh2_96_192 --seq_len 96 --pred_len 192 --enc_in 7 --dec_in 7 --c_out 7 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --dropout 0.0 --batch_size 32 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4

Args in experiment:
Namespace(is_training=1, train_only=False, model_id='ETTh2_96_192', model='FreDEA', data='ETTh2', root_path='./dataset/', data_path='ETTh2.csv', channel_independence=0, features='M', target='OT', freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=192, individual=False, embed_type=0, enc_in=7, dec_in=7, c_out=7, d_model=64, reg_lambda=0.01, wavelet='haar', n_bins=8, n_heads=8, e_layers=1, d_layers=1, d_ff=128, moving_avg=25, factor=1, distil=True, rev_affine=1, memory_size=64, bottleneck_dim=-1, ablation_freq=0, ablation_tea=0, ablation_cea=0, dropout=0.0, embed='timeF', activation='gelu', output_attention=False, do_predict=False, num_workers=4, itr=1, train_epochs=20, batch_size=32, patience=5, learning_rate=0.001, des='Exp', loss='mse', lradj='type1', use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=False, devices='0,1,2,3', test_flop=False)
Use GPU: cuda:0
>>>>>>>start training : ETTh2_96_192_FreDEA_ETTh2_ftM_sl96_ll48_pl192_dm64_nh8_el1_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0>>>>>>>>>>>>>>>>>>>>>>>>>>
train 8353
val 2689
test 2689

============================================================
TRAINING CONFIGURATION
============================================================
Model:           FreDEA
Dataset:         ETTh2
Seq/Pred Len:    96 -> 192
Batch Size:      32
Learning Rate:   0.001
Train Epochs:    20
Total Params:    58,196
d_model:         64
n_heads:         8
e_layers:        1
============================================================

[Epoch 01] Iter  100/261 | loss=0.45440 | 0.019s/iter | ETA: 1.7min
[Epoch 01] Iter  200/261 | loss=0.59289 | 0.006s/iter | ETA: 0.5min

------------------------------------------------------------
[Epoch 01] Summary | Time: 2.4s
Train Loss: 0.574329
Vali  Loss: 0.308612
Test  Loss: 0.409553
Validation loss decreased (inf --> 0.308612).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.001
[Epoch 02] Iter  100/261 | loss=0.33286 | 0.023s/iter | ETA: 1.9min
[Epoch 02] Iter  200/261 | loss=0.55293 | 0.008s/iter | ETA: 0.7min

------------------------------------------------------------
[Epoch 02] Summary | Time: 2.3s
Train Loss: 0.536169
Vali  Loss: 0.305640
Test  Loss: 0.407652
Validation loss decreased (0.308612 --> 0.305640).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 03] Iter  100/261 | loss=0.38476 | 0.023s/iter | ETA: 1.8min
[Epoch 03] Iter  200/261 | loss=0.36704 | 0.008s/iter | ETA: 0.6min

------------------------------------------------------------
[Epoch 03] Summary | Time: 2.2s
Train Loss: 0.486138
Vali  Loss: 0.302125
Test  Loss: 0.400735
Validation loss decreased (0.305640 --> 0.302125).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.00025
[Epoch 04] Iter  100/261 | loss=0.59644 | 0.023s/iter | ETA: 1.7min
[Epoch 04] Iter  200/261 | loss=0.53067 | 0.010s/iter | ETA: 0.7min

------------------------------------------------------------
[Epoch 04] Summary | Time: 2.8s
Train Loss: 0.451623
Vali  Loss: 0.308098
Test  Loss: 0.401488
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 0.000125
[Epoch 05] Iter  100/261 | loss=0.48402 | 0.027s/iter | ETA: 1.8min
[Epoch 05] Iter  200/261 | loss=0.39779 | 0.013s/iter | ETA: 0.9min

------------------------------------------------------------
[Epoch 05] Summary | Time: 3.2s
Train Loss: 0.435232
Vali  Loss: 0.303378
Test  Loss: 0.399190
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 6.25e-05
[Epoch 06] Iter  100/261 | loss=0.79907 | 0.029s/iter | ETA: 1.9min
[Epoch 06] Iter  200/261 | loss=0.27327 | 0.011s/iter | ETA: 0.7min

------------------------------------------------------------
[Epoch 06] Summary | Time: 3.1s
Train Loss: 0.428697
Vali  Loss: 0.306491
Test  Loss: 0.400777
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 3.125e-05
[Epoch 07] Iter  100/261 | loss=0.28305 | 0.026s/iter | ETA: 1.5min
[Epoch 07] Iter  200/261 | loss=0.62739 | 0.008s/iter | ETA: 0.5min

------------------------------------------------------------
[Epoch 07] Summary | Time: 2.3s
Train Loss: 0.425310
Vali  Loss: 0.308298
Test  Loss: 0.402284
EarlyStopping counter: 4 out of 5
------------------------------------------------------------
Updating learning rate to 1.5625e-05
[Epoch 08] Iter  100/261 | loss=0.23641 | 0.023s/iter | ETA: 1.3min
[Epoch 08] Iter  200/261 | loss=0.28154 | 0.008s/iter | ETA: 0.4min

------------------------------------------------------------
[Epoch 08] Summary | Time: 2.3s
Train Loss: 0.423569
Vali  Loss: 0.308445
Test  Loss: 0.402270
EarlyStopping counter: 5 out of 5
------------------------------------------------------------

************************************************************
EARLY STOPPING TRIGGERED
************************************************************
>>>>>>>testing : ETTh2_96_192_FreDEA_ETTh2_ftM_sl96_ll48_pl192_dm64_nh8_el1_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 2689
mse:0.3940102159976959, mae:0.4103425145149231, rmse:0.6277023553848267
✅ 实验完成: MSE=0.394010, MAE=0.410343

================================================================================
运行实验: ETTh2 - Pred 336
================================================================================
命令: python -u run_longExp.py --data ETTh2 --data_path ETTh2.csv --model FreDEA --model_id ETTh2_96_336 --seq_len 96 --pred_len 336 --enc_in 7 --dec_in 7 --c_out 7 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim -1 --dropout 0.2 --batch_size 32 --learning_rate 0.001 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4

Args in experiment:
Namespace(is_training=1, train_only=False, model_id='ETTh2_96_336', model='FreDEA', data='ETTh2', root_path='./dataset/', data_path='ETTh2.csv', channel_independence=0, features='M', target='OT', freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=336, individual=False, embed_type=0, enc_in=7, dec_in=7, c_out=7, d_model=64, reg_lambda=0.01, wavelet='haar', n_bins=8, n_heads=8, e_layers=1, d_layers=1, d_ff=128, moving_avg=25, factor=1, distil=True, rev_affine=1, memory_size=64, bottleneck_dim=-1, ablation_freq=0, ablation_tea=0, ablation_cea=0, dropout=0.2, embed='timeF', activation='gelu', output_attention=False, do_predict=False, num_workers=4, itr=1, train_epochs=20, batch_size=32, patience=5, learning_rate=0.001, des='Exp', loss='mse', lradj='type1', use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=False, devices='0,1,2,3', test_flop=False)
Use GPU: cuda:0
>>>>>>>start training : ETTh2_96_336_FreDEA_ETTh2_ftM_sl96_ll48_pl336_dm64_nh8_el1_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0>>>>>>>>>>>>>>>>>>>>>>>>>>
train 8209
val 2545
test 2545

============================================================
TRAINING CONFIGURATION
============================================================
Model:           FreDEA
Dataset:         ETTh2
Seq/Pred Len:    96 -> 336
Batch Size:      32
Learning Rate:   0.001
Train Epochs:    20
Total Params:    67,556
d_model:         64
n_heads:         8
e_layers:        1
============================================================

[Epoch 01] Iter  100/256 | loss=0.62643 | 0.018s/iter | ETA: 1.5min
[Epoch 01] Iter  200/256 | loss=0.56833 | 0.008s/iter | ETA: 0.7min

------------------------------------------------------------
[Epoch 01] Summary | Time: 2.6s
Train Loss: 0.684539
Vali  Loss: 0.381934
Test  Loss: 0.441884
Validation loss decreased (inf --> 0.381934).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.001
[Epoch 02] Iter  100/256 | loss=0.46434 | 0.025s/iter | ETA: 2.0min
[Epoch 02] Iter  200/256 | loss=0.73285 | 0.014s/iter | ETA: 1.1min

------------------------------------------------------------
[Epoch 02] Summary | Time: 3.2s
Train Loss: 0.632857
Vali  Loss: 0.378984
Test  Loss: 0.426220
Validation loss decreased (0.381934 --> 0.378984).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 03] Iter  100/256 | loss=0.50950 | 0.025s/iter | ETA: 1.9min
[Epoch 03] Iter  200/256 | loss=0.44824 | 0.005s/iter | ETA: 0.4min

------------------------------------------------------------
[Epoch 03] Summary | Time: 1.6s
Train Loss: 0.595648
Vali  Loss: 0.379877
Test  Loss: 0.432007
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 0.00025
[Epoch 04] Iter  100/256 | loss=0.28946 | 0.021s/iter | ETA: 1.5min
[Epoch 04] Iter  200/256 | loss=0.54003 | 0.009s/iter | ETA: 0.6min

------------------------------------------------------------
[Epoch 04] Summary | Time: 2.4s
Train Loss: 0.577692
Vali  Loss: 0.380503
Test  Loss: 0.438276
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 0.000125
[Epoch 05] Iter  100/256 | loss=0.42628 | 0.025s/iter | ETA: 1.7min
[Epoch 05] Iter  200/256 | loss=0.62389 | 0.009s/iter | ETA: 0.6min

------------------------------------------------------------
[Epoch 05] Summary | Time: 2.4s
Train Loss: 0.567301
Vali  Loss: 0.381034
Test  Loss: 0.435430
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 6.25e-05
[Epoch 06] Iter  100/256 | loss=0.60265 | 0.028s/iter | ETA: 1.8min
[Epoch 06] Iter  200/256 | loss=0.71258 | 0.013s/iter | ETA: 0.8min

------------------------------------------------------------
[Epoch 06] Summary | Time: 3.4s
Train Loss: 0.563820
Vali  Loss: 0.381888
Test  Loss: 0.436062
EarlyStopping counter: 4 out of 5
------------------------------------------------------------
Updating learning rate to 3.125e-05
[Epoch 07] Iter  100/256 | loss=0.62661 | 0.027s/iter | ETA: 1.6min
[Epoch 07] Iter  200/256 | loss=0.72576 | 0.009s/iter | ETA: 0.5min

------------------------------------------------------------
[Epoch 07] Summary | Time: 2.5s
Train Loss: 0.561893
Vali  Loss: 0.382528
Test  Loss: 0.437660
EarlyStopping counter: 5 out of 5
------------------------------------------------------------

************************************************************
EARLY STOPPING TRIGGERED
************************************************************
>>>>>>>testing : ETTh2_96_336_FreDEA_ETTh2_ftM_sl96_ll48_pl336_dm64_nh8_el1_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 2545
mse:0.42229920625686646, mae:0.4322494566440582, rmse:0.649845540523529
✅ 实验完成: MSE=0.422299, MAE=0.432249

================================================================================
运行实验: ETTh2 - Pred 720
================================================================================
命令: python -u run_longExp.py --data ETTh2 --data_path ETTh2.csv --model FreDEA --model_id ETTh2_96_720 --seq_len 96 --pred_len 720 --enc_in 7 --dec_in 7 --c_out 7 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim 1 --dropout 0.45 --batch_size 32 --learning_rate 0.0003 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4

Args in experiment:
Namespace(is_training=1, train_only=False, model_id='ETTh2_96_720', model='FreDEA', data='ETTh2', root_path='./dataset/', data_path='ETTh2.csv', channel_independence=0, features='M', target='OT', freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=720, individual=False, embed_type=0, enc_in=7, dec_in=7, c_out=7, d_model=64, reg_lambda=0.01, wavelet='haar', n_bins=8, n_heads=8, e_layers=1, d_layers=1, d_ff=128, moving_avg=25, factor=1, distil=True, rev_affine=1, memory_size=64, bottleneck_dim=1, ablation_freq=0, ablation_tea=0, ablation_cea=0, dropout=0.45, embed='timeF', activation='gelu', output_attention=False, do_predict=False, num_workers=4, itr=1, train_epochs=20, batch_size=32, patience=5, learning_rate=0.0003, des='Exp', loss='mse', lradj='type1', use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=False, devices='0,1,2,3', test_flop=False)
Use GPU: cuda:0
>>>>>>>start training : ETTh2_96_720_FreDEA_ETTh2_ftM_sl96_ll48_pl720_dm64_nh8_el1_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0>>>>>>>>>>>>>>>>>>>>>>>>>>
train 7825
val 2161
test 2161

============================================================
TRAINING CONFIGURATION
============================================================
Model:           FreDEA
Dataset:         ETTh2
Seq/Pred Len:    96 -> 720
Batch Size:      32
Learning Rate:   0.0003
Train Epochs:    20
Total Params:    92,516
d_model:         64
n_heads:         8
e_layers:        1
============================================================

[Epoch 01] Iter  100/244 | loss=0.81152 | 0.018s/iter | ETA: 1.4min
[Epoch 01] Iter  200/244 | loss=0.90769 | 0.009s/iter | ETA: 0.7min

------------------------------------------------------------
[Epoch 01] Summary | Time: 2.5s
Train Loss: 0.904062
Vali  Loss: 0.661844
Test  Loss: 0.454228
Validation loss decreased (inf --> 0.661844).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0003
[Epoch 02] Iter  100/244 | loss=0.67334 | 0.024s/iter | ETA: 1.8min
[Epoch 02] Iter  200/244 | loss=0.81541 | 0.009s/iter | ETA: 0.7min

------------------------------------------------------------
[Epoch 02] Summary | Time: 2.4s
Train Loss: 0.890150
Vali  Loss: 0.660599
Test  Loss: 0.456657
Validation loss decreased (0.661844 --> 0.660599).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.00015
[Epoch 03] Iter  100/244 | loss=0.74729 | 0.028s/iter | ETA: 2.0min
[Epoch 03] Iter  200/244 | loss=1.35021 | 0.013s/iter | ETA: 0.9min

------------------------------------------------------------
[Epoch 03] Summary | Time: 3.4s
Train Loss: 0.886158
Vali  Loss: 0.659232
Test  Loss: 0.458129
Validation loss decreased (0.660599 --> 0.659232).  Saving model ...
------------------------------------------------------------
Updating learning rate to 7.5e-05
[Epoch 04] Iter  100/244 | loss=0.79844 | 0.025s/iter | ETA: 1.7min
[Epoch 04] Iter  200/244 | loss=0.88828 | 0.009s/iter | ETA: 0.6min

------------------------------------------------------------
[Epoch 04] Summary | Time: 2.3s
Train Loss: 0.881437
Vali  Loss: 0.657433
Test  Loss: 0.458626
Validation loss decreased (0.659232 --> 0.657433).  Saving model ...
------------------------------------------------------------
Updating learning rate to 3.75e-05
[Epoch 05] Iter  100/244 | loss=0.63852 | 0.023s/iter | ETA: 1.5min
[Epoch 05] Iter  200/244 | loss=0.90844 | 0.009s/iter | ETA: 0.6min

------------------------------------------------------------
[Epoch 05] Summary | Time: 2.4s
Train Loss: 0.882446
Vali  Loss: 0.656709
Test  Loss: 0.458197
Validation loss decreased (0.657433 --> 0.656709).  Saving model ...
------------------------------------------------------------
Updating learning rate to 1.875e-05
[Epoch 06] Iter  100/244 | loss=0.56609 | 0.026s/iter | ETA: 1.5min
[Epoch 06] Iter  200/244 | loss=0.94816 | 0.011s/iter | ETA: 0.6min

------------------------------------------------------------
[Epoch 06] Summary | Time: 2.9s
Train Loss: 0.880391
Vali  Loss: 0.656710
Test  Loss: 0.457901
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 9.375e-06
[Epoch 07] Iter  100/244 | loss=0.56104 | 0.028s/iter | ETA: 1.6min
[Epoch 07] Iter  200/244 | loss=1.12209 | 0.012s/iter | ETA: 0.7min

------------------------------------------------------------
[Epoch 07] Summary | Time: 3.2s
Train Loss: 0.878993
Vali  Loss: 0.656241
Test  Loss: 0.457720
Validation loss decreased (0.656709 --> 0.656241).  Saving model ...
------------------------------------------------------------
Updating learning rate to 4.6875e-06
[Epoch 08] Iter  100/244 | loss=0.79313 | 0.029s/iter | ETA: 1.5min
[Epoch 08] Iter  200/244 | loss=0.71548 | 0.013s/iter | ETA: 0.6min

------------------------------------------------------------
[Epoch 08] Summary | Time: 3.3s
Train Loss: 0.878321
Vali  Loss: 0.658069
Test  Loss: 0.457660
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 2.34375e-06
[Epoch 09] Iter  100/244 | loss=0.98542 | 0.026s/iter | ETA: 1.2min
[Epoch 09] Iter  200/244 | loss=0.83070 | 0.009s/iter | ETA: 0.4min

------------------------------------------------------------
[Epoch 09] Summary | Time: 2.4s
Train Loss: 0.878423
Vali  Loss: 0.658609
Test  Loss: 0.457635
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 1.171875e-06
[Epoch 10] Iter  100/244 | loss=0.58343 | 0.025s/iter | ETA: 1.1min
[Epoch 10] Iter  200/244 | loss=0.74833 | 0.011s/iter | ETA: 0.4min

------------------------------------------------------------
[Epoch 10] Summary | Time: 2.7s
Train Loss: 0.878401
Vali  Loss: 0.658222
Test  Loss: 0.457628
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 5.859375e-07
[Epoch 11] Iter  100/244 | loss=1.55093 | 0.027s/iter | ETA: 1.0min
[Epoch 11] Iter  200/244 | loss=0.95968 | 0.011s/iter | ETA: 0.4min

------------------------------------------------------------
[Epoch 11] Summary | Time: 2.9s
Train Loss: 0.877708
Vali  Loss: 0.655874
Test  Loss: 0.457628
Validation loss decreased (0.656241 --> 0.655874).  Saving model ...
------------------------------------------------------------
Updating learning rate to 2.9296875e-07
[Epoch 12] Iter  100/244 | loss=1.03864 | 0.027s/iter | ETA: 0.9min
[Epoch 12] Iter  200/244 | loss=0.76353 | 0.012s/iter | ETA: 0.4min

------------------------------------------------------------
[Epoch 12] Summary | Time: 3.0s
Train Loss: 0.877903
Vali  Loss: 0.658301
Test  Loss: 0.457626
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 1.46484375e-07
[Epoch 13] Iter  100/244 | loss=0.71567 | 0.025s/iter | ETA: 0.8min
[Epoch 13] Iter  200/244 | loss=0.54644 | 0.009s/iter | ETA: 0.3min

------------------------------------------------------------
[Epoch 13] Summary | Time: 2.3s
Train Loss: 0.878794
Vali  Loss: 0.657790
Test  Loss: 0.457625
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 7.32421875e-08
[Epoch 14] Iter  100/244 | loss=0.67895 | 0.025s/iter | ETA: 0.7min
[Epoch 14] Iter  200/244 | loss=0.83753 | 0.011s/iter | ETA: 0.3min

------------------------------------------------------------
[Epoch 14] Summary | Time: 2.8s
Train Loss: 0.877566
Vali  Loss: 0.656710
Test  Loss: 0.457624
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 3.662109375e-08
[Epoch 15] Iter  100/244 | loss=1.04032 | 0.026s/iter | ETA: 0.6min
[Epoch 15] Iter  200/244 | loss=0.59769 | 0.009s/iter | ETA: 0.2min

------------------------------------------------------------
[Epoch 15] Summary | Time: 2.5s
Train Loss: 0.878607
Vali  Loss: 0.657176
Test  Loss: 0.457624
EarlyStopping counter: 4 out of 5
------------------------------------------------------------
Updating learning rate to 1.8310546875e-08
[Epoch 16] Iter  100/244 | loss=0.78944 | 0.026s/iter | ETA: 0.5min
[Epoch 16] Iter  200/244 | loss=0.62063 | 0.012s/iter | ETA: 0.2min

------------------------------------------------------------
[Epoch 16] Summary | Time: 3.0s
Train Loss: 0.878515
Vali  Loss: 0.656942
Test  Loss: 0.457624
EarlyStopping counter: 5 out of 5
------------------------------------------------------------

************************************************************
EARLY STOPPING TRIGGERED
************************************************************
>>>>>>>testing : ETTh2_96_720_FreDEA_ETTh2_ftM_sl96_ll48_pl720_dm64_nh8_el1_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 2161
mse:0.4559215307235718, mae:0.4631706774234772, rmse:0.6752195954322815
✅ 实验完成: MSE=0.455922, MAE=0.463171

================================================================================
运行实验: exchange - Pred 96
================================================================================
命令: python -u run_longExp.py --data exchange --data_path exchange_rate.csv --model FreDEA --model_id exchange_96_96 --seq_len 96 --pred_len 96 --enc_in 8 --dec_in 8 --c_out 8 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 32 --bottleneck_dim -1 --dropout 0.5 --batch_size 32 --learning_rate 0.0001 --lradj 3 --train_epochs 20 --patience 5 --itr 1 --num_workers 4

Args in experiment:
Namespace(is_training=1, train_only=False, model_id='exchange_96_96', model='FreDEA', data='exchange', root_path='./dataset/', data_path='exchange_rate.csv', channel_independence=0, features='M', target='OT', freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=96, individual=False, embed_type=0, enc_in=8, dec_in=8, c_out=8, d_model=64, reg_lambda=0.01, wavelet='haar', n_bins=8, n_heads=8, e_layers=1, d_layers=1, d_ff=128, moving_avg=25, factor=1, distil=True, rev_affine=1, memory_size=32, bottleneck_dim=-1, ablation_freq=0, ablation_tea=0, ablation_cea=0, dropout=0.5, embed='timeF', activation='gelu', output_attention=False, do_predict=False, num_workers=4, itr=1, train_epochs=20, batch_size=32, patience=5, learning_rate=0.0001, des='Exp', loss='mse', lradj='3', use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=False, devices='0,1,2,3', test_flop=False)
Use GPU: cuda:0
>>>>>>>start training : exchange_96_96_FreDEA_exchange_ftM_sl96_ll48_pl96_dm64_nh8_el1_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0>>>>>>>>>>>>>>>>>>>>>>>>>>
train 5120
val 665
test 1422

============================================================
TRAINING CONFIGURATION
============================================================
Model:           FreDEA
Dataset:         exchange
Seq/Pred Len:    96 -> 96
Batch Size:      32
Learning Rate:   0.0001
Train Epochs:    20
Total Params:    45,430
d_model:         64
n_heads:         8
e_layers:        1
============================================================

[Epoch 01] Iter  100/160 | loss=0.26931 | 0.017s/iter | ETA: 0.9min

------------------------------------------------------------
[Epoch 01] Summary | Time: 1.6s
Train Loss: 0.238285
Vali  Loss: 0.187510
Test  Loss: 0.133326
Validation loss decreased (inf --> 0.187510).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0001
[Epoch 02] Iter  100/160 | loss=0.23247 | 0.022s/iter | ETA: 1.1min

------------------------------------------------------------
[Epoch 02] Summary | Time: 1.9s
Train Loss: 0.205653
Vali  Loss: 0.175605
Test  Loss: 0.121305
Validation loss decreased (0.187510 --> 0.175605).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0001
[Epoch 03] Iter  100/160 | loss=0.13064 | 0.024s/iter | ETA: 1.1min

------------------------------------------------------------
[Epoch 03] Summary | Time: 1.8s
Train Loss: 0.185227
Vali  Loss: 0.160933
Test  Loss: 0.105738
Validation loss decreased (0.175605 --> 0.160933).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0001
[Epoch 04] Iter  100/160 | loss=0.21836 | 0.024s/iter | ETA: 1.0min

------------------------------------------------------------
[Epoch 04] Summary | Time: 1.7s
Train Loss: 0.166813
Vali  Loss: 0.151601
Test  Loss: 0.097976
Validation loss decreased (0.160933 --> 0.151601).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0001
[Epoch 05] Iter  100/160 | loss=0.15756 | 0.023s/iter | ETA: 0.9min

------------------------------------------------------------
[Epoch 05] Summary | Time: 1.9s
Train Loss: 0.155274
Vali  Loss: 0.143558
Test  Loss: 0.091769
Validation loss decreased (0.151601 --> 0.143558).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0001
[Epoch 06] Iter  100/160 | loss=0.25483 | 0.025s/iter | ETA: 0.9min

------------------------------------------------------------
[Epoch 06] Summary | Time: 2.0s
Train Loss: 0.147819
Vali  Loss: 0.140823
Test  Loss: 0.091458
Validation loss decreased (0.143558 --> 0.140823).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0001
[Epoch 07] Iter  100/160 | loss=0.12067 | 0.023s/iter | ETA: 0.8min

------------------------------------------------------------
[Epoch 07] Summary | Time: 1.5s
Train Loss: 0.142530
Vali  Loss: 0.141405
Test  Loss: 0.088265
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 0.0001
[Epoch 08] Iter  100/160 | loss=0.14219 | 0.024s/iter | ETA: 0.8min

------------------------------------------------------------
[Epoch 08] Summary | Time: 2.1s
Train Loss: 0.140007
Vali  Loss: 0.140545
Test  Loss: 0.086881
Validation loss decreased (0.140823 --> 0.140545).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0001
[Epoch 09] Iter  100/160 | loss=0.14802 | 0.026s/iter | ETA: 0.8min

------------------------------------------------------------
[Epoch 09] Summary | Time: 2.0s
Train Loss: 0.137077
Vali  Loss: 0.138989
Test  Loss: 0.087436
Validation loss decreased (0.140545 --> 0.138989).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0001
[Epoch 10] Iter  100/160 | loss=0.16283 | 0.025s/iter | ETA: 0.7min

------------------------------------------------------------
[Epoch 10] Summary | Time: 1.7s
Train Loss: 0.135380
Vali  Loss: 0.138611
Test  Loss: 0.087464
Validation loss decreased (0.138989 --> 0.138611).  Saving model ...
------------------------------------------------------------
Updating learning rate to 1e-05
[Epoch 11] Iter  100/160 | loss=0.26429 | 0.023s/iter | ETA: 0.6min

------------------------------------------------------------
[Epoch 11] Summary | Time: 1.7s
Train Loss: 0.134795
Vali  Loss: 0.138316
Test  Loss: 0.086919
Validation loss decreased (0.138611 --> 0.138316).  Saving model ...
------------------------------------------------------------
Updating learning rate to 1e-05
[Epoch 12] Iter  100/160 | loss=0.12595 | 0.023s/iter | ETA: 0.5min

------------------------------------------------------------
[Epoch 12] Summary | Time: 1.7s
Train Loss: 0.134449
Vali  Loss: 0.138915
Test  Loss: 0.086552
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 1e-05
[Epoch 13] Iter  100/160 | loss=0.12326 | 0.023s/iter | ETA: 0.4min

------------------------------------------------------------
[Epoch 13] Summary | Time: 1.9s
Train Loss: 0.134357
Vali  Loss: 0.137983
Test  Loss: 0.086270
Validation loss decreased (0.138316 --> 0.137983).  Saving model ...
------------------------------------------------------------
Updating learning rate to 1e-05
[Epoch 14] Iter  100/160 | loss=0.16969 | 0.026s/iter | ETA: 0.4min

------------------------------------------------------------
[Epoch 14] Summary | Time: 1.9s
Train Loss: 0.133904
Vali  Loss: 0.139500
Test  Loss: 0.086343
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 1e-05
[Epoch 15] Iter  100/160 | loss=0.09651 | 0.022s/iter | ETA: 0.3min

------------------------------------------------------------
[Epoch 15] Summary | Time: 1.4s
Train Loss: 0.133913
Vali  Loss: 0.137023
Test  Loss: 0.086408
Validation loss decreased (0.137983 --> 0.137023).  Saving model ...
------------------------------------------------------------
Updating learning rate to 1e-05
[Epoch 16] Iter  100/160 | loss=0.12293 | 0.021s/iter | ETA: 0.2min

------------------------------------------------------------
[Epoch 16] Summary | Time: 1.6s
Train Loss: 0.133493
Vali  Loss: 0.138591
Test  Loss: 0.086439
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 1e-05
[Epoch 17] Iter  100/160 | loss=0.10173 | 0.021s/iter | ETA: 0.2min

------------------------------------------------------------
[Epoch 17] Summary | Time: 1.7s
Train Loss: 0.132937
Vali  Loss: 0.138103
Test  Loss: 0.086185
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 1e-05
[Epoch 18] Iter  100/160 | loss=0.13154 | 0.022s/iter | ETA: 0.1min

------------------------------------------------------------
[Epoch 18] Summary | Time: 1.5s
Train Loss: 0.132456
Vali  Loss: 0.138502
Test  Loss: 0.086328
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 1e-05
[Epoch 19] Iter  100/160 | loss=0.14987 | 0.019s/iter | ETA: 0.1min

------------------------------------------------------------
[Epoch 19] Summary | Time: 1.5s
Train Loss: 0.132910
Vali  Loss: 0.138433
Test  Loss: 0.086175
EarlyStopping counter: 4 out of 5
------------------------------------------------------------
Updating learning rate to 1e-05
[Epoch 20] Iter  100/160 | loss=0.10682 | 0.020s/iter | ETA: 0.0min

------------------------------------------------------------
[Epoch 20] Summary | Time: 1.5s
Train Loss: 0.132515
Vali  Loss: 0.137922
Test  Loss: 0.086083
EarlyStopping counter: 5 out of 5
------------------------------------------------------------

************************************************************
EARLY STOPPING TRIGGERED
************************************************************
>>>>>>>testing : exchange_96_96_FreDEA_exchange_ftM_sl96_ll48_pl96_dm64_nh8_el1_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 1422
mse:0.08578086644411087, mae:0.20494331419467926, rmse:0.2928836941719055
✅ 实验完成: MSE=0.085781, MAE=0.204943

================================================================================
运行实验: exchange - Pred 192
================================================================================
命令: python -u run_longExp.py --data exchange --data_path exchange_rate.csv --model FreDEA --model_id exchange_96_192 --seq_len 96 --pred_len 192 --enc_in 8 --dec_in 8 --c_out 8 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 32 --bottleneck_dim -1 --dropout 0.5 --batch_size 32 --learning_rate 0.0001 --lradj 3 --train_epochs 20 --patience 5 --itr 1 --num_workers 4

Args in experiment:
Namespace(is_training=1, train_only=False, model_id='exchange_96_192', model='FreDEA', data='exchange', root_path='./dataset/', data_path='exchange_rate.csv', channel_independence=0, features='M', target='OT', freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=192, individual=False, embed_type=0, enc_in=8, dec_in=8, c_out=8, d_model=64, reg_lambda=0.01, wavelet='haar', n_bins=8, n_heads=8, e_layers=1, d_layers=1, d_ff=128, moving_avg=25, factor=1, distil=True, rev_affine=1, memory_size=32, bottleneck_dim=-1, ablation_freq=0, ablation_tea=0, ablation_cea=0, dropout=0.5, embed='timeF', activation='gelu', output_attention=False, do_predict=False, num_workers=4, itr=1, train_epochs=20, batch_size=32, patience=5, learning_rate=0.0001, des='Exp', loss='mse', lradj='3', use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=False, devices='0,1,2,3', test_flop=False)
Use GPU: cuda:0
>>>>>>>start training : exchange_96_192_FreDEA_exchange_ftM_sl96_ll48_pl192_dm64_nh8_el1_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0>>>>>>>>>>>>>>>>>>>>>>>>>>
train 5024
val 569
test 1326

============================================================
TRAINING CONFIGURATION
============================================================
Model:           FreDEA
Dataset:         exchange
Seq/Pred Len:    96 -> 192
Batch Size:      32
Learning Rate:   0.0001
Train Epochs:    20
Total Params:    51,670
d_model:         64
n_heads:         8
e_layers:        1
============================================================

[Epoch 01] Iter  100/157 | loss=0.29826 | 0.018s/iter | ETA: 0.9min

------------------------------------------------------------
[Epoch 01] Summary | Time: 1.7s
Train Loss: 0.343537
Vali  Loss: 0.305793
Test  Loss: 0.233840
Validation loss decreased (inf --> 0.305793).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0001
[Epoch 02] Iter  100/157 | loss=0.33359 | 0.022s/iter | ETA: 1.0min

------------------------------------------------------------
[Epoch 02] Summary | Time: 1.5s
Train Loss: 0.326118
Vali  Loss: 0.274360
Test  Loss: 0.225354
Validation loss decreased (0.305793 --> 0.274360).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0001
[Epoch 03] Iter  100/157 | loss=0.31285 | 0.018s/iter | ETA: 0.8min

------------------------------------------------------------
[Epoch 03] Summary | Time: 1.4s
Train Loss: 0.302780
Vali  Loss: 0.246542
Test  Loss: 0.197518
Validation loss decreased (0.274360 --> 0.246542).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0001
[Epoch 04] Iter  100/157 | loss=0.36413 | 0.024s/iter | ETA: 1.0min

------------------------------------------------------------
[Epoch 04] Summary | Time: 1.6s
Train Loss: 0.283960
Vali  Loss: 0.239651
Test  Loss: 0.187187
Validation loss decreased (0.246542 --> 0.239651).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0001
[Epoch 05] Iter  100/157 | loss=0.31354 | 0.026s/iter | ETA: 1.0min

------------------------------------------------------------
[Epoch 05] Summary | Time: 2.4s
Train Loss: 0.272802
Vali  Loss: 0.230315
Test  Loss: 0.184755
Validation loss decreased (0.239651 --> 0.230315).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0001
[Epoch 06] Iter  100/157 | loss=0.20350 | 0.027s/iter | ETA: 1.0min

------------------------------------------------------------
[Epoch 06] Summary | Time: 1.6s
Train Loss: 0.266457
Vali  Loss: 0.234576
Test  Loss: 0.179364
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 0.0001
[Epoch 07] Iter  100/157 | loss=0.27132 | 0.022s/iter | ETA: 0.8min

------------------------------------------------------------
[Epoch 07] Summary | Time: 1.6s
Train Loss: 0.261572
Vali  Loss: 0.233376
Test  Loss: 0.180917
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 0.0001
[Epoch 08] Iter  100/157 | loss=0.32271 | 0.022s/iter | ETA: 0.7min

------------------------------------------------------------
[Epoch 08] Summary | Time: 1.6s
Train Loss: 0.258815
Vali  Loss: 0.229712
Test  Loss: 0.179781
Validation loss decreased (0.230315 --> 0.229712).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0001
[Epoch 09] Iter  100/157 | loss=0.28702 | 0.022s/iter | ETA: 0.7min

------------------------------------------------------------
[Epoch 09] Summary | Time: 1.7s
Train Loss: 0.256236
Vali  Loss: 0.230950
Test  Loss: 0.179431
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 0.0001
[Epoch 10] Iter  100/157 | loss=0.22308 | 0.025s/iter | ETA: 0.7min

------------------------------------------------------------
[Epoch 10] Summary | Time: 2.1s
Train Loss: 0.254238
Vali  Loss: 0.228990
Test  Loss: 0.181384
Validation loss decreased (0.229712 --> 0.228990).  Saving model ...
------------------------------------------------------------
Updating learning rate to 1e-05
[Epoch 11] Iter  100/157 | loss=0.20265 | 0.024s/iter | ETA: 0.6min

------------------------------------------------------------
[Epoch 11] Summary | Time: 1.5s
Train Loss: 0.252931
Vali  Loss: 0.229856
Test  Loss: 0.181691
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 1e-05
[Epoch 12] Iter  100/157 | loss=0.32448 | 0.019s/iter | ETA: 0.4min

------------------------------------------------------------
[Epoch 12] Summary | Time: 1.4s
Train Loss: 0.252787
Vali  Loss: 0.229793
Test  Loss: 0.181879
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 1e-05
[Epoch 13] Iter  100/157 | loss=0.21076 | 0.024s/iter | ETA: 0.5min

------------------------------------------------------------
[Epoch 13] Summary | Time: 1.7s
Train Loss: 0.252927
Vali  Loss: 0.231826
Test  Loss: 0.181555
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 1e-05
[Epoch 14] Iter  100/157 | loss=0.28294 | 0.023s/iter | ETA: 0.4min

------------------------------------------------------------
[Epoch 14] Summary | Time: 1.8s
Train Loss: 0.252362
Vali  Loss: 0.228982
Test  Loss: 0.181449
Validation loss decreased (0.228990 --> 0.228982).  Saving model ...
------------------------------------------------------------
Updating learning rate to 1e-05
[Epoch 15] Iter  100/157 | loss=0.20575 | 0.022s/iter | ETA: 0.3min

------------------------------------------------------------
[Epoch 15] Summary | Time: 1.7s
Train Loss: 0.252280
Vali  Loss: 0.230876
Test  Loss: 0.181446
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 1e-05
[Epoch 16] Iter  100/157 | loss=0.22102 | 0.024s/iter | ETA: 0.3min

------------------------------------------------------------
[Epoch 16] Summary | Time: 1.7s
Train Loss: 0.252672
Vali  Loss: 0.229532
Test  Loss: 0.181440
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 1e-05
[Epoch 17] Iter  100/157 | loss=0.19040 | 0.023s/iter | ETA: 0.2min

------------------------------------------------------------
[Epoch 17] Summary | Time: 1.8s
Train Loss: 0.252390
Vali  Loss: 0.230665
Test  Loss: 0.181346
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 1e-05
[Epoch 18] Iter  100/157 | loss=0.18894 | 0.021s/iter | ETA: 0.1min

------------------------------------------------------------
[Epoch 18] Summary | Time: 1.4s
Train Loss: 0.251889
Vali  Loss: 0.229123
Test  Loss: 0.181811
EarlyStopping counter: 4 out of 5
------------------------------------------------------------
Updating learning rate to 1e-05
[Epoch 19] Iter  100/157 | loss=0.22293 | 0.024s/iter | ETA: 0.1min

------------------------------------------------------------
[Epoch 19] Summary | Time: 2.2s
Train Loss: 0.251527
Vali  Loss: 0.229396
Test  Loss: 0.181621
EarlyStopping counter: 5 out of 5
------------------------------------------------------------

************************************************************
EARLY STOPPING TRIGGERED
************************************************************
>>>>>>>testing : exchange_96_192_FreDEA_exchange_ftM_sl96_ll48_pl192_dm64_nh8_el1_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 1326
mse:0.18148523569107056, mae:0.3027130663394928, rmse:0.4260108470916748
✅ 实验完成: MSE=0.181485, MAE=0.302713

================================================================================
运行实验: exchange - Pred 336
================================================================================
命令: python -u run_longExp.py --data exchange --data_path exchange_rate.csv --model FreDEA --model_id exchange_96_336 --seq_len 96 --pred_len 336 --enc_in 8 --dec_in 8 --c_out 8 --d_model 32 --d_ff 64 --e_layers 1 --memory_size 32 --bottleneck_dim 1 --dropout 0.65 --batch_size 32 --learning_rate 0.0005 --lradj 3 --train_epochs 20 --patience 5 --itr 1 --num_workers 4 --rev_affine 0

Args in experiment:
Namespace(is_training=1, train_only=False, model_id='exchange_96_336', model='FreDEA', data='exchange', root_path='./dataset/', data_path='exchange_rate.csv', channel_independence=0, features='M', target='OT', freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=336, individual=False, embed_type=0, enc_in=8, dec_in=8, c_out=8, d_model=32, reg_lambda=0.01, wavelet='haar', n_bins=8, n_heads=8, e_layers=1, d_layers=1, d_ff=64, moving_avg=25, factor=1, distil=True, rev_affine=0, memory_size=32, bottleneck_dim=1, ablation_freq=0, ablation_tea=0, ablation_cea=0, dropout=0.65, embed='timeF', activation='gelu', output_attention=False, do_predict=False, num_workers=4, itr=1, train_epochs=20, batch_size=32, patience=5, learning_rate=0.0005, des='Exp', loss='mse', lradj='3', use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=False, devices='0,1,2,3', test_flop=False)
Use GPU: cuda:0
>>>>>>>start training : exchange_96_336_FreDEA_exchange_ftM_sl96_ll48_pl336_dm32_nh8_el1_dl1_df64_fc1_ebtimeF_dtTrue_Exp_0>>>>>>>>>>>>>>>>>>>>>>>>>>
train 4880
val 425
test 1182

============================================================
TRAINING CONFIGURATION
============================================================
Model:           FreDEA
Dataset:         exchange
Seq/Pred Len:    96 -> 336
Batch Size:      32
Learning Rate:   0.0005
Train Epochs:    20
Total Params:    31,382
d_model:         32
n_heads:         8
e_layers:        1
============================================================

[Epoch 01] Iter  100/152 | loss=0.47949 | 0.019s/iter | ETA: 0.9min

------------------------------------------------------------
[Epoch 01] Summary | Time: 1.9s
Train Loss: 0.498503
Vali  Loss: 0.426536
Test  Loss: 0.372496
Validation loss decreased (inf --> 0.426536).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 02] Iter  100/152 | loss=0.42693 | 0.018s/iter | ETA: 0.8min

------------------------------------------------------------
[Epoch 02] Summary | Time: 1.4s
Train Loss: 0.465191
Vali  Loss: 0.408313
Test  Loss: 0.343857
Validation loss decreased (0.426536 --> 0.408313).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 03] Iter  100/152 | loss=0.53453 | 0.025s/iter | ETA: 1.1min

------------------------------------------------------------
[Epoch 03] Summary | Time: 1.9s
Train Loss: 0.450845
Vali  Loss: 0.397632
Test  Loss: 0.335502
Validation loss decreased (0.408313 --> 0.397632).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 04] Iter  100/152 | loss=0.42584 | 0.021s/iter | ETA: 0.9min

------------------------------------------------------------
[Epoch 04] Summary | Time: 1.4s
Train Loss: 0.444301
Vali  Loss: 0.388761
Test  Loss: 0.332561
Validation loss decreased (0.397632 --> 0.388761).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 05] Iter  100/152 | loss=0.35299 | 0.016s/iter | ETA: 0.6min

------------------------------------------------------------
[Epoch 05] Summary | Time: 1.2s
Train Loss: 0.442358
Vali  Loss: 0.389853
Test  Loss: 0.338860
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 06] Iter  100/152 | loss=0.36320 | 0.021s/iter | ETA: 0.8min

------------------------------------------------------------
[Epoch 06] Summary | Time: 1.7s
Train Loss: 0.441300
Vali  Loss: 0.390950
Test  Loss: 0.344511
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 07] Iter  100/152 | loss=0.44506 | 0.020s/iter | ETA: 0.7min

------------------------------------------------------------
[Epoch 07] Summary | Time: 1.6s
Train Loss: 0.439790
Vali  Loss: 0.389498
Test  Loss: 0.342099
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 08] Iter  100/152 | loss=0.47163 | 0.022s/iter | ETA: 0.7min

------------------------------------------------------------
[Epoch 08] Summary | Time: 1.7s
Train Loss: 0.437006
Vali  Loss: 0.382680
Test  Loss: 0.338489
Validation loss decreased (0.388761 --> 0.382680).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 09] Iter  100/152 | loss=0.43713 | 0.021s/iter | ETA: 0.6min

------------------------------------------------------------
[Epoch 09] Summary | Time: 1.7s
Train Loss: 0.436303
Vali  Loss: 0.392200
Test  Loss: 0.340496
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 10] Iter  100/152 | loss=0.37005 | 0.022s/iter | ETA: 0.6min

------------------------------------------------------------
[Epoch 10] Summary | Time: 1.5s
Train Loss: 0.434705
Vali  Loss: 0.388201
Test  Loss: 0.341598
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 5e-05
[Epoch 11] Iter  100/152 | loss=0.44963 | 0.021s/iter | ETA: 0.5min

------------------------------------------------------------
[Epoch 11] Summary | Time: 1.7s
Train Loss: 0.432975
Vali  Loss: 0.388367
Test  Loss: 0.341191
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 5e-05
[Epoch 12] Iter  100/152 | loss=0.46774 | 0.023s/iter | ETA: 0.5min

------------------------------------------------------------
[Epoch 12] Summary | Time: 1.6s
Train Loss: 0.432436
Vali  Loss: 0.386117
Test  Loss: 0.341087
EarlyStopping counter: 4 out of 5
------------------------------------------------------------
Updating learning rate to 5e-05
[Epoch 13] Iter  100/152 | loss=0.38446 | 0.020s/iter | ETA: 0.4min

------------------------------------------------------------
[Epoch 13] Summary | Time: 1.4s
Train Loss: 0.433009
Vali  Loss: 0.389320
Test  Loss: 0.342818
EarlyStopping counter: 5 out of 5
------------------------------------------------------------

************************************************************
EARLY STOPPING TRIGGERED
************************************************************
>>>>>>>testing : exchange_96_336_FreDEA_exchange_ftM_sl96_ll48_pl336_dm32_nh8_el1_dl1_df64_fc1_ebtimeF_dtTrue_Exp_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 1182
mse:0.3385639786720276, mae:0.4211060404777527, rmse:0.5818625092506409
✅ 实验完成: MSE=0.338564, MAE=0.421106

================================================================================
运行实验: exchange - Pred 720
================================================================================
命令: python -u run_longExp.py --data exchange --data_path exchange_rate.csv --model FreDEA --model_id exchange_96_720 --seq_len 96 --pred_len 720 --enc_in 8 --dec_in 8 --c_out 8 --d_model 32 --d_ff 64 --e_layers 1 --memory_size 32 --bottleneck_dim 1 --dropout 0.6 --batch_size 32 --learning_rate 0.0005 --lradj 3 --train_epochs 20 --patience 5 --itr 1 --num_workers 4 --rev_affine 1

Args in experiment:
Namespace(is_training=1, train_only=False, model_id='exchange_96_720', model='FreDEA', data='exchange', root_path='./dataset/', data_path='exchange_rate.csv', channel_independence=0, features='M', target='OT', freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=720, individual=False, embed_type=0, enc_in=8, dec_in=8, c_out=8, d_model=32, reg_lambda=0.01, wavelet='haar', n_bins=8, n_heads=8, e_layers=1, d_layers=1, d_ff=64, moving_avg=25, factor=1, distil=True, rev_affine=1, memory_size=32, bottleneck_dim=1, ablation_freq=0, ablation_tea=0, ablation_cea=0, dropout=0.6, embed='timeF', activation='gelu', output_attention=False, do_predict=False, num_workers=4, itr=1, train_epochs=20, batch_size=32, patience=5, learning_rate=0.0005, des='Exp', loss='mse', lradj='3', use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=False, devices='0,1,2,3', test_flop=False)
Use GPU: cuda:0
>>>>>>>start training : exchange_96_720_FreDEA_exchange_ftM_sl96_ll48_pl720_dm32_nh8_el1_dl1_df64_fc1_ebtimeF_dtTrue_Exp_0>>>>>>>>>>>>>>>>>>>>>>>>>>
train 4496
val 41
test 798

============================================================
TRAINING CONFIGURATION
============================================================
Model:           FreDEA
Dataset:         exchange
Seq/Pred Len:    96 -> 720
Batch Size:      32
Learning Rate:   0.0005
Train Epochs:    20
Total Params:    44,070
d_model:         32
n_heads:         8
e_layers:        1
============================================================

[Epoch 01] Iter  100/140 | loss=0.97851 | 0.023s/iter | ETA: 1.0min

------------------------------------------------------------
[Epoch 01] Summary | Time: 2.2s
Train Loss: 0.870575
Vali  Loss: 1.191822
Test  Loss: 0.864627
Validation loss decreased (inf --> 1.191822).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 02] Iter  100/140 | loss=0.84632 | 0.023s/iter | ETA: 1.0min

------------------------------------------------------------
[Epoch 02] Summary | Time: 2.0s
Train Loss: 0.832752
Vali  Loss: 1.051413
Test  Loss: 0.853973
Validation loss decreased (1.191822 --> 1.051413).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 03] Iter  100/140 | loss=0.88389 | 0.020s/iter | ETA: 0.8min

------------------------------------------------------------
[Epoch 03] Summary | Time: 1.4s
Train Loss: 0.820667
Vali  Loss: 1.025840
Test  Loss: 0.861029
Validation loss decreased (1.051413 --> 1.025840).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 04] Iter  100/140 | loss=0.85830 | 0.019s/iter | ETA: 0.7min

------------------------------------------------------------
[Epoch 04] Summary | Time: 1.7s
Train Loss: 0.813426
Vali  Loss: 1.051460
Test  Loss: 0.859044
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 05] Iter  100/140 | loss=0.95711 | 0.021s/iter | ETA: 0.7min

------------------------------------------------------------
[Epoch 05] Summary | Time: 1.7s
Train Loss: 0.808886
Vali  Loss: 1.035226
Test  Loss: 0.862376
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 06] Iter  100/140 | loss=0.85236 | 0.023s/iter | ETA: 0.8min

------------------------------------------------------------
[Epoch 06] Summary | Time: 1.9s
Train Loss: 0.805553
Vali  Loss: 1.033158
Test  Loss: 0.848340
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 07] Iter  100/140 | loss=0.83801 | 0.023s/iter | ETA: 0.7min

------------------------------------------------------------
[Epoch 07] Summary | Time: 1.8s
Train Loss: 0.800683
Vali  Loss: 1.049371
Test  Loss: 0.863375
EarlyStopping counter: 4 out of 5
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 08] Iter  100/140 | loss=0.80498 | 0.022s/iter | ETA: 0.6min

------------------------------------------------------------
[Epoch 08] Summary | Time: 1.7s
Train Loss: 0.795934
Vali  Loss: 1.081211
Test  Loss: 0.848260
EarlyStopping counter: 5 out of 5
------------------------------------------------------------

************************************************************
EARLY STOPPING TRIGGERED
************************************************************
>>>>>>>testing : exchange_96_720_FreDEA_exchange_ftM_sl96_ll48_pl720_dm32_nh8_el1_dl1_df64_fc1_ebtimeF_dtTrue_Exp_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 798
mse:0.8619589805603027, mae:0.6990378499031067, rmse:0.928417444229126
✅ 实验完成: MSE=0.861959, MAE=0.699038

================================================================================
运行实验: weather - Pred 96
================================================================================
命令: python -u run_longExp.py --data weather --data_path weather.csv --model FreDEA --model_id weather_96_96 --seq_len 96 --pred_len 96 --enc_in 21 --dec_in 21 --c_out 21 --d_model 128 --d_ff 256 --e_layers 2 --memory_size 128 --bottleneck_dim 4 --dropout 0.1 --batch_size 64 --learning_rate 0.0005 --lradj 3 --train_epochs 30 --patience 5 --itr 1 --num_workers 4

Args in experiment:
Namespace(is_training=1, train_only=False, model_id='weather_96_96', model='FreDEA', data='weather', root_path='./dataset/', data_path='weather.csv', channel_independence=0, features='M', target='OT', freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=96, individual=False, embed_type=0, enc_in=21, dec_in=21, c_out=21, d_model=128, reg_lambda=0.01, wavelet='haar', n_bins=8, n_heads=8, e_layers=2, d_layers=1, d_ff=256, moving_avg=25, factor=1, distil=True, rev_affine=1, memory_size=128, bottleneck_dim=4, ablation_freq=0, ablation_tea=0, ablation_cea=0, dropout=0.1, embed='timeF', activation='gelu', output_attention=False, do_predict=False, num_workers=4, itr=1, train_epochs=30, batch_size=64, patience=5, learning_rate=0.0005, des='Exp', loss='mse', lradj='3', use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=False, devices='0,1,2,3', test_flop=False)
Use GPU: cuda:0
>>>>>>>start training : weather_96_96_FreDEA_weather_ftM_sl96_ll48_pl96_dm128_nh8_el2_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0>>>>>>>>>>>>>>>>>>>>>>>>>>
train 36696
val 5175
test 10444

============================================================
TRAINING CONFIGURATION
============================================================
Model:           FreDEA
Dataset:         weather
Seq/Pred Len:    96 -> 96
Batch Size:      64
Learning Rate:   0.0005
Train Epochs:    30
Total Params:    240,016
d_model:         128
n_heads:         8
e_layers:        2
============================================================

[Epoch 01] Iter  100/573 | loss=0.46686 | 0.038s/iter | ETA: 10.9min
[Epoch 01] Iter  200/573 | loss=0.78721 | 0.028s/iter | ETA: 7.8min
[Epoch 01] Iter  300/573 | loss=0.52296 | 0.028s/iter | ETA: 7.8min
[Epoch 01] Iter  400/573 | loss=0.31472 | 0.028s/iter | ETA: 7.8min
[Epoch 01] Iter  500/573 | loss=0.43435 | 0.028s/iter | ETA: 7.7min

------------------------------------------------------------
[Epoch 01] Summary | Time: 16.4s
Train Loss: 0.522930
Vali  Loss: 0.441223
Test  Loss: 0.182653
Validation loss decreased (inf --> 0.441223).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 02] Iter  100/573 | loss=0.28020 | 0.077s/iter | ETA: 21.2min
[Epoch 02] Iter  200/573 | loss=0.31908 | 0.028s/iter | ETA: 7.6min
[Epoch 02] Iter  300/573 | loss=0.31426 | 0.028s/iter | ETA: 7.5min
[Epoch 02] Iter  400/573 | loss=0.31225 | 0.028s/iter | ETA: 7.5min
[Epoch 02] Iter  500/573 | loss=0.41526 | 0.028s/iter | ETA: 7.4min

------------------------------------------------------------
[Epoch 02] Summary | Time: 16.0s
Train Loss: 0.436563
Vali  Loss: 0.408980
Test  Loss: 0.160575
Validation loss decreased (0.441223 --> 0.408980).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 03] Iter  100/573 | loss=0.39604 | 0.078s/iter | ETA: 20.7min
[Epoch 03] Iter  200/573 | loss=0.30164 | 0.027s/iter | ETA: 7.1min
[Epoch 03] Iter  300/573 | loss=0.29237 | 0.027s/iter | ETA: 7.0min
[Epoch 03] Iter  400/573 | loss=0.33235 | 0.027s/iter | ETA: 7.0min
[Epoch 03] Iter  500/573 | loss=0.27127 | 0.027s/iter | ETA: 6.9min

------------------------------------------------------------
[Epoch 03] Summary | Time: 15.5s
Train Loss: 0.412561
Vali  Loss: 0.391682
Test  Loss: 0.157818
Validation loss decreased (0.408980 --> 0.391682).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 04] Iter  100/573 | loss=0.30683 | 0.078s/iter | ETA: 20.0min
[Epoch 04] Iter  200/573 | loss=0.25443 | 0.027s/iter | ETA: 6.8min
[Epoch 04] Iter  300/573 | loss=0.32484 | 0.027s/iter | ETA: 6.8min
[Epoch 04] Iter  400/573 | loss=0.30187 | 0.027s/iter | ETA: 6.9min
[Epoch 04] Iter  500/573 | loss=0.41732 | 0.028s/iter | ETA: 6.9min

------------------------------------------------------------
[Epoch 04] Summary | Time: 15.8s
Train Loss: 0.401472
Vali  Loss: 0.391646
Test  Loss: 0.156709
Validation loss decreased (0.391682 --> 0.391646).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 05] Iter  100/573 | loss=0.27878 | 0.078s/iter | ETA: 19.2min
[Epoch 05] Iter  200/573 | loss=0.27996 | 0.028s/iter | ETA: 6.8min
[Epoch 05] Iter  300/573 | loss=0.32854 | 0.028s/iter | ETA: 6.7min
[Epoch 05] Iter  400/573 | loss=0.77059 | 0.028s/iter | ETA: 6.7min
[Epoch 05] Iter  500/573 | loss=0.30429 | 0.028s/iter | ETA: 6.6min

------------------------------------------------------------
[Epoch 05] Summary | Time: 16.0s
Train Loss: 0.394008
Vali  Loss: 0.385393
Test  Loss: 0.154380
Validation loss decreased (0.391646 --> 0.385393).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 06] Iter  100/573 | loss=0.81706 | 0.082s/iter | ETA: 19.4min
[Epoch 06] Iter  200/573 | loss=0.35635 | 0.028s/iter | ETA: 6.5min
[Epoch 06] Iter  300/573 | loss=0.42250 | 0.027s/iter | ETA: 6.4min
[Epoch 06] Iter  400/573 | loss=0.21087 | 0.027s/iter | ETA: 6.2min
[Epoch 06] Iter  500/573 | loss=0.22160 | 0.027s/iter | ETA: 6.1min

------------------------------------------------------------
[Epoch 06] Summary | Time: 15.7s
Train Loss: 0.387366
Vali  Loss: 0.389339
Test  Loss: 0.152485
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 07] Iter  100/573 | loss=0.26780 | 0.077s/iter | ETA: 17.4min
[Epoch 07] Iter  200/573 | loss=0.78976 | 0.028s/iter | ETA: 6.2min
[Epoch 07] Iter  300/573 | loss=0.36334 | 0.028s/iter | ETA: 6.2min
[Epoch 07] Iter  400/573 | loss=0.25160 | 0.028s/iter | ETA: 6.1min
[Epoch 07] Iter  500/573 | loss=0.89715 | 0.028s/iter | ETA: 6.1min

------------------------------------------------------------
[Epoch 07] Summary | Time: 15.9s
Train Loss: 0.382421
Vali  Loss: 0.384590
Test  Loss: 0.153543
Validation loss decreased (0.385393 --> 0.384590).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 08] Iter  100/573 | loss=0.77176 | 0.077s/iter | ETA: 16.9min
[Epoch 08] Iter  200/573 | loss=0.26309 | 0.027s/iter | ETA: 5.9min
[Epoch 08] Iter  300/573 | loss=0.25345 | 0.027s/iter | ETA: 5.9min
[Epoch 08] Iter  400/573 | loss=0.27316 | 0.027s/iter | ETA: 5.8min
[Epoch 08] Iter  500/573 | loss=0.29893 | 0.028s/iter | ETA: 5.8min

------------------------------------------------------------
[Epoch 08] Summary | Time: 15.8s
Train Loss: 0.378690
Vali  Loss: 0.384533
Test  Loss: 0.154547
Validation loss decreased (0.384590 --> 0.384533).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 09] Iter  100/573 | loss=0.23554 | 0.077s/iter | ETA: 16.1min
[Epoch 09] Iter  200/573 | loss=0.26797 | 0.028s/iter | ETA: 5.7min
[Epoch 09] Iter  300/573 | loss=0.35354 | 0.028s/iter | ETA: 5.7min
[Epoch 09] Iter  400/573 | loss=0.29984 | 0.028s/iter | ETA: 5.6min
[Epoch 09] Iter  500/573 | loss=0.35386 | 0.028s/iter | ETA: 5.6min

------------------------------------------------------------
[Epoch 09] Summary | Time: 16.0s
Train Loss: 0.374228
Vali  Loss: 0.386311
Test  Loss: 0.157383
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 10] Iter  100/573 | loss=0.41265 | 0.077s/iter | ETA: 15.4min
[Epoch 10] Iter  200/573 | loss=0.67795 | 0.028s/iter | ETA: 5.4min
[Epoch 10] Iter  300/573 | loss=0.28349 | 0.028s/iter | ETA: 5.4min
[Epoch 10] Iter  400/573 | loss=0.23587 | 0.028s/iter | ETA: 5.3min
[Epoch 10] Iter  500/573 | loss=0.31847 | 0.027s/iter | ETA: 5.3min

------------------------------------------------------------
[Epoch 10] Summary | Time: 15.9s
Train Loss: 0.370592
Vali  Loss: 0.382631
Test  Loss: 0.153933
Validation loss decreased (0.384533 --> 0.382631).  Saving model ...
------------------------------------------------------------
Updating learning rate to 5e-05
[Epoch 11] Iter  100/573 | loss=0.35890 | 0.078s/iter | ETA: 14.7min
[Epoch 11] Iter  200/573 | loss=0.32941 | 0.028s/iter | ETA: 5.2min
[Epoch 11] Iter  300/573 | loss=0.25816 | 0.028s/iter | ETA: 5.1min
[Epoch 11] Iter  400/573 | loss=0.74882 | 0.027s/iter | ETA: 5.1min
[Epoch 11] Iter  500/573 | loss=0.37658 | 0.028s/iter | ETA: 5.0min

------------------------------------------------------------
[Epoch 11] Summary | Time: 15.9s
Train Loss: 0.365550
Vali  Loss: 0.383004
Test  Loss: 0.154128
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 5e-05
[Epoch 12] Iter  100/573 | loss=0.36764 | 0.077s/iter | ETA: 13.9min
[Epoch 12] Iter  200/573 | loss=0.20463 | 0.028s/iter | ETA: 4.9min
[Epoch 12] Iter  300/573 | loss=0.21772 | 0.028s/iter | ETA: 4.9min
[Epoch 12] Iter  400/573 | loss=0.33016 | 0.028s/iter | ETA: 4.8min
[Epoch 12] Iter  500/573 | loss=0.85039 | 0.028s/iter | ETA: 4.8min

------------------------------------------------------------
[Epoch 12] Summary | Time: 15.9s
Train Loss: 0.363452
Vali  Loss: 0.379891
Test  Loss: 0.154191
Validation loss decreased (0.382631 --> 0.379891).  Saving model ...
------------------------------------------------------------
Updating learning rate to 5e-05
[Epoch 13] Iter  100/573 | loss=0.27910 | 0.078s/iter | ETA: 13.3min
[Epoch 13] Iter  200/573 | loss=0.30197 | 0.028s/iter | ETA: 4.7min
[Epoch 13] Iter  300/573 | loss=0.27954 | 0.028s/iter | ETA: 4.6min
[Epoch 13] Iter  400/573 | loss=0.44018 | 0.028s/iter | ETA: 4.6min
[Epoch 13] Iter  500/573 | loss=0.30128 | 0.028s/iter | ETA: 4.5min

------------------------------------------------------------
[Epoch 13] Summary | Time: 16.0s
Train Loss: 0.362673
Vali  Loss: 0.383100
Test  Loss: 0.154873
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 5e-05
[Epoch 14] Iter  100/573 | loss=0.80456 | 0.078s/iter | ETA: 12.6min
[Epoch 14] Iter  200/573 | loss=0.18418 | 0.028s/iter | ETA: 4.4min
[Epoch 14] Iter  300/573 | loss=1.07911 | 0.028s/iter | ETA: 4.3min
[Epoch 14] Iter  400/573 | loss=0.84844 | 0.028s/iter | ETA: 4.3min
[Epoch 14] Iter  500/573 | loss=0.32289 | 0.028s/iter | ETA: 4.3min

------------------------------------------------------------
[Epoch 14] Summary | Time: 16.0s
Train Loss: 0.362136
Vali  Loss: 0.383275
Test  Loss: 0.154815
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 5e-05
[Epoch 15] Iter  100/573 | loss=0.27764 | 0.078s/iter | ETA: 11.7min
[Epoch 15] Iter  200/573 | loss=0.29689 | 0.028s/iter | ETA: 4.1min
[Epoch 15] Iter  300/573 | loss=0.67600 | 0.028s/iter | ETA: 4.1min
[Epoch 15] Iter  400/573 | loss=0.62029 | 0.028s/iter | ETA: 4.0min
[Epoch 15] Iter  500/573 | loss=0.27035 | 0.028s/iter | ETA: 4.0min

------------------------------------------------------------
[Epoch 15] Summary | Time: 15.9s
Train Loss: 0.361566
Vali  Loss: 0.381919
Test  Loss: 0.154471
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 5e-05
[Epoch 16] Iter  100/573 | loss=0.24142 | 0.077s/iter | ETA: 10.8min
[Epoch 16] Iter  200/573 | loss=0.32755 | 0.028s/iter | ETA: 3.9min
[Epoch 16] Iter  300/573 | loss=0.27229 | 0.028s/iter | ETA: 3.8min
[Epoch 16] Iter  400/573 | loss=0.42025 | 0.028s/iter | ETA: 3.8min
[Epoch 16] Iter  500/573 | loss=0.22231 | 0.028s/iter | ETA: 3.7min

------------------------------------------------------------
[Epoch 16] Summary | Time: 15.9s
Train Loss: 0.360525
Vali  Loss: 0.380682
Test  Loss: 0.155055
EarlyStopping counter: 4 out of 5
------------------------------------------------------------
Updating learning rate to 5e-05
[Epoch 17] Iter  100/573 | loss=0.73279 | 0.078s/iter | ETA: 10.2min
[Epoch 17] Iter  200/573 | loss=0.31404 | 0.027s/iter | ETA: 3.6min
[Epoch 17] Iter  300/573 | loss=0.29331 | 0.028s/iter | ETA: 3.6min
[Epoch 17] Iter  400/573 | loss=0.24427 | 0.028s/iter | ETA: 3.5min
[Epoch 17] Iter  500/573 | loss=0.84020 | 0.027s/iter | ETA: 3.4min

------------------------------------------------------------
[Epoch 17] Summary | Time: 15.9s
Train Loss: 0.360138
Vali  Loss: 0.381673
Test  Loss: 0.155221
EarlyStopping counter: 5 out of 5
------------------------------------------------------------

************************************************************
EARLY STOPPING TRIGGERED
************************************************************
>>>>>>>testing : weather_96_96_FreDEA_weather_ftM_sl96_ll48_pl96_dm128_nh8_el2_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 10444
mse:0.15468180179595947, mae:0.20213642716407776, rmse:0.3932960629463196
✅ 实验完成: MSE=0.154682, MAE=0.202136

================================================================================
运行实验: weather - Pred 192
================================================================================
命令: python -u run_longExp.py --data weather --data_path weather.csv --model FreDEA --model_id weather_96_192 --seq_len 96 --pred_len 192 --enc_in 21 --dec_in 21 --c_out 21 --d_model 128 --d_ff 256 --e_layers 2 --memory_size 128 --bottleneck_dim 4 --dropout 0.1 --batch_size 64 --learning_rate 0.0005 --lradj 3 --train_epochs 30 --patience 5 --itr 1 --num_workers 4

Args in experiment:
Namespace(is_training=1, train_only=False, model_id='weather_96_192', model='FreDEA', data='weather', root_path='./dataset/', data_path='weather.csv', channel_independence=0, features='M', target='OT', freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=192, individual=False, embed_type=0, enc_in=21, dec_in=21, c_out=21, d_model=128, reg_lambda=0.01, wavelet='haar', n_bins=8, n_heads=8, e_layers=2, d_layers=1, d_ff=256, moving_avg=25, factor=1, distil=True, rev_affine=1, memory_size=128, bottleneck_dim=4, ablation_freq=0, ablation_tea=0, ablation_cea=0, dropout=0.1, embed='timeF', activation='gelu', output_attention=False, do_predict=False, num_workers=4, itr=1, train_epochs=30, batch_size=64, patience=5, learning_rate=0.0005, des='Exp', loss='mse', lradj='3', use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=False, devices='0,1,2,3', test_flop=False)
Use GPU: cuda:0
>>>>>>>start training : weather_96_192_FreDEA_weather_ftM_sl96_ll48_pl192_dm128_nh8_el2_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0>>>>>>>>>>>>>>>>>>>>>>>>>>
train 36600
val 5079
test 10348

============================================================
TRAINING CONFIGURATION
============================================================
Model:           FreDEA
Dataset:         weather
Seq/Pred Len:    96 -> 192
Batch Size:      64
Learning Rate:   0.0005
Train Epochs:    30
Total Params:    252,400
d_model:         128
n_heads:         8
e_layers:        2
============================================================

[Epoch 01] Iter  100/571 | loss=0.57247 | 0.039s/iter | ETA: 11.1min
[Epoch 01] Iter  200/571 | loss=0.71672 | 0.028s/iter | ETA: 7.8min
[Epoch 01] Iter  300/571 | loss=0.49393 | 0.028s/iter | ETA: 7.8min
[Epoch 01] Iter  400/571 | loss=1.11824 | 0.028s/iter | ETA: 7.7min
[Epoch 01] Iter  500/571 | loss=0.42034 | 0.028s/iter | ETA: 7.7min

------------------------------------------------------------
[Epoch 01] Summary | Time: 16.4s
Train Loss: 0.579267
Vali  Loss: 0.498740
Test  Loss: 0.225737
Validation loss decreased (inf --> 0.498740).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 02] Iter  100/571 | loss=0.46305 | 0.078s/iter | ETA: 21.5min
[Epoch 02] Iter  200/571 | loss=0.70052 | 0.028s/iter | ETA: 7.6min
[Epoch 02] Iter  300/571 | loss=0.41978 | 0.028s/iter | ETA: 7.5min
[Epoch 02] Iter  400/571 | loss=0.38188 | 0.028s/iter | ETA: 7.5min
[Epoch 02] Iter  500/571 | loss=0.62648 | 0.028s/iter | ETA: 7.4min

------------------------------------------------------------
[Epoch 02] Summary | Time: 16.0s
Train Loss: 0.507808
Vali  Loss: 0.490057
Test  Loss: 0.220012
Validation loss decreased (0.498740 --> 0.490057).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 03] Iter  100/571 | loss=0.37535 | 0.078s/iter | ETA: 20.6min
[Epoch 03] Iter  200/571 | loss=0.59792 | 0.028s/iter | ETA: 7.3min
[Epoch 03] Iter  300/571 | loss=0.53808 | 0.028s/iter | ETA: 7.3min
[Epoch 03] Iter  400/571 | loss=0.67761 | 0.028s/iter | ETA: 7.2min
[Epoch 03] Iter  500/571 | loss=0.65371 | 0.028s/iter | ETA: 7.2min

------------------------------------------------------------
[Epoch 03] Summary | Time: 16.0s
Train Loss: 0.491305
Vali  Loss: 0.465119
Test  Loss: 0.208252
Validation loss decreased (0.490057 --> 0.465119).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 04] Iter  100/571 | loss=0.43802 | 0.079s/iter | ETA: 20.2min
[Epoch 04] Iter  200/571 | loss=0.41151 | 0.028s/iter | ETA: 7.1min
[Epoch 04] Iter  300/571 | loss=0.34765 | 0.028s/iter | ETA: 7.0min
[Epoch 04] Iter  400/571 | loss=0.61786 | 0.028s/iter | ETA: 7.0min
[Epoch 04] Iter  500/571 | loss=0.40301 | 0.028s/iter | ETA: 6.9min

------------------------------------------------------------
[Epoch 04] Summary | Time: 16.1s
Train Loss: 0.474226
Vali  Loss: 0.460243
Test  Loss: 0.204870
Validation loss decreased (0.465119 --> 0.460243).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 05] Iter  100/571 | loss=0.39758 | 0.079s/iter | ETA: 19.3min
[Epoch 05] Iter  200/571 | loss=0.34658 | 0.028s/iter | ETA: 6.7min
[Epoch 05] Iter  300/571 | loss=0.59460 | 0.028s/iter | ETA: 6.7min
[Epoch 05] Iter  400/571 | loss=0.32917 | 0.028s/iter | ETA: 6.6min
[Epoch 05] Iter  500/571 | loss=0.44225 | 0.028s/iter | ETA: 6.6min

------------------------------------------------------------
[Epoch 05] Summary | Time: 15.9s
Train Loss: 0.463243
Vali  Loss: 0.454116
Test  Loss: 0.205970
Validation loss decreased (0.460243 --> 0.454116).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 06] Iter  100/571 | loss=0.40392 | 0.078s/iter | ETA: 18.5min
[Epoch 06] Iter  200/571 | loss=0.60318 | 0.028s/iter | ETA: 6.5min
[Epoch 06] Iter  300/571 | loss=0.37298 | 0.028s/iter | ETA: 6.5min
[Epoch 06] Iter  400/571 | loss=0.30386 | 0.028s/iter | ETA: 6.5min
[Epoch 06] Iter  500/571 | loss=0.35624 | 0.028s/iter | ETA: 6.4min

------------------------------------------------------------
[Epoch 06] Summary | Time: 16.0s
Train Loss: 0.455290
Vali  Loss: 0.453406
Test  Loss: 0.204828
Validation loss decreased (0.454116 --> 0.453406).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 07] Iter  100/571 | loss=0.34513 | 0.080s/iter | ETA: 18.2min
[Epoch 07] Iter  200/571 | loss=0.81915 | 0.028s/iter | ETA: 6.3min
[Epoch 07] Iter  300/571 | loss=0.55186 | 0.028s/iter | ETA: 6.2min
[Epoch 07] Iter  400/571 | loss=0.36176 | 0.028s/iter | ETA: 6.2min
[Epoch 07] Iter  500/571 | loss=0.84213 | 0.028s/iter | ETA: 6.2min

------------------------------------------------------------
[Epoch 07] Summary | Time: 16.2s
Train Loss: 0.448882
Vali  Loss: 0.456855
Test  Loss: 0.203679
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 08] Iter  100/571 | loss=0.38163 | 0.081s/iter | ETA: 17.6min
[Epoch 08] Iter  200/571 | loss=0.31097 | 0.028s/iter | ETA: 6.0min
[Epoch 08] Iter  300/571 | loss=0.39008 | 0.028s/iter | ETA: 5.9min
[Epoch 08] Iter  400/571 | loss=0.34250 | 0.028s/iter | ETA: 5.9min
[Epoch 08] Iter  500/571 | loss=0.59024 | 0.028s/iter | ETA: 5.9min

------------------------------------------------------------
[Epoch 08] Summary | Time: 16.0s
Train Loss: 0.443379
Vali  Loss: 0.456940
Test  Loss: 0.206304
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 09] Iter  100/571 | loss=0.37041 | 0.079s/iter | ETA: 16.5min
[Epoch 09] Iter  200/571 | loss=0.55982 | 0.028s/iter | ETA: 5.7min
[Epoch 09] Iter  300/571 | loss=0.43543 | 0.028s/iter | ETA: 5.7min
[Epoch 09] Iter  400/571 | loss=0.45460 | 0.028s/iter | ETA: 5.7min
[Epoch 09] Iter  500/571 | loss=0.28872 | 0.028s/iter | ETA: 5.6min

------------------------------------------------------------
[Epoch 09] Summary | Time: 16.1s
Train Loss: 0.437680
Vali  Loss: 0.453909
Test  Loss: 0.206347
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 10] Iter  100/571 | loss=0.56225 | 0.079s/iter | ETA: 15.6min
[Epoch 10] Iter  200/571 | loss=0.56813 | 0.028s/iter | ETA: 5.4min
[Epoch 10] Iter  300/571 | loss=0.49781 | 0.028s/iter | ETA: 5.4min
[Epoch 10] Iter  400/571 | loss=0.31356 | 0.028s/iter | ETA: 5.4min
[Epoch 10] Iter  500/571 | loss=0.30037 | 0.028s/iter | ETA: 5.3min

------------------------------------------------------------
[Epoch 10] Summary | Time: 16.0s
Train Loss: 0.432125
Vali  Loss: 0.458434
Test  Loss: 0.207560
EarlyStopping counter: 4 out of 5
------------------------------------------------------------
Updating learning rate to 5e-05
[Epoch 11] Iter  100/571 | loss=0.31564 | 0.079s/iter | ETA: 14.8min
[Epoch 11] Iter  200/571 | loss=0.33886 | 0.028s/iter | ETA: 5.2min
[Epoch 11] Iter  300/571 | loss=0.27677 | 0.028s/iter | ETA: 5.1min
[Epoch 11] Iter  400/571 | loss=0.33865 | 0.028s/iter | ETA: 5.1min
[Epoch 11] Iter  500/571 | loss=0.66615 | 0.028s/iter | ETA: 5.1min

------------------------------------------------------------
[Epoch 11] Summary | Time: 16.0s
Train Loss: 0.425096
Vali  Loss: 0.453834
Test  Loss: 0.207022
EarlyStopping counter: 5 out of 5
------------------------------------------------------------

************************************************************
EARLY STOPPING TRIGGERED
************************************************************
>>>>>>>testing : weather_96_192_FreDEA_weather_ftM_sl96_ll48_pl192_dm128_nh8_el2_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 10348
mse:0.20510391891002655, mae:0.24965310096740723, rmse:0.4528839886188507
✅ 实验完成: MSE=0.205104, MAE=0.249653

================================================================================
运行实验: weather - Pred 336
================================================================================
命令: python -u run_longExp.py --data weather --data_path weather.csv --model FreDEA --model_id weather_96_336 --seq_len 96 --pred_len 336 --enc_in 21 --dec_in 21 --c_out 21 --d_model 64 --d_ff 128 --e_layers 2 --memory_size 128 --bottleneck_dim 4 --dropout 0.1 --batch_size 64 --learning_rate 0.0005 --lradj 3 --train_epochs 30 --patience 5 --itr 1 --num_workers 4

Args in experiment:
Namespace(is_training=1, train_only=False, model_id='weather_96_336', model='FreDEA', data='weather', root_path='./dataset/', data_path='weather.csv', channel_independence=0, features='M', target='OT', freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=336, individual=False, embed_type=0, enc_in=21, dec_in=21, c_out=21, d_model=64, reg_lambda=0.01, wavelet='haar', n_bins=8, n_heads=8, e_layers=2, d_layers=1, d_ff=128, moving_avg=25, factor=1, distil=True, rev_affine=1, memory_size=128, bottleneck_dim=4, ablation_freq=0, ablation_tea=0, ablation_cea=0, dropout=0.1, embed='timeF', activation='gelu', output_attention=False, do_predict=False, num_workers=4, itr=1, train_epochs=30, batch_size=64, patience=5, learning_rate=0.0005, des='Exp', loss='mse', lradj='3', use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=False, devices='0,1,2,3', test_flop=False)
Use GPU: cuda:0
>>>>>>>start training : weather_96_336_FreDEA_weather_ftM_sl96_ll48_pl336_dm64_nh8_el2_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0>>>>>>>>>>>>>>>>>>>>>>>>>>
train 36456
val 4935
test 10204

============================================================
TRAINING CONFIGURATION
============================================================
Model:           FreDEA
Dataset:         weather
Seq/Pred Len:    96 -> 336
Batch Size:      64
Learning Rate:   0.0005
Train Epochs:    30
Total Params:    131,264
d_model:         64
n_heads:         8
e_layers:        2
============================================================

[Epoch 01] Iter  100/569 | loss=0.69163 | 0.027s/iter | ETA: 7.6min
[Epoch 01] Iter  200/569 | loss=0.65732 | 0.016s/iter | ETA: 4.5min
[Epoch 01] Iter  300/569 | loss=0.71301 | 0.018s/iter | ETA: 4.9min
[Epoch 01] Iter  400/569 | loss=0.69857 | 0.019s/iter | ETA: 5.3min
[Epoch 01] Iter  500/569 | loss=0.79513 | 0.019s/iter | ETA: 5.3min

------------------------------------------------------------
[Epoch 01] Summary | Time: 10.6s
Train Loss: 0.649940
Vali  Loss: 0.578272
Test  Loss: 0.280233
Validation loss decreased (inf --> 0.578272).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 02] Iter  100/569 | loss=0.59136 | 0.055s/iter | ETA: 14.9min
[Epoch 02] Iter  200/569 | loss=0.63010 | 0.019s/iter | ETA: 5.1min
[Epoch 02] Iter  300/569 | loss=0.76320 | 0.017s/iter | ETA: 4.6min
[Epoch 02] Iter  400/569 | loss=0.54565 | 0.017s/iter | ETA: 4.6min
[Epoch 02] Iter  500/569 | loss=0.58468 | 0.020s/iter | ETA: 5.3min

------------------------------------------------------------
[Epoch 02] Summary | Time: 10.6s
Train Loss: 0.576650
Vali  Loss: 0.560041
Test  Loss: 0.269967
Validation loss decreased (0.578272 --> 0.560041).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 03] Iter  100/569 | loss=0.51721 | 0.059s/iter | ETA: 15.5min
[Epoch 03] Iter  200/569 | loss=0.69909 | 0.021s/iter | ETA: 5.4min
[Epoch 03] Iter  300/569 | loss=0.61276 | 0.020s/iter | ETA: 5.3min
[Epoch 03] Iter  400/569 | loss=0.48488 | 0.020s/iter | ETA: 5.1min
[Epoch 03] Iter  500/569 | loss=0.56763 | 0.019s/iter | ETA: 4.9min

------------------------------------------------------------
[Epoch 03] Summary | Time: 11.2s
Train Loss: 0.557625
Vali  Loss: 0.545288
Test  Loss: 0.265336
Validation loss decreased (0.560041 --> 0.545288).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 04] Iter  100/569 | loss=0.83193 | 0.051s/iter | ETA: 13.1min
[Epoch 04] Iter  200/569 | loss=0.73947 | 0.016s/iter | ETA: 4.1min
[Epoch 04] Iter  300/569 | loss=0.43113 | 0.018s/iter | ETA: 4.6min
[Epoch 04] Iter  400/569 | loss=0.62074 | 0.016s/iter | ETA: 3.9min
[Epoch 04] Iter  500/569 | loss=0.49455 | 0.016s/iter | ETA: 3.9min

------------------------------------------------------------
[Epoch 04] Summary | Time: 9.4s
Train Loss: 0.542737
Vali  Loss: 0.540560
Test  Loss: 0.265860
Validation loss decreased (0.545288 --> 0.540560).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 05] Iter  100/569 | loss=0.45405 | 0.052s/iter | ETA: 12.7min
[Epoch 05] Iter  200/569 | loss=0.48391 | 0.019s/iter | ETA: 4.5min
[Epoch 05] Iter  300/569 | loss=0.46390 | 0.019s/iter | ETA: 4.7min
[Epoch 05] Iter  400/569 | loss=0.40974 | 0.020s/iter | ETA: 4.7min
[Epoch 05] Iter  500/569 | loss=0.52933 | 0.020s/iter | ETA: 4.8min

------------------------------------------------------------
[Epoch 05] Summary | Time: 11.1s
Train Loss: 0.531553
Vali  Loss: 0.536662
Test  Loss: 0.265881
Validation loss decreased (0.540560 --> 0.536662).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 06] Iter  100/569 | loss=0.53337 | 0.054s/iter | ETA: 12.6min
[Epoch 06] Iter  200/569 | loss=0.59595 | 0.017s/iter | ETA: 3.9min
[Epoch 06] Iter  300/569 | loss=0.45715 | 0.018s/iter | ETA: 4.1min
[Epoch 06] Iter  400/569 | loss=0.52526 | 0.019s/iter | ETA: 4.4min
[Epoch 06] Iter  500/569 | loss=0.40645 | 0.019s/iter | ETA: 4.4min

------------------------------------------------------------
[Epoch 06] Summary | Time: 10.4s
Train Loss: 0.523165
Vali  Loss: 0.535996
Test  Loss: 0.267078
Validation loss decreased (0.536662 --> 0.535996).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 07] Iter  100/569 | loss=0.57376 | 0.054s/iter | ETA: 12.2min
[Epoch 07] Iter  200/569 | loss=0.39322 | 0.015s/iter | ETA: 3.5min
[Epoch 07] Iter  300/569 | loss=0.54674 | 0.015s/iter | ETA: 3.4min
[Epoch 07] Iter  400/569 | loss=0.39679 | 0.015s/iter | ETA: 3.4min
[Epoch 07] Iter  500/569 | loss=0.55271 | 0.016s/iter | ETA: 3.6min

------------------------------------------------------------
[Epoch 07] Summary | Time: 9.0s
Train Loss: 0.515622
Vali  Loss: 0.533515
Test  Loss: 0.264412
Validation loss decreased (0.535996 --> 0.533515).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 08] Iter  100/569 | loss=0.53268 | 0.054s/iter | ETA: 11.7min
[Epoch 08] Iter  200/569 | loss=0.52882 | 0.016s/iter | ETA: 3.5min
[Epoch 08] Iter  300/569 | loss=0.56809 | 0.018s/iter | ETA: 3.8min
[Epoch 08] Iter  400/569 | loss=0.42634 | 0.019s/iter | ETA: 4.0min
[Epoch 08] Iter  500/569 | loss=0.43466 | 0.019s/iter | ETA: 4.0min

------------------------------------------------------------
[Epoch 08] Summary | Time: 10.6s
Train Loss: 0.510247
Vali  Loss: 0.536548
Test  Loss: 0.265026
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 09] Iter  100/569 | loss=0.40817 | 0.054s/iter | ETA: 11.1min
[Epoch 09] Iter  200/569 | loss=0.52971 | 0.018s/iter | ETA: 3.8min
[Epoch 09] Iter  300/569 | loss=0.58928 | 0.020s/iter | ETA: 4.0min
[Epoch 09] Iter  400/569 | loss=0.64117 | 0.020s/iter | ETA: 4.0min
[Epoch 09] Iter  500/569 | loss=0.43112 | 0.021s/iter | ETA: 4.2min

------------------------------------------------------------
[Epoch 09] Summary | Time: 11.1s
Train Loss: 0.504776
Vali  Loss: 0.539491
Test  Loss: 0.263638
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 10] Iter  100/569 | loss=0.52328 | 0.055s/iter | ETA: 10.8min
[Epoch 10] Iter  200/569 | loss=0.43799 | 0.015s/iter | ETA: 3.0min
[Epoch 10] Iter  300/569 | loss=0.58680 | 0.013s/iter | ETA: 2.5min
[Epoch 10] Iter  400/569 | loss=0.38797 | 0.016s/iter | ETA: 3.0min
[Epoch 10] Iter  500/569 | loss=0.39145 | 0.018s/iter | ETA: 3.4min

------------------------------------------------------------
[Epoch 10] Summary | Time: 9.1s
Train Loss: 0.500270
Vali  Loss: 0.542229
Test  Loss: 0.265460
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 5e-05
[Epoch 11] Iter  100/569 | loss=0.40570 | 0.054s/iter | ETA: 10.2min
[Epoch 11] Iter  200/569 | loss=0.57459 | 0.019s/iter | ETA: 3.5min
[Epoch 11] Iter  300/569 | loss=0.74760 | 0.019s/iter | ETA: 3.5min
[Epoch 11] Iter  400/569 | loss=0.75971 | 0.019s/iter | ETA: 3.5min
[Epoch 11] Iter  500/569 | loss=0.88847 | 0.019s/iter | ETA: 3.5min

------------------------------------------------------------
[Epoch 11] Summary | Time: 10.9s
Train Loss: 0.494427
Vali  Loss: 0.542000
Test  Loss: 0.264962
EarlyStopping counter: 4 out of 5
------------------------------------------------------------
Updating learning rate to 5e-05
[Epoch 12] Iter  100/569 | loss=0.54162 | 0.056s/iter | ETA: 9.9min
[Epoch 12] Iter  200/569 | loss=0.74612 | 0.019s/iter | ETA: 3.3min
[Epoch 12] Iter  300/569 | loss=0.52647 | 0.019s/iter | ETA: 3.3min
[Epoch 12] Iter  400/569 | loss=0.57472 | 0.019s/iter | ETA: 3.3min
[Epoch 12] Iter  500/569 | loss=0.42670 | 0.019s/iter | ETA: 3.2min

------------------------------------------------------------
[Epoch 12] Summary | Time: 10.7s
Train Loss: 0.493516
Vali  Loss: 0.541884
Test  Loss: 0.264505
EarlyStopping counter: 5 out of 5
------------------------------------------------------------

************************************************************
EARLY STOPPING TRIGGERED
************************************************************
>>>>>>>testing : weather_96_336_FreDEA_weather_ftM_sl96_ll48_pl336_dm64_nh8_el2_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 10204
mse:0.2649058401584625, mae:0.29191967844963074, rmse:0.5146900415420532
✅ 实验完成: MSE=0.264906, MAE=0.291920

================================================================================
运行实验: weather - Pred 720
================================================================================
命令: python -u run_longExp.py --data weather --data_path weather.csv --model FreDEA --model_id weather_96_720 --seq_len 96 --pred_len 720 --enc_in 21 --dec_in 21 --c_out 21 --d_model 128 --d_ff 256 --e_layers 2 --memory_size 128 --bottleneck_dim 4 --dropout 0.1 --batch_size 64 --learning_rate 0.0005 --lradj 3 --train_epochs 30 --patience 5 --itr 1 --num_workers 4

Args in experiment:
Namespace(is_training=1, train_only=False, model_id='weather_96_720', model='FreDEA', data='weather', root_path='./dataset/', data_path='weather.csv', channel_independence=0, features='M', target='OT', freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=720, individual=False, embed_type=0, enc_in=21, dec_in=21, c_out=21, d_model=128, reg_lambda=0.01, wavelet='haar', n_bins=8, n_heads=8, e_layers=2, d_layers=1, d_ff=256, moving_avg=25, factor=1, distil=True, rev_affine=1, memory_size=128, bottleneck_dim=4, ablation_freq=0, ablation_tea=0, ablation_cea=0, dropout=0.1, embed='timeF', activation='gelu', output_attention=False, do_predict=False, num_workers=4, itr=1, train_epochs=30, batch_size=64, patience=5, learning_rate=0.0005, des='Exp', loss='mse', lradj='3', use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=False, devices='0,1,2,3', test_flop=False)
Use GPU: cuda:0
>>>>>>>start training : weather_96_720_FreDEA_weather_ftM_sl96_ll48_pl720_dm128_nh8_el2_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0>>>>>>>>>>>>>>>>>>>>>>>>>>
train 36072
val 4551
test 9820

============================================================
TRAINING CONFIGURATION
============================================================
Model:           FreDEA
Dataset:         weather
Seq/Pred Len:    96 -> 720
Batch Size:      64
Learning Rate:   0.0005
Train Epochs:    30
Total Params:    320,512
d_model:         128
n_heads:         8
e_layers:        2
============================================================

[Epoch 01] Iter  100/563 | loss=0.73213 | 0.039s/iter | ETA: 10.9min
[Epoch 01] Iter  200/563 | loss=0.78907 | 0.028s/iter | ETA: 7.8min
[Epoch 01] Iter  300/563 | loss=0.64717 | 0.028s/iter | ETA: 7.7min
[Epoch 01] Iter  400/563 | loss=0.63064 | 0.028s/iter | ETA: 7.8min
[Epoch 01] Iter  500/563 | loss=0.69844 | 0.028s/iter | ETA: 7.8min

------------------------------------------------------------
[Epoch 01] Summary | Time: 16.4s
Train Loss: 0.717218
Vali  Loss: 0.704644
Test  Loss: 0.353351
Validation loss decreased (inf --> 0.704644).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 02] Iter  100/563 | loss=0.72038 | 0.082s/iter | ETA: 22.1min
[Epoch 02] Iter  200/563 | loss=0.64807 | 0.028s/iter | ETA: 7.6min
[Epoch 02] Iter  300/563 | loss=0.57760 | 0.028s/iter | ETA: 7.6min
[Epoch 02] Iter  400/563 | loss=0.78244 | 0.029s/iter | ETA: 7.6min
[Epoch 02] Iter  500/563 | loss=0.63707 | 0.028s/iter | ETA: 7.5min

------------------------------------------------------------
[Epoch 02] Summary | Time: 16.2s
Train Loss: 0.656559
Vali  Loss: 0.682672
Test  Loss: 0.347351
Validation loss decreased (0.704644 --> 0.682672).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 03] Iter  100/563 | loss=0.65111 | 0.082s/iter | ETA: 21.4min
[Epoch 03] Iter  200/563 | loss=0.51856 | 0.028s/iter | ETA: 7.4min
[Epoch 03] Iter  300/563 | loss=0.60739 | 0.028s/iter | ETA: 7.3min
[Epoch 03] Iter  400/563 | loss=0.62651 | 0.029s/iter | ETA: 7.3min
[Epoch 03] Iter  500/563 | loss=0.63888 | 0.029s/iter | ETA: 7.3min

------------------------------------------------------------
[Epoch 03] Summary | Time: 16.3s
Train Loss: 0.636538
Vali  Loss: 0.679834
Test  Loss: 0.345101
Validation loss decreased (0.682672 --> 0.679834).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 04] Iter  100/563 | loss=0.75362 | 0.083s/iter | ETA: 20.9min
[Epoch 04] Iter  200/563 | loss=0.61702 | 0.028s/iter | ETA: 7.1min
[Epoch 04] Iter  300/563 | loss=0.59581 | 0.028s/iter | ETA: 7.0min
[Epoch 04] Iter  400/563 | loss=0.72128 | 0.028s/iter | ETA: 7.0min
[Epoch 04] Iter  500/563 | loss=0.56766 | 0.028s/iter | ETA: 6.9min

------------------------------------------------------------
[Epoch 04] Summary | Time: 16.1s
Train Loss: 0.617518
Vali  Loss: 0.674355
Test  Loss: 0.348518
Validation loss decreased (0.679834 --> 0.674355).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 05] Iter  100/563 | loss=0.63036 | 0.081s/iter | ETA: 19.6min
[Epoch 05] Iter  200/563 | loss=0.72262 | 0.027s/iter | ETA: 6.6min
[Epoch 05] Iter  300/563 | loss=0.55591 | 0.027s/iter | ETA: 6.6min
[Epoch 05] Iter  400/563 | loss=0.56877 | 0.028s/iter | ETA: 6.6min
[Epoch 05] Iter  500/563 | loss=0.69482 | 0.028s/iter | ETA: 6.7min

------------------------------------------------------------
[Epoch 05] Summary | Time: 15.8s
Train Loss: 0.603447
Vali  Loss: 0.670275
Test  Loss: 0.347902
Validation loss decreased (0.674355 --> 0.670275).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 06] Iter  100/563 | loss=0.50828 | 0.080s/iter | ETA: 18.5min
[Epoch 06] Iter  200/563 | loss=0.59022 | 0.027s/iter | ETA: 6.3min
[Epoch 06] Iter  300/563 | loss=0.51058 | 0.028s/iter | ETA: 6.4min
[Epoch 06] Iter  400/563 | loss=0.64012 | 0.028s/iter | ETA: 6.4min
[Epoch 06] Iter  500/563 | loss=0.58789 | 0.028s/iter | ETA: 6.4min

------------------------------------------------------------
[Epoch 06] Summary | Time: 15.8s
Train Loss: 0.591963
Vali  Loss: 0.674335
Test  Loss: 0.353121
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 07] Iter  100/563 | loss=0.65604 | 0.081s/iter | ETA: 18.0min
[Epoch 07] Iter  200/563 | loss=0.70946 | 0.027s/iter | ETA: 6.1min
[Epoch 07] Iter  300/563 | loss=0.70004 | 0.027s/iter | ETA: 6.0min
[Epoch 07] Iter  400/563 | loss=0.55926 | 0.027s/iter | ETA: 6.0min
[Epoch 07] Iter  500/563 | loss=0.60049 | 0.027s/iter | ETA: 6.0min

------------------------------------------------------------
[Epoch 07] Summary | Time: 15.6s
Train Loss: 0.582283
Vali  Loss: 0.674302
Test  Loss: 0.350678
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 08] Iter  100/563 | loss=0.51068 | 0.080s/iter | ETA: 17.2min
[Epoch 08] Iter  200/563 | loss=0.56193 | 0.027s/iter | ETA: 5.8min
[Epoch 08] Iter  300/563 | loss=0.50794 | 0.027s/iter | ETA: 5.8min
[Epoch 08] Iter  400/563 | loss=0.47414 | 0.027s/iter | ETA: 5.7min
[Epoch 08] Iter  500/563 | loss=0.66711 | 0.028s/iter | ETA: 5.8min

------------------------------------------------------------
[Epoch 08] Summary | Time: 15.7s
Train Loss: 0.574652
Vali  Loss: 0.671813
Test  Loss: 0.353413
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 09] Iter  100/563 | loss=0.66331 | 0.081s/iter | ETA: 16.6min
[Epoch 09] Iter  200/563 | loss=0.49869 | 0.028s/iter | ETA: 5.7min
[Epoch 09] Iter  300/563 | loss=0.59221 | 0.028s/iter | ETA: 5.7min
[Epoch 09] Iter  400/563 | loss=0.53416 | 0.028s/iter | ETA: 5.7min
[Epoch 09] Iter  500/563 | loss=0.44281 | 0.028s/iter | ETA: 5.6min

------------------------------------------------------------
[Epoch 09] Summary | Time: 16.1s
Train Loss: 0.566931
Vali  Loss: 0.678449
Test  Loss: 0.360040
EarlyStopping counter: 4 out of 5
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 10] Iter  100/563 | loss=0.58228 | 0.081s/iter | ETA: 15.9min
[Epoch 10] Iter  200/563 | loss=0.58823 | 0.029s/iter | ETA: 5.5min
[Epoch 10] Iter  300/563 | loss=0.58618 | 0.028s/iter | ETA: 5.5min
[Epoch 10] Iter  400/563 | loss=0.55796 | 0.029s/iter | ETA: 5.5min
[Epoch 10] Iter  500/563 | loss=0.51902 | 0.029s/iter | ETA: 5.4min

------------------------------------------------------------
[Epoch 10] Summary | Time: 16.2s
Train Loss: 0.560392
Vali  Loss: 0.679798
Test  Loss: 0.354064
EarlyStopping counter: 5 out of 5
------------------------------------------------------------

************************************************************
EARLY STOPPING TRIGGERED
************************************************************
>>>>>>>testing : weather_96_720_FreDEA_weather_ftM_sl96_ll48_pl720_dm128_nh8_el2_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 9820
mse:0.347055584192276, mae:0.344896525144577, rmse:0.589114248752594
✅ 实验完成: MSE=0.347056, MAE=0.344897

================================================================================
运行实验: electricity - Pred 96
================================================================================
命令: python -u run_longExp.py --data electricity --data_path electricity.csv --model FreDEA --model_id electricity_96_96 --seq_len 96 --pred_len 96 --enc_in 321 --dec_in 321 --c_out 321 --d_model 64 --d_ff 128 --e_layers 2 --memory_size 64 --bottleneck_dim 2 --dropout 0.05 --batch_size 16 --learning_rate 0.0005 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4

Args in experiment:
Namespace(is_training=1, train_only=False, model_id='electricity_96_96', model='FreDEA', data='electricity', root_path='./dataset/', data_path='electricity.csv', channel_independence=0, features='M', target='OT', freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=96, individual=False, embed_type=0, enc_in=321, dec_in=321, c_out=321, d_model=64, reg_lambda=0.01, wavelet='haar', n_bins=8, n_heads=8, e_layers=2, d_layers=1, d_ff=128, moving_avg=25, factor=1, distil=True, rev_affine=1, memory_size=64, bottleneck_dim=2, ablation_freq=0, ablation_tea=0, ablation_cea=0, dropout=0.05, embed='timeF', activation='gelu', output_attention=False, do_predict=False, num_workers=4, itr=1, train_epochs=20, batch_size=16, patience=5, learning_rate=0.0005, des='Exp', loss='mse', lradj='type1', use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=False, devices='0,1,2,3', test_flop=False)
Use GPU: cuda:0
>>>>>>>start training : electricity_96_96_FreDEA_electricity_ftM_sl96_ll48_pl96_dm64_nh8_el2_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0>>>>>>>>>>>>>>>>>>>>>>>>>>
train 18221
val 2537
test 5165

============================================================
TRAINING CONFIGURATION
============================================================
Model:           FreDEA
Dataset:         electricity
Seq/Pred Len:    96 -> 96
Batch Size:      16
Learning Rate:   0.0005
Train Epochs:    20
Total Params:    163,112
d_model:         64
n_heads:         8
e_layers:        2
============================================================

[Epoch 01] Iter  100/1138 | loss=0.41619 | 0.048s/iter | ETA: 18.2min
[Epoch 01] Iter  200/1138 | loss=0.30643 | 0.037s/iter | ETA: 13.9min
[Epoch 01] Iter  300/1138 | loss=0.25617 | 0.037s/iter | ETA: 13.7min
[Epoch 01] Iter  400/1138 | loss=0.25776 | 0.036s/iter | ETA: 13.6min
[Epoch 01] Iter  500/1138 | loss=0.24071 | 0.037s/iter | ETA: 13.6min
[Epoch 01] Iter  600/1138 | loss=0.22261 | 0.037s/iter | ETA: 13.5min
[Epoch 01] Iter  700/1138 | loss=0.20765 | 0.036s/iter | ETA: 13.4min
[Epoch 01] Iter  800/1138 | loss=0.21219 | 0.036s/iter | ETA: 13.4min
[Epoch 01] Iter  900/1138 | loss=0.22607 | 0.037s/iter | ETA: 13.4min
[Epoch 01] Iter 1000/1138 | loss=0.17917 | 0.037s/iter | ETA: 13.3min
[Epoch 01] Iter 1100/1138 | loss=0.22021 | 0.037s/iter | ETA: 13.3min

------------------------------------------------------------
[Epoch 01] Summary | Time: 42.3s
Train Loss: 0.287038
Vali  Loss: 0.167361
Test  Loss: 0.190051
Validation loss decreased (inf --> 0.167361).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 02] Iter  100/1138 | loss=0.20236 | 0.136s/iter | ETA: 48.7min
[Epoch 02] Iter  200/1138 | loss=0.17060 | 0.037s/iter | ETA: 13.2min
[Epoch 02] Iter  300/1138 | loss=0.20364 | 0.037s/iter | ETA: 13.1min
[Epoch 02] Iter  400/1138 | loss=0.18693 | 0.037s/iter | ETA: 13.1min
[Epoch 02] Iter  500/1138 | loss=0.19172 | 0.037s/iter | ETA: 13.0min
[Epoch 02] Iter  600/1138 | loss=0.18319 | 0.037s/iter | ETA: 13.0min
[Epoch 02] Iter  700/1138 | loss=0.15812 | 0.037s/iter | ETA: 12.8min
[Epoch 02] Iter  800/1138 | loss=0.19012 | 0.037s/iter | ETA: 12.7min
[Epoch 02] Iter  900/1138 | loss=0.18113 | 0.037s/iter | ETA: 12.7min
[Epoch 02] Iter 1000/1138 | loss=0.17307 | 0.037s/iter | ETA: 12.7min
[Epoch 02] Iter 1100/1138 | loss=0.18883 | 0.037s/iter | ETA: 12.7min

------------------------------------------------------------
[Epoch 02] Summary | Time: 42.1s
Train Loss: 0.186700
Vali  Loss: 0.149225
Test  Loss: 0.170853
Validation loss decreased (0.167361 --> 0.149225).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.00025
[Epoch 03] Iter  100/1138 | loss=0.18016 | 0.136s/iter | ETA: 46.2min
[Epoch 03] Iter  200/1138 | loss=0.16872 | 0.036s/iter | ETA: 12.3min
[Epoch 03] Iter  300/1138 | loss=0.19531 | 0.036s/iter | ETA: 12.3min
[Epoch 03] Iter  400/1138 | loss=0.18617 | 0.036s/iter | ETA: 12.2min
[Epoch 03] Iter  500/1138 | loss=0.16598 | 0.036s/iter | ETA: 12.2min
[Epoch 03] Iter  600/1138 | loss=0.18570 | 0.036s/iter | ETA: 12.1min
[Epoch 03] Iter  700/1138 | loss=0.16250 | 0.036s/iter | ETA: 12.0min
[Epoch 03] Iter  800/1138 | loss=0.18555 | 0.036s/iter | ETA: 12.0min
[Epoch 03] Iter  900/1138 | loss=0.18234 | 0.036s/iter | ETA: 11.9min
[Epoch 03] Iter 1000/1138 | loss=0.15831 | 0.036s/iter | ETA: 11.8min
[Epoch 03] Iter 1100/1138 | loss=0.14870 | 0.037s/iter | ETA: 11.9min

------------------------------------------------------------
[Epoch 03] Summary | Time: 41.7s
Train Loss: 0.172255
Vali  Loss: 0.144414
Test  Loss: 0.165243
Validation loss decreased (0.149225 --> 0.144414).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.000125
[Epoch 04] Iter  100/1138 | loss=0.17149 | 0.138s/iter | ETA: 44.1min
[Epoch 04] Iter  200/1138 | loss=0.16768 | 0.037s/iter | ETA: 11.8min
[Epoch 04] Iter  300/1138 | loss=0.16964 | 0.037s/iter | ETA: 11.8min
[Epoch 04] Iter  400/1138 | loss=0.20326 | 0.037s/iter | ETA: 11.7min
[Epoch 04] Iter  500/1138 | loss=0.16026 | 0.037s/iter | ETA: 11.6min
[Epoch 04] Iter  600/1138 | loss=0.16978 | 0.037s/iter | ETA: 11.6min
[Epoch 04] Iter  700/1138 | loss=0.16188 | 0.037s/iter | ETA: 11.5min
[Epoch 04] Iter  800/1138 | loss=0.18176 | 0.037s/iter | ETA: 11.4min
[Epoch 04] Iter  900/1138 | loss=0.16583 | 0.037s/iter | ETA: 11.4min
[Epoch 04] Iter 1000/1138 | loss=0.16180 | 0.037s/iter | ETA: 11.3min
[Epoch 04] Iter 1100/1138 | loss=0.16116 | 0.037s/iter | ETA: 11.3min

------------------------------------------------------------
[Epoch 04] Summary | Time: 42.3s
Train Loss: 0.167453
Vali  Loss: 0.143192
Test  Loss: 0.163764
Validation loss decreased (0.144414 --> 0.143192).  Saving model ...
------------------------------------------------------------
Updating learning rate to 6.25e-05
[Epoch 05] Iter  100/1138 | loss=0.17126 | 0.137s/iter | ETA: 41.4min
[Epoch 05] Iter  200/1138 | loss=0.14889 | 0.037s/iter | ETA: 11.1min
[Epoch 05] Iter  300/1138 | loss=0.15444 | 0.037s/iter | ETA: 11.0min
[Epoch 05] Iter  400/1138 | loss=0.14181 | 0.037s/iter | ETA: 11.0min
[Epoch 05] Iter  500/1138 | loss=0.17155 | 0.037s/iter | ETA: 10.9min
[Epoch 05] Iter  600/1138 | loss=0.15561 | 0.037s/iter | ETA: 10.7min
[Epoch 05] Iter  700/1138 | loss=0.20686 | 0.036s/iter | ETA: 10.6min
[Epoch 05] Iter  800/1138 | loss=0.14730 | 0.036s/iter | ETA: 10.6min
[Epoch 05] Iter  900/1138 | loss=0.14022 | 0.036s/iter | ETA: 10.5min
[Epoch 05] Iter 1000/1138 | loss=0.16543 | 0.037s/iter | ETA: 10.6min
[Epoch 05] Iter 1100/1138 | loss=0.17054 | 0.037s/iter | ETA: 10.5min

------------------------------------------------------------
[Epoch 05] Summary | Time: 42.0s
Train Loss: 0.165215
Vali  Loss: 0.142197
Test  Loss: 0.163038
Validation loss decreased (0.143192 --> 0.142197).  Saving model ...
------------------------------------------------------------
Updating learning rate to 3.125e-05
[Epoch 06] Iter  100/1138 | loss=0.14948 | 0.136s/iter | ETA: 38.4min
[Epoch 06] Iter  200/1138 | loss=0.17862 | 0.036s/iter | ETA: 10.2min
[Epoch 06] Iter  300/1138 | loss=0.17265 | 0.036s/iter | ETA: 10.2min
[Epoch 06] Iter  400/1138 | loss=0.17329 | 0.036s/iter | ETA: 10.1min
[Epoch 06] Iter  500/1138 | loss=0.14492 | 0.036s/iter | ETA: 10.0min
[Epoch 06] Iter  600/1138 | loss=0.15653 | 0.036s/iter | ETA: 10.0min
[Epoch 06] Iter  700/1138 | loss=0.16458 | 0.037s/iter | ETA: 10.1min
[Epoch 06] Iter  800/1138 | loss=0.15331 | 0.037s/iter | ETA: 10.0min
[Epoch 06] Iter  900/1138 | loss=0.16621 | 0.037s/iter | ETA: 9.9min
[Epoch 06] Iter 1000/1138 | loss=0.14663 | 0.037s/iter | ETA: 9.9min
[Epoch 06] Iter 1100/1138 | loss=0.15324 | 0.037s/iter | ETA: 9.8min

------------------------------------------------------------
[Epoch 06] Summary | Time: 41.9s
Train Loss: 0.164134
Vali  Loss: 0.141983
Test  Loss: 0.162413
Validation loss decreased (0.142197 --> 0.141983).  Saving model ...
------------------------------------------------------------
Updating learning rate to 1.5625e-05
[Epoch 07] Iter  100/1138 | loss=0.18359 | 0.137s/iter | ETA: 36.3min
[Epoch 07] Iter  200/1138 | loss=0.15750 | 0.037s/iter | ETA: 9.7min
[Epoch 07] Iter  300/1138 | loss=0.15996 | 0.037s/iter | ETA: 9.6min
[Epoch 07] Iter  400/1138 | loss=0.18979 | 0.037s/iter | ETA: 9.5min
[Epoch 07] Iter  500/1138 | loss=0.16011 | 0.037s/iter | ETA: 9.5min
[Epoch 07] Iter  600/1138 | loss=0.19464 | 0.037s/iter | ETA: 9.5min
[Epoch 07] Iter  700/1138 | loss=0.16034 | 0.037s/iter | ETA: 9.4min
[Epoch 07] Iter  800/1138 | loss=0.17451 | 0.037s/iter | ETA: 9.3min
[Epoch 07] Iter  900/1138 | loss=0.13325 | 0.037s/iter | ETA: 9.3min
[Epoch 07] Iter 1000/1138 | loss=0.14001 | 0.037s/iter | ETA: 9.2min
[Epoch 07] Iter 1100/1138 | loss=0.15943 | 0.037s/iter | ETA: 9.1min

------------------------------------------------------------
[Epoch 07] Summary | Time: 42.2s
Train Loss: 0.163595
Vali  Loss: 0.141634
Test  Loss: 0.162085
Validation loss decreased (0.141983 --> 0.141634).  Saving model ...
------------------------------------------------------------
Updating learning rate to 7.8125e-06
[Epoch 08] Iter  100/1138 | loss=0.15348 | 0.140s/iter | ETA: 34.3min
[Epoch 08] Iter  200/1138 | loss=0.16390 | 0.037s/iter | ETA: 9.0min
[Epoch 08] Iter  300/1138 | loss=0.18217 | 0.037s/iter | ETA: 8.9min
[Epoch 08] Iter  400/1138 | loss=0.15856 | 0.037s/iter | ETA: 8.9min
[Epoch 08] Iter  500/1138 | loss=0.16639 | 0.037s/iter | ETA: 8.7min
[Epoch 08] Iter  600/1138 | loss=0.16001 | 0.037s/iter | ETA: 8.7min
[Epoch 08] Iter  700/1138 | loss=0.14326 | 0.037s/iter | ETA: 8.7min
[Epoch 08] Iter  800/1138 | loss=0.14270 | 0.037s/iter | ETA: 8.6min
[Epoch 08] Iter  900/1138 | loss=0.15751 | 0.037s/iter | ETA: 8.6min
[Epoch 08] Iter 1000/1138 | loss=0.17314 | 0.037s/iter | ETA: 8.5min
[Epoch 08] Iter 1100/1138 | loss=0.14715 | 0.037s/iter | ETA: 8.4min

------------------------------------------------------------
[Epoch 08] Summary | Time: 42.3s
Train Loss: 0.163333
Vali  Loss: 0.141572
Test  Loss: 0.162033
Validation loss decreased (0.141634 --> 0.141572).  Saving model ...
------------------------------------------------------------
Updating learning rate to 3.90625e-06
[Epoch 09] Iter  100/1138 | loss=0.15475 | 0.136s/iter | ETA: 30.8min
[Epoch 09] Iter  200/1138 | loss=0.14747 | 0.037s/iter | ETA: 8.2min
[Epoch 09] Iter  300/1138 | loss=0.16728 | 0.037s/iter | ETA: 8.1min
[Epoch 09] Iter  400/1138 | loss=0.15505 | 0.037s/iter | ETA: 8.1min
[Epoch 09] Iter  500/1138 | loss=0.17803 | 0.036s/iter | ETA: 8.0min
[Epoch 09] Iter  600/1138 | loss=0.14674 | 0.037s/iter | ETA: 7.9min
[Epoch 09] Iter  700/1138 | loss=0.16258 | 0.037s/iter | ETA: 7.9min
[Epoch 09] Iter  800/1138 | loss=0.16088 | 0.037s/iter | ETA: 7.8min
[Epoch 09] Iter  900/1138 | loss=0.14183 | 0.037s/iter | ETA: 7.8min
[Epoch 09] Iter 1000/1138 | loss=0.15040 | 0.037s/iter | ETA: 7.7min
[Epoch 09] Iter 1100/1138 | loss=0.14313 | 0.037s/iter | ETA: 7.6min

------------------------------------------------------------
[Epoch 09] Summary | Time: 41.7s
Train Loss: 0.163190
Vali  Loss: 0.141603
Test  Loss: 0.162212
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 1.953125e-06
[Epoch 10] Iter  100/1138 | loss=0.17124 | 0.136s/iter | ETA: 28.2min
[Epoch 10] Iter  200/1138 | loss=0.15539 | 0.036s/iter | ETA: 7.5min
[Epoch 10] Iter  300/1138 | loss=0.17756 | 0.036s/iter | ETA: 7.4min
[Epoch 10] Iter  400/1138 | loss=0.16703 | 0.036s/iter | ETA: 7.4min
[Epoch 10] Iter  500/1138 | loss=0.14122 | 0.036s/iter | ETA: 7.3min
[Epoch 10] Iter  600/1138 | loss=0.15511 | 0.036s/iter | ETA: 7.2min
[Epoch 10] Iter  700/1138 | loss=0.17067 | 0.037s/iter | ETA: 7.3min
[Epoch 10] Iter  800/1138 | loss=0.16505 | 0.037s/iter | ETA: 7.1min
[Epoch 10] Iter  900/1138 | loss=0.14567 | 0.036s/iter | ETA: 7.1min
[Epoch 10] Iter 1000/1138 | loss=0.16426 | 0.036s/iter | ETA: 7.0min
[Epoch 10] Iter 1100/1138 | loss=0.17591 | 0.036s/iter | ETA: 6.9min

------------------------------------------------------------
[Epoch 10] Summary | Time: 41.7s
Train Loss: 0.163135
Vali  Loss: 0.141479
Test  Loss: 0.162133
Validation loss decreased (0.141572 --> 0.141479).  Saving model ...
------------------------------------------------------------
Updating learning rate to 9.765625e-07
[Epoch 11] Iter  100/1138 | loss=0.17262 | 0.137s/iter | ETA: 25.7min
[Epoch 11] Iter  200/1138 | loss=0.15931 | 0.036s/iter | ETA: 6.8min
[Epoch 11] Iter  300/1138 | loss=0.16040 | 0.037s/iter | ETA: 6.7min
[Epoch 11] Iter  400/1138 | loss=0.17463 | 0.036s/iter | ETA: 6.7min
[Epoch 11] Iter  500/1138 | loss=0.15525 | 0.036s/iter | ETA: 6.6min
[Epoch 11] Iter  600/1138 | loss=0.16558 | 0.036s/iter | ETA: 6.5min
[Epoch 11] Iter  700/1138 | loss=0.18217 | 0.036s/iter | ETA: 6.5min
[Epoch 11] Iter  800/1138 | loss=0.16359 | 0.036s/iter | ETA: 6.4min
[Epoch 11] Iter  900/1138 | loss=0.17877 | 0.036s/iter | ETA: 6.4min
[Epoch 11] Iter 1000/1138 | loss=0.16606 | 0.036s/iter | ETA: 6.3min
[Epoch 11] Iter 1100/1138 | loss=0.14761 | 0.036s/iter | ETA: 6.2min

------------------------------------------------------------
[Epoch 11] Summary | Time: 41.6s
Train Loss: 0.163100
Vali  Loss: 0.141441
Test  Loss: 0.162079
Validation loss decreased (0.141479 --> 0.141441).  Saving model ...
------------------------------------------------------------
Updating learning rate to 4.8828125e-07
[Epoch 12] Iter  100/1138 | loss=0.16330 | 0.136s/iter | ETA: 22.9min
[Epoch 12] Iter  200/1138 | loss=0.16002 | 0.036s/iter | ETA: 6.1min
[Epoch 12] Iter  300/1138 | loss=0.16149 | 0.036s/iter | ETA: 6.0min
[Epoch 12] Iter  400/1138 | loss=0.19589 | 0.036s/iter | ETA: 6.0min
[Epoch 12] Iter  500/1138 | loss=0.16495 | 0.036s/iter | ETA: 5.9min
[Epoch 12] Iter  600/1138 | loss=0.16785 | 0.036s/iter | ETA: 5.9min
[Epoch 12] Iter  700/1138 | loss=0.18398 | 0.037s/iter | ETA: 5.8min
[Epoch 12] Iter  800/1138 | loss=0.15760 | 0.037s/iter | ETA: 5.8min
[Epoch 12] Iter  900/1138 | loss=0.16709 | 0.037s/iter | ETA: 5.7min
[Epoch 12] Iter 1000/1138 | loss=0.13505 | 0.037s/iter | ETA: 5.7min
[Epoch 12] Iter 1100/1138 | loss=0.13760 | 0.037s/iter | ETA: 5.6min

------------------------------------------------------------
[Epoch 12] Summary | Time: 41.9s
Train Loss: 0.163058
Vali  Loss: 0.141472
Test  Loss: 0.162043
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 2.44140625e-07
[Epoch 13] Iter  100/1138 | loss=0.16499 | 0.136s/iter | ETA: 20.5min
[Epoch 13] Iter  200/1138 | loss=0.15671 | 0.036s/iter | ETA: 5.4min
[Epoch 13] Iter  300/1138 | loss=0.15281 | 0.036s/iter | ETA: 5.4min
[Epoch 13] Iter  400/1138 | loss=0.14110 | 0.036s/iter | ETA: 5.3min
[Epoch 13] Iter  500/1138 | loss=0.18239 | 0.036s/iter | ETA: 5.2min
[Epoch 13] Iter  600/1138 | loss=0.14512 | 0.036s/iter | ETA: 5.2min
[Epoch 13] Iter  700/1138 | loss=0.20953 | 0.036s/iter | ETA: 5.1min
[Epoch 13] Iter  800/1138 | loss=0.13950 | 0.036s/iter | ETA: 5.0min
[Epoch 13] Iter  900/1138 | loss=0.14792 | 0.036s/iter | ETA: 5.0min
[Epoch 13] Iter 1000/1138 | loss=0.15569 | 0.037s/iter | ETA: 4.9min
[Epoch 13] Iter 1100/1138 | loss=0.17064 | 0.036s/iter | ETA: 4.9min

------------------------------------------------------------
[Epoch 13] Summary | Time: 41.7s
Train Loss: 0.163034
Vali  Loss: 0.141489
Test  Loss: 0.162027
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 1.220703125e-07
[Epoch 14] Iter  100/1138 | loss=0.17190 | 0.135s/iter | ETA: 17.7min
[Epoch 14] Iter  200/1138 | loss=0.18077 | 0.036s/iter | ETA: 4.7min
[Epoch 14] Iter  300/1138 | loss=0.14648 | 0.037s/iter | ETA: 4.7min
[Epoch 14] Iter  400/1138 | loss=0.17480 | 0.037s/iter | ETA: 4.7min
[Epoch 14] Iter  500/1138 | loss=0.16662 | 0.037s/iter | ETA: 4.6min
[Epoch 14] Iter  600/1138 | loss=0.14911 | 0.037s/iter | ETA: 4.5min
[Epoch 14] Iter  700/1138 | loss=0.15723 | 0.037s/iter | ETA: 4.4min
[Epoch 14] Iter  800/1138 | loss=0.15723 | 0.036s/iter | ETA: 4.3min
[Epoch 14] Iter  900/1138 | loss=0.15421 | 0.036s/iter | ETA: 4.3min
[Epoch 14] Iter 1000/1138 | loss=0.17456 | 0.036s/iter | ETA: 4.2min
[Epoch 14] Iter 1100/1138 | loss=0.15418 | 0.036s/iter | ETA: 4.2min

------------------------------------------------------------
[Epoch 14] Summary | Time: 41.8s
Train Loss: 0.163044
Vali  Loss: 0.141455
Test  Loss: 0.162009
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 6.103515625e-08
[Epoch 15] Iter  100/1138 | loss=0.19588 | 0.137s/iter | ETA: 15.3min
[Epoch 15] Iter  200/1138 | loss=0.16585 | 0.037s/iter | ETA: 4.1min
[Epoch 15] Iter  300/1138 | loss=0.18182 | 0.037s/iter | ETA: 4.0min
[Epoch 15] Iter  400/1138 | loss=0.15012 | 0.037s/iter | ETA: 4.0min
[Epoch 15] Iter  500/1138 | loss=0.16900 | 0.037s/iter | ETA: 3.9min
[Epoch 15] Iter  600/1138 | loss=0.16860 | 0.037s/iter | ETA: 3.8min
[Epoch 15] Iter  700/1138 | loss=0.13884 | 0.037s/iter | ETA: 3.8min
[Epoch 15] Iter  800/1138 | loss=0.16122 | 0.037s/iter | ETA: 3.7min
[Epoch 15] Iter  900/1138 | loss=0.15530 | 0.037s/iter | ETA: 3.6min
[Epoch 15] Iter 1000/1138 | loss=0.20796 | 0.037s/iter | ETA: 3.6min
[Epoch 15] Iter 1100/1138 | loss=0.19422 | 0.037s/iter | ETA: 3.5min

------------------------------------------------------------
[Epoch 15] Summary | Time: 42.1s
Train Loss: 0.163030
Vali  Loss: 0.141424
Test  Loss: 0.162026
Validation loss decreased (0.141441 --> 0.141424).  Saving model ...
------------------------------------------------------------
Updating learning rate to 3.0517578125e-08
[Epoch 16] Iter  100/1138 | loss=0.13781 | 0.140s/iter | ETA: 13.1min
[Epoch 16] Iter  200/1138 | loss=0.18444 | 0.037s/iter | ETA: 3.4min
[Epoch 16] Iter  300/1138 | loss=0.17516 | 0.037s/iter | ETA: 3.3min
[Epoch 16] Iter  400/1138 | loss=0.15952 | 0.036s/iter | ETA: 3.2min
[Epoch 16] Iter  500/1138 | loss=0.17712 | 0.036s/iter | ETA: 3.2min
[Epoch 16] Iter  600/1138 | loss=0.14092 | 0.036s/iter | ETA: 3.1min
[Epoch 16] Iter  700/1138 | loss=0.15130 | 0.036s/iter | ETA: 3.0min
[Epoch 16] Iter  800/1138 | loss=0.15128 | 0.036s/iter | ETA: 3.0min
[Epoch 16] Iter  900/1138 | loss=0.14696 | 0.036s/iter | ETA: 2.9min
[Epoch 16] Iter 1000/1138 | loss=0.15109 | 0.036s/iter | ETA: 2.8min
[Epoch 16] Iter 1100/1138 | loss=0.17556 | 0.036s/iter | ETA: 2.8min

------------------------------------------------------------
[Epoch 16] Summary | Time: 41.8s
Train Loss: 0.163041
Vali  Loss: 0.141477
Test  Loss: 0.162028
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 1.52587890625e-08
[Epoch 17] Iter  100/1138 | loss=0.14962 | 0.135s/iter | ETA: 10.1min
[Epoch 17] Iter  200/1138 | loss=0.17720 | 0.036s/iter | ETA: 2.6min
[Epoch 17] Iter  300/1138 | loss=0.19806 | 0.036s/iter | ETA: 2.6min
[Epoch 17] Iter  400/1138 | loss=0.17466 | 0.036s/iter | ETA: 2.5min
[Epoch 17] Iter  500/1138 | loss=0.14584 | 0.036s/iter | ETA: 2.5min
[Epoch 17] Iter  600/1138 | loss=0.18351 | 0.036s/iter | ETA: 2.4min
[Epoch 17] Iter  700/1138 | loss=0.15691 | 0.037s/iter | ETA: 2.3min
[Epoch 17] Iter  800/1138 | loss=0.15405 | 0.036s/iter | ETA: 2.3min
[Epoch 17] Iter  900/1138 | loss=0.14251 | 0.036s/iter | ETA: 2.2min
[Epoch 17] Iter 1000/1138 | loss=0.15072 | 0.037s/iter | ETA: 2.2min
[Epoch 17] Iter 1100/1138 | loss=0.16982 | 0.037s/iter | ETA: 2.1min

------------------------------------------------------------
[Epoch 17] Summary | Time: 41.7s
Train Loss: 0.163053
Vali  Loss: 0.141456
Test  Loss: 0.162028
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 7.62939453125e-09
[Epoch 18] Iter  100/1138 | loss=0.14108 | 0.140s/iter | ETA: 7.7min
[Epoch 18] Iter  200/1138 | loss=0.16918 | 0.037s/iter | ETA: 2.0min
[Epoch 18] Iter  300/1138 | loss=0.14204 | 0.037s/iter | ETA: 1.9min
[Epoch 18] Iter  400/1138 | loss=0.17269 | 0.037s/iter | ETA: 1.8min
[Epoch 18] Iter  500/1138 | loss=0.17429 | 0.037s/iter | ETA: 1.8min
[Epoch 18] Iter  600/1138 | loss=0.14011 | 0.036s/iter | ETA: 1.7min
[Epoch 18] Iter  700/1138 | loss=0.14032 | 0.036s/iter | ETA: 1.6min
[Epoch 18] Iter  800/1138 | loss=0.15194 | 0.036s/iter | ETA: 1.6min
[Epoch 18] Iter  900/1138 | loss=0.15802 | 0.036s/iter | ETA: 1.5min
[Epoch 18] Iter 1000/1138 | loss=0.19439 | 0.036s/iter | ETA: 1.5min
[Epoch 18] Iter 1100/1138 | loss=0.17978 | 0.036s/iter | ETA: 1.4min

------------------------------------------------------------
[Epoch 18] Summary | Time: 41.9s
Train Loss: 0.163046
Vali  Loss: 0.141471
Test  Loss: 0.162028
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 3.814697265625e-09
[Epoch 19] Iter  100/1138 | loss=0.14859 | 0.138s/iter | ETA: 5.0min
[Epoch 19] Iter  200/1138 | loss=0.17204 | 0.037s/iter | ETA: 1.3min
[Epoch 19] Iter  300/1138 | loss=0.18282 | 0.037s/iter | ETA: 1.2min
[Epoch 19] Iter  400/1138 | loss=0.17360 | 0.037s/iter | ETA: 1.2min
[Epoch 19] Iter  500/1138 | loss=0.17899 | 0.037s/iter | ETA: 1.1min
[Epoch 19] Iter  600/1138 | loss=0.15607 | 0.037s/iter | ETA: 1.0min
[Epoch 19] Iter  700/1138 | loss=0.14772 | 0.037s/iter | ETA: 1.0min
[Epoch 19] Iter  800/1138 | loss=0.15821 | 0.037s/iter | ETA: 0.9min
[Epoch 19] Iter  900/1138 | loss=0.17885 | 0.037s/iter | ETA: 0.8min
[Epoch 19] Iter 1000/1138 | loss=0.17229 | 0.037s/iter | ETA: 0.8min
[Epoch 19] Iter 1100/1138 | loss=0.15570 | 0.037s/iter | ETA: 0.7min

------------------------------------------------------------
[Epoch 19] Summary | Time: 42.2s
Train Loss: 0.163046
Vali  Loss: 0.141425
Test  Loss: 0.162028
EarlyStopping counter: 4 out of 5
------------------------------------------------------------
Updating learning rate to 1.9073486328125e-09
[Epoch 20] Iter  100/1138 | loss=0.14617 | 0.137s/iter | ETA: 2.4min
[Epoch 20] Iter  200/1138 | loss=0.18158 | 0.037s/iter | ETA: 0.6min
[Epoch 20] Iter  300/1138 | loss=0.15885 | 0.037s/iter | ETA: 0.5min
[Epoch 20] Iter  400/1138 | loss=0.16002 | 0.037s/iter | ETA: 0.5min
[Epoch 20] Iter  500/1138 | loss=0.15828 | 0.037s/iter | ETA: 0.4min
[Epoch 20] Iter  600/1138 | loss=0.15155 | 0.037s/iter | ETA: 0.3min
[Epoch 20] Iter  700/1138 | loss=0.15728 | 0.036s/iter | ETA: 0.3min
[Epoch 20] Iter  800/1138 | loss=0.17861 | 0.036s/iter | ETA: 0.2min
[Epoch 20] Iter  900/1138 | loss=0.18022 | 0.036s/iter | ETA: 0.1min
[Epoch 20] Iter 1000/1138 | loss=0.15810 | 0.036s/iter | ETA: 0.1min
[Epoch 20] Iter 1100/1138 | loss=0.16444 | 0.037s/iter | ETA: 0.0min

------------------------------------------------------------
[Epoch 20] Summary | Time: 41.8s
Train Loss: 0.163026
Vali  Loss: 0.141533
Test  Loss: 0.162028
EarlyStopping counter: 5 out of 5
------------------------------------------------------------

************************************************************
EARLY STOPPING TRIGGERED
************************************************************
>>>>>>>testing : electricity_96_96_FreDEA_electricity_ftM_sl96_ll48_pl96_dm64_nh8_el2_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 5165
mse:0.1619887501001358, mae:0.2634004056453705, rmse:0.40247824788093567
✅ 实验完成: MSE=0.161989, MAE=0.263400

================================================================================
运行实验: electricity - Pred 192
================================================================================
命令: python -u run_longExp.py --data electricity --data_path electricity.csv --model FreDEA --model_id electricity_96_192 --seq_len 96 --pred_len 192 --enc_in 321 --dec_in 321 --c_out 321 --d_model 64 --d_ff 128 --e_layers 2 --memory_size 64 --bottleneck_dim 2 --dropout 0.05 --batch_size 16 --learning_rate 0.0005 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4

Args in experiment:
Namespace(is_training=1, train_only=False, model_id='electricity_96_192', model='FreDEA', data='electricity', root_path='./dataset/', data_path='electricity.csv', channel_independence=0, features='M', target='OT', freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=192, individual=False, embed_type=0, enc_in=321, dec_in=321, c_out=321, d_model=64, reg_lambda=0.01, wavelet='haar', n_bins=8, n_heads=8, e_layers=2, d_layers=1, d_ff=128, moving_avg=25, factor=1, distil=True, rev_affine=1, memory_size=64, bottleneck_dim=2, ablation_freq=0, ablation_tea=0, ablation_cea=0, dropout=0.05, embed='timeF', activation='gelu', output_attention=False, do_predict=False, num_workers=4, itr=1, train_epochs=20, batch_size=16, patience=5, learning_rate=0.0005, des='Exp', loss='mse', lradj='type1', use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=False, devices='0,1,2,3', test_flop=False)
Use GPU: cuda:0
>>>>>>>start training : electricity_96_192_FreDEA_electricity_ftM_sl96_ll48_pl192_dm64_nh8_el2_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0>>>>>>>>>>>>>>>>>>>>>>>>>>
train 18125
val 2441
test 5069

============================================================
TRAINING CONFIGURATION
============================================================
Model:           FreDEA
Dataset:         electricity
Seq/Pred Len:    96 -> 192
Batch Size:      16
Learning Rate:   0.0005
Train Epochs:    20
Total Params:    169,352
d_model:         64
n_heads:         8
e_layers:        2
============================================================

[Epoch 01] Iter  100/1132 | loss=0.42606 | 0.049s/iter | ETA: 18.5min
[Epoch 01] Iter  200/1132 | loss=0.28008 | 0.037s/iter | ETA: 13.7min
[Epoch 01] Iter  300/1132 | loss=0.26438 | 0.037s/iter | ETA: 13.7min
[Epoch 01] Iter  400/1132 | loss=0.28441 | 0.037s/iter | ETA: 13.6min
[Epoch 01] Iter  500/1132 | loss=0.22904 | 0.037s/iter | ETA: 13.6min
[Epoch 01] Iter  600/1132 | loss=0.22550 | 0.037s/iter | ETA: 13.5min
[Epoch 01] Iter  700/1132 | loss=0.23848 | 0.037s/iter | ETA: 13.4min
[Epoch 01] Iter  800/1132 | loss=0.23571 | 0.037s/iter | ETA: 13.4min
[Epoch 01] Iter  900/1132 | loss=0.22324 | 0.037s/iter | ETA: 13.3min
[Epoch 01] Iter 1000/1132 | loss=0.20796 | 0.037s/iter | ETA: 13.2min
[Epoch 01] Iter 1100/1132 | loss=0.25057 | 0.037s/iter | ETA: 13.2min

------------------------------------------------------------
[Epoch 01] Summary | Time: 42.2s
Train Loss: 0.294149
Vali  Loss: 0.177998
Test  Loss: 0.205305
Validation loss decreased (inf --> 0.177998).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 02] Iter  100/1132 | loss=0.20501 | 0.149s/iter | ETA: 53.0min
[Epoch 02] Iter  200/1132 | loss=0.22314 | 0.037s/iter | ETA: 13.0min
[Epoch 02] Iter  300/1132 | loss=0.20713 | 0.037s/iter | ETA: 12.9min
[Epoch 02] Iter  400/1132 | loss=0.17232 | 0.037s/iter | ETA: 12.9min
[Epoch 02] Iter  500/1132 | loss=0.20778 | 0.037s/iter | ETA: 12.8min
[Epoch 02] Iter  600/1132 | loss=0.19395 | 0.037s/iter | ETA: 12.8min
[Epoch 02] Iter  700/1132 | loss=0.21922 | 0.037s/iter | ETA: 12.7min
[Epoch 02] Iter  800/1132 | loss=0.17516 | 0.037s/iter | ETA: 12.7min
[Epoch 02] Iter  900/1132 | loss=0.17261 | 0.037s/iter | ETA: 12.6min
[Epoch 02] Iter 1000/1132 | loss=0.19506 | 0.037s/iter | ETA: 12.6min
[Epoch 02] Iter 1100/1132 | loss=0.19998 | 0.037s/iter | ETA: 12.5min

------------------------------------------------------------
[Epoch 02] Summary | Time: 41.6s
Train Loss: 0.202184
Vali  Loss: 0.161321
Test  Loss: 0.186992
Validation loss decreased (0.177998 --> 0.161321).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.00025
[Epoch 03] Iter  100/1132 | loss=0.19692 | 0.148s/iter | ETA: 49.9min
[Epoch 03] Iter  200/1132 | loss=0.20571 | 0.037s/iter | ETA: 12.3min
[Epoch 03] Iter  300/1132 | loss=0.17434 | 0.037s/iter | ETA: 12.4min
[Epoch 03] Iter  400/1132 | loss=0.18357 | 0.037s/iter | ETA: 12.2min
[Epoch 03] Iter  500/1132 | loss=0.18992 | 0.037s/iter | ETA: 12.1min
[Epoch 03] Iter  600/1132 | loss=0.19652 | 0.037s/iter | ETA: 12.1min
[Epoch 03] Iter  700/1132 | loss=0.20825 | 0.037s/iter | ETA: 12.1min
[Epoch 03] Iter  800/1132 | loss=0.20251 | 0.037s/iter | ETA: 12.0min
[Epoch 03] Iter  900/1132 | loss=0.18555 | 0.037s/iter | ETA: 11.9min
[Epoch 03] Iter 1000/1132 | loss=0.18554 | 0.037s/iter | ETA: 11.8min
[Epoch 03] Iter 1100/1132 | loss=0.18899 | 0.037s/iter | ETA: 11.8min

------------------------------------------------------------
[Epoch 03] Summary | Time: 41.7s
Train Loss: 0.188744
Vali  Loss: 0.158111
Test  Loss: 0.182515
Validation loss decreased (0.161321 --> 0.158111).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.000125
[Epoch 04] Iter  100/1132 | loss=0.17097 | 0.148s/iter | ETA: 47.2min
[Epoch 04] Iter  200/1132 | loss=0.16014 | 0.037s/iter | ETA: 11.6min
[Epoch 04] Iter  300/1132 | loss=0.18049 | 0.037s/iter | ETA: 11.6min
[Epoch 04] Iter  400/1132 | loss=0.17969 | 0.037s/iter | ETA: 11.5min
[Epoch 04] Iter  500/1132 | loss=0.18010 | 0.037s/iter | ETA: 11.4min
[Epoch 04] Iter  600/1132 | loss=0.18110 | 0.037s/iter | ETA: 11.4min
[Epoch 04] Iter  700/1132 | loss=0.18576 | 0.037s/iter | ETA: 11.3min
[Epoch 04] Iter  800/1132 | loss=0.16383 | 0.037s/iter | ETA: 11.3min
[Epoch 04] Iter  900/1132 | loss=0.18922 | 0.037s/iter | ETA: 11.2min
[Epoch 04] Iter 1000/1132 | loss=0.18609 | 0.037s/iter | ETA: 11.1min
[Epoch 04] Iter 1100/1132 | loss=0.19335 | 0.037s/iter | ETA: 11.1min

------------------------------------------------------------
[Epoch 04] Summary | Time: 41.7s
Train Loss: 0.184132
Vali  Loss: 0.155842
Test  Loss: 0.180864
Validation loss decreased (0.158111 --> 0.155842).  Saving model ...
------------------------------------------------------------
Updating learning rate to 6.25e-05
[Epoch 05] Iter  100/1132 | loss=0.19773 | 0.148s/iter | ETA: 44.4min
[Epoch 05] Iter  200/1132 | loss=0.18886 | 0.037s/iter | ETA: 11.1min
[Epoch 05] Iter  300/1132 | loss=0.19607 | 0.037s/iter | ETA: 11.0min
[Epoch 05] Iter  400/1132 | loss=0.17143 | 0.037s/iter | ETA: 10.8min
[Epoch 05] Iter  500/1132 | loss=0.16924 | 0.037s/iter | ETA: 10.8min
[Epoch 05] Iter  600/1132 | loss=0.17595 | 0.037s/iter | ETA: 10.7min
[Epoch 05] Iter  700/1132 | loss=0.19839 | 0.037s/iter | ETA: 10.6min
[Epoch 05] Iter  800/1132 | loss=0.17793 | 0.037s/iter | ETA: 10.6min
[Epoch 05] Iter  900/1132 | loss=0.16980 | 0.037s/iter | ETA: 10.7min
[Epoch 05] Iter 1000/1132 | loss=0.18360 | 0.037s/iter | ETA: 10.5min
[Epoch 05] Iter 1100/1132 | loss=0.17741 | 0.037s/iter | ETA: 10.5min

------------------------------------------------------------
[Epoch 05] Summary | Time: 41.9s
Train Loss: 0.181838
Vali  Loss: 0.155164
Test  Loss: 0.180705
Validation loss decreased (0.155842 --> 0.155164).  Saving model ...
------------------------------------------------------------
Updating learning rate to 3.125e-05
[Epoch 06] Iter  100/1132 | loss=0.16886 | 0.148s/iter | ETA: 41.6min
[Epoch 06] Iter  200/1132 | loss=0.19342 | 0.037s/iter | ETA: 10.3min
[Epoch 06] Iter  300/1132 | loss=0.19233 | 0.037s/iter | ETA: 10.2min
[Epoch 06] Iter  400/1132 | loss=0.19438 | 0.037s/iter | ETA: 10.1min
[Epoch 06] Iter  500/1132 | loss=0.16820 | 0.037s/iter | ETA: 10.1min
[Epoch 06] Iter  600/1132 | loss=0.18261 | 0.037s/iter | ETA: 10.0min
[Epoch 06] Iter  700/1132 | loss=0.19745 | 0.037s/iter | ETA: 10.0min
[Epoch 06] Iter  800/1132 | loss=0.17835 | 0.037s/iter | ETA: 9.9min
[Epoch 06] Iter  900/1132 | loss=0.16857 | 0.037s/iter | ETA: 9.8min
[Epoch 06] Iter 1000/1132 | loss=0.20630 | 0.037s/iter | ETA: 9.8min
[Epoch 06] Iter 1100/1132 | loss=0.18209 | 0.036s/iter | ETA: 9.6min

------------------------------------------------------------
[Epoch 06] Summary | Time: 41.7s
Train Loss: 0.180731
Vali  Loss: 0.154944
Test  Loss: 0.179973
Validation loss decreased (0.155164 --> 0.154944).  Saving model ...
------------------------------------------------------------
Updating learning rate to 1.5625e-05
[Epoch 07] Iter  100/1132 | loss=0.18013 | 0.151s/iter | ETA: 39.6min
[Epoch 07] Iter  200/1132 | loss=0.17016 | 0.037s/iter | ETA: 9.7min
[Epoch 07] Iter  300/1132 | loss=0.21795 | 0.037s/iter | ETA: 9.7min
[Epoch 07] Iter  400/1132 | loss=0.18386 | 0.037s/iter | ETA: 9.6min
[Epoch 07] Iter  500/1132 | loss=0.19376 | 0.037s/iter | ETA: 9.5min
[Epoch 07] Iter  600/1132 | loss=0.21053 | 0.037s/iter | ETA: 9.4min
[Epoch 07] Iter  700/1132 | loss=0.16549 | 0.037s/iter | ETA: 9.3min
[Epoch 07] Iter  800/1132 | loss=0.16448 | 0.037s/iter | ETA: 9.2min
[Epoch 07] Iter  900/1132 | loss=0.16824 | 0.037s/iter | ETA: 9.1min
[Epoch 07] Iter 1000/1132 | loss=0.17374 | 0.037s/iter | ETA: 9.1min
[Epoch 07] Iter 1100/1132 | loss=0.21017 | 0.037s/iter | ETA: 9.0min

------------------------------------------------------------
[Epoch 07] Summary | Time: 42.0s
Train Loss: 0.180187
Vali  Loss: 0.154839
Test  Loss: 0.180026
Validation loss decreased (0.154944 --> 0.154839).  Saving model ...
------------------------------------------------------------
Updating learning rate to 7.8125e-06
[Epoch 08] Iter  100/1132 | loss=0.19028 | 0.148s/iter | ETA: 36.1min
[Epoch 08] Iter  200/1132 | loss=0.18274 | 0.037s/iter | ETA: 8.9min
[Epoch 08] Iter  300/1132 | loss=0.19098 | 0.037s/iter | ETA: 8.8min
[Epoch 08] Iter  400/1132 | loss=0.18547 | 0.037s/iter | ETA: 8.9min
[Epoch 08] Iter  500/1132 | loss=0.18255 | 0.037s/iter | ETA: 8.9min
[Epoch 08] Iter  600/1132 | loss=0.18903 | 0.037s/iter | ETA: 8.8min
[Epoch 08] Iter  700/1132 | loss=0.17077 | 0.037s/iter | ETA: 8.7min
[Epoch 08] Iter  800/1132 | loss=0.18515 | 0.037s/iter | ETA: 8.7min
[Epoch 08] Iter  900/1132 | loss=0.18521 | 0.037s/iter | ETA: 8.6min
[Epoch 08] Iter 1000/1132 | loss=0.19633 | 0.037s/iter | ETA: 8.5min
[Epoch 08] Iter 1100/1132 | loss=0.17563 | 0.037s/iter | ETA: 8.5min

------------------------------------------------------------
[Epoch 08] Summary | Time: 42.2s
Train Loss: 0.179889
Vali  Loss: 0.154707
Test  Loss: 0.179874
Validation loss decreased (0.154839 --> 0.154707).  Saving model ...
------------------------------------------------------------
Updating learning rate to 3.90625e-06
[Epoch 09] Iter  100/1132 | loss=0.19580 | 0.149s/iter | ETA: 33.4min
[Epoch 09] Iter  200/1132 | loss=0.15813 | 0.037s/iter | ETA: 8.3min
[Epoch 09] Iter  300/1132 | loss=0.19383 | 0.037s/iter | ETA: 8.3min
[Epoch 09] Iter  400/1132 | loss=0.21665 | 0.037s/iter | ETA: 8.2min
[Epoch 09] Iter  500/1132 | loss=0.18972 | 0.037s/iter | ETA: 8.1min
[Epoch 09] Iter  600/1132 | loss=0.19152 | 0.037s/iter | ETA: 8.1min
[Epoch 09] Iter  700/1132 | loss=0.23333 | 0.037s/iter | ETA: 7.9min
[Epoch 09] Iter  800/1132 | loss=0.18754 | 0.037s/iter | ETA: 7.8min
[Epoch 09] Iter  900/1132 | loss=0.17894 | 0.037s/iter | ETA: 7.7min
[Epoch 09] Iter 1000/1132 | loss=0.19276 | 0.037s/iter | ETA: 7.7min
[Epoch 09] Iter 1100/1132 | loss=0.16639 | 0.037s/iter | ETA: 7.6min

------------------------------------------------------------
[Epoch 09] Summary | Time: 42.0s
Train Loss: 0.179779
Vali  Loss: 0.154701
Test  Loss: 0.179848
Validation loss decreased (0.154707 --> 0.154701).  Saving model ...
------------------------------------------------------------
Updating learning rate to 1.953125e-06
[Epoch 10] Iter  100/1132 | loss=0.16522 | 0.149s/iter | ETA: 30.7min
[Epoch 10] Iter  200/1132 | loss=0.16214 | 0.037s/iter | ETA: 7.6min
[Epoch 10] Iter  300/1132 | loss=0.18038 | 0.037s/iter | ETA: 7.6min
[Epoch 10] Iter  400/1132 | loss=0.21478 | 0.037s/iter | ETA: 7.5min
[Epoch 10] Iter  500/1132 | loss=0.20497 | 0.037s/iter | ETA: 7.4min
[Epoch 10] Iter  600/1132 | loss=0.17696 | 0.037s/iter | ETA: 7.2min
[Epoch 10] Iter  700/1132 | loss=0.17548 | 0.037s/iter | ETA: 7.2min
[Epoch 10] Iter  800/1132 | loss=0.20596 | 0.037s/iter | ETA: 7.2min
[Epoch 10] Iter  900/1132 | loss=0.16327 | 0.037s/iter | ETA: 7.2min
[Epoch 10] Iter 1000/1132 | loss=0.19151 | 0.037s/iter | ETA: 7.1min
[Epoch 10] Iter 1100/1132 | loss=0.18994 | 0.037s/iter | ETA: 7.0min

------------------------------------------------------------
[Epoch 10] Summary | Time: 42.2s
Train Loss: 0.179696
Vali  Loss: 0.154747
Test  Loss: 0.179849
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 9.765625e-07
[Epoch 11] Iter  100/1132 | loss=0.16766 | 0.151s/iter | ETA: 28.3min
[Epoch 11] Iter  200/1132 | loss=0.18020 | 0.037s/iter | ETA: 6.9min
[Epoch 11] Iter  300/1132 | loss=0.18079 | 0.037s/iter | ETA: 6.8min
[Epoch 11] Iter  400/1132 | loss=0.18733 | 0.037s/iter | ETA: 6.8min
[Epoch 11] Iter  500/1132 | loss=0.19881 | 0.037s/iter | ETA: 6.7min
[Epoch 11] Iter  600/1132 | loss=0.16637 | 0.037s/iter | ETA: 6.6min
[Epoch 11] Iter  700/1132 | loss=0.18083 | 0.037s/iter | ETA: 6.6min
[Epoch 11] Iter  800/1132 | loss=0.18177 | 0.037s/iter | ETA: 6.5min
[Epoch 11] Iter  900/1132 | loss=0.19574 | 0.037s/iter | ETA: 6.5min
[Epoch 11] Iter 1000/1132 | loss=0.17213 | 0.037s/iter | ETA: 6.4min
[Epoch 11] Iter 1100/1132 | loss=0.16576 | 0.037s/iter | ETA: 6.4min

------------------------------------------------------------
[Epoch 11] Summary | Time: 42.3s
Train Loss: 0.179674
Vali  Loss: 0.154647
Test  Loss: 0.179827
Validation loss decreased (0.154701 --> 0.154647).  Saving model ...
------------------------------------------------------------
Updating learning rate to 4.8828125e-07
[Epoch 12] Iter  100/1132 | loss=0.18118 | 0.151s/iter | ETA: 25.3min
[Epoch 12] Iter  200/1132 | loss=0.17768 | 0.037s/iter | ETA: 6.1min
[Epoch 12] Iter  300/1132 | loss=0.17538 | 0.037s/iter | ETA: 6.0min
[Epoch 12] Iter  400/1132 | loss=0.16266 | 0.037s/iter | ETA: 6.0min
[Epoch 12] Iter  500/1132 | loss=0.18886 | 0.037s/iter | ETA: 5.9min
[Epoch 12] Iter  600/1132 | loss=0.16970 | 0.037s/iter | ETA: 5.9min
[Epoch 12] Iter  700/1132 | loss=0.16999 | 0.037s/iter | ETA: 5.8min
[Epoch 12] Iter  800/1132 | loss=0.17625 | 0.037s/iter | ETA: 5.7min
[Epoch 12] Iter  900/1132 | loss=0.17161 | 0.037s/iter | ETA: 5.7min
[Epoch 12] Iter 1000/1132 | loss=0.18345 | 0.037s/iter | ETA: 5.6min
[Epoch 12] Iter 1100/1132 | loss=0.17242 | 0.037s/iter | ETA: 5.6min

------------------------------------------------------------
[Epoch 12] Summary | Time: 41.8s
Train Loss: 0.179654
Vali  Loss: 0.154643
Test  Loss: 0.179823
Validation loss decreased (0.154647 --> 0.154643).  Saving model ...
------------------------------------------------------------
Updating learning rate to 2.44140625e-07
[Epoch 13] Iter  100/1132 | loss=0.19053 | 0.148s/iter | ETA: 22.1min
[Epoch 13] Iter  200/1132 | loss=0.16624 | 0.037s/iter | ETA: 5.4min
[Epoch 13] Iter  300/1132 | loss=0.17662 | 0.037s/iter | ETA: 5.3min
[Epoch 13] Iter  400/1132 | loss=0.18567 | 0.037s/iter | ETA: 5.3min
[Epoch 13] Iter  500/1132 | loss=0.18439 | 0.037s/iter | ETA: 5.2min
[Epoch 13] Iter  600/1132 | loss=0.17552 | 0.037s/iter | ETA: 5.2min
[Epoch 13] Iter  700/1132 | loss=0.20685 | 0.037s/iter | ETA: 5.1min
[Epoch 13] Iter  800/1132 | loss=0.17352 | 0.037s/iter | ETA: 5.0min
[Epoch 13] Iter  900/1132 | loss=0.17110 | 0.037s/iter | ETA: 5.0min
[Epoch 13] Iter 1000/1132 | loss=0.18311 | 0.037s/iter | ETA: 4.9min
[Epoch 13] Iter 1100/1132 | loss=0.16930 | 0.037s/iter | ETA: 4.9min

------------------------------------------------------------
[Epoch 13] Summary | Time: 41.7s
Train Loss: 0.179608
Vali  Loss: 0.154682
Test  Loss: 0.179826
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 1.220703125e-07
[Epoch 14] Iter  100/1132 | loss=0.19750 | 0.149s/iter | ETA: 19.4min
[Epoch 14] Iter  200/1132 | loss=0.20519 | 0.037s/iter | ETA: 4.7min
[Epoch 14] Iter  300/1132 | loss=0.17527 | 0.037s/iter | ETA: 4.7min
[Epoch 14] Iter  400/1132 | loss=0.16581 | 0.037s/iter | ETA: 4.6min
[Epoch 14] Iter  500/1132 | loss=0.20205 | 0.037s/iter | ETA: 4.6min
[Epoch 14] Iter  600/1132 | loss=0.15707 | 0.037s/iter | ETA: 4.5min
[Epoch 14] Iter  700/1132 | loss=0.18493 | 0.037s/iter | ETA: 4.5min
[Epoch 14] Iter  800/1132 | loss=0.17518 | 0.037s/iter | ETA: 4.5min
[Epoch 14] Iter  900/1132 | loss=0.14852 | 0.037s/iter | ETA: 4.4min
[Epoch 14] Iter 1000/1132 | loss=0.18165 | 0.037s/iter | ETA: 4.3min
[Epoch 14] Iter 1100/1132 | loss=0.16654 | 0.037s/iter | ETA: 4.3min

------------------------------------------------------------
[Epoch 14] Summary | Time: 42.3s
Train Loss: 0.179620
Vali  Loss: 0.154666
Test  Loss: 0.179828
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 6.103515625e-08
[Epoch 15] Iter  100/1132 | loss=0.20531 | 0.148s/iter | ETA: 16.6min
[Epoch 15] Iter  200/1132 | loss=0.18753 | 0.037s/iter | ETA: 4.1min
[Epoch 15] Iter  300/1132 | loss=0.18999 | 0.037s/iter | ETA: 4.0min
[Epoch 15] Iter  400/1132 | loss=0.16663 | 0.037s/iter | ETA: 4.0min
[Epoch 15] Iter  500/1132 | loss=0.18286 | 0.037s/iter | ETA: 3.9min
[Epoch 15] Iter  600/1132 | loss=0.18430 | 0.037s/iter | ETA: 3.9min
[Epoch 15] Iter  700/1132 | loss=0.17962 | 0.037s/iter | ETA: 3.8min
[Epoch 15] Iter  800/1132 | loss=0.20311 | 0.037s/iter | ETA: 3.7min
[Epoch 15] Iter  900/1132 | loss=0.17574 | 0.037s/iter | ETA: 3.7min
[Epoch 15] Iter 1000/1132 | loss=0.17010 | 0.037s/iter | ETA: 3.6min
[Epoch 15] Iter 1100/1132 | loss=0.19210 | 0.037s/iter | ETA: 3.5min

------------------------------------------------------------
[Epoch 15] Summary | Time: 42.3s
Train Loss: 0.179626
Vali  Loss: 0.154586
Test  Loss: 0.179827
Validation loss decreased (0.154643 --> 0.154586).  Saving model ...
------------------------------------------------------------
Updating learning rate to 3.0517578125e-08
[Epoch 16] Iter  100/1132 | loss=0.17742 | 0.148s/iter | ETA: 13.7min
[Epoch 16] Iter  200/1132 | loss=0.16882 | 0.037s/iter | ETA: 3.4min
[Epoch 16] Iter  300/1132 | loss=0.17083 | 0.037s/iter | ETA: 3.3min
[Epoch 16] Iter  400/1132 | loss=0.18970 | 0.037s/iter | ETA: 3.2min
[Epoch 16] Iter  500/1132 | loss=0.19993 | 0.037s/iter | ETA: 3.2min
[Epoch 16] Iter  600/1132 | loss=0.18692 | 0.037s/iter | ETA: 3.1min
[Epoch 16] Iter  700/1132 | loss=0.17622 | 0.037s/iter | ETA: 3.0min
[Epoch 16] Iter  800/1132 | loss=0.19408 | 0.037s/iter | ETA: 3.0min
[Epoch 16] Iter  900/1132 | loss=0.17528 | 0.037s/iter | ETA: 2.9min
[Epoch 16] Iter 1000/1132 | loss=0.18536 | 0.037s/iter | ETA: 2.9min
[Epoch 16] Iter 1100/1132 | loss=0.16854 | 0.037s/iter | ETA: 2.8min

------------------------------------------------------------
[Epoch 16] Summary | Time: 41.8s
Train Loss: 0.179610
Vali  Loss: 0.154650
Test  Loss: 0.179827
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 1.52587890625e-08
[Epoch 17] Iter  100/1132 | loss=0.17572 | 0.148s/iter | ETA: 11.0min
[Epoch 17] Iter  200/1132 | loss=0.18710 | 0.037s/iter | ETA: 2.6min
[Epoch 17] Iter  300/1132 | loss=0.18659 | 0.037s/iter | ETA: 2.6min
[Epoch 17] Iter  400/1132 | loss=0.17906 | 0.037s/iter | ETA: 2.5min
[Epoch 17] Iter  500/1132 | loss=0.17862 | 0.037s/iter | ETA: 2.5min
[Epoch 17] Iter  600/1132 | loss=0.17255 | 0.037s/iter | ETA: 2.4min
[Epoch 17] Iter  700/1132 | loss=0.16892 | 0.037s/iter | ETA: 2.3min
[Epoch 17] Iter  800/1132 | loss=0.18288 | 0.037s/iter | ETA: 2.3min
[Epoch 17] Iter  900/1132 | loss=0.16189 | 0.037s/iter | ETA: 2.2min
[Epoch 17] Iter 1000/1132 | loss=0.13981 | 0.037s/iter | ETA: 2.2min
[Epoch 17] Iter 1100/1132 | loss=0.19211 | 0.037s/iter | ETA: 2.1min

------------------------------------------------------------
[Epoch 17] Summary | Time: 41.7s
Train Loss: 0.179612
Vali  Loss: 0.154571
Test  Loss: 0.179827
Validation loss decreased (0.154586 --> 0.154571).  Saving model ...
------------------------------------------------------------
Updating learning rate to 7.62939453125e-09
[Epoch 18] Iter  100/1132 | loss=0.16208 | 0.149s/iter | ETA: 8.2min
[Epoch 18] Iter  200/1132 | loss=0.17758 | 0.037s/iter | ETA: 2.0min
[Epoch 18] Iter  300/1132 | loss=0.16891 | 0.037s/iter | ETA: 1.9min
[Epoch 18] Iter  400/1132 | loss=0.21209 | 0.037s/iter | ETA: 1.8min
[Epoch 18] Iter  500/1132 | loss=0.18799 | 0.037s/iter | ETA: 1.8min
[Epoch 18] Iter  600/1132 | loss=0.18646 | 0.037s/iter | ETA: 1.7min
[Epoch 18] Iter  700/1132 | loss=0.19486 | 0.037s/iter | ETA: 1.6min
[Epoch 18] Iter  800/1132 | loss=0.17136 | 0.037s/iter | ETA: 1.6min
[Epoch 18] Iter  900/1132 | loss=0.17545 | 0.037s/iter | ETA: 1.5min
[Epoch 18] Iter 1000/1132 | loss=0.18766 | 0.037s/iter | ETA: 1.5min
[Epoch 18] Iter 1100/1132 | loss=0.18553 | 0.037s/iter | ETA: 1.4min

------------------------------------------------------------
[Epoch 18] Summary | Time: 41.7s
Train Loss: 0.179608
Vali  Loss: 0.154657
Test  Loss: 0.179827
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 3.814697265625e-09
[Epoch 19] Iter  100/1132 | loss=0.19410 | 0.149s/iter | ETA: 5.4min
[Epoch 19] Iter  200/1132 | loss=0.17021 | 0.037s/iter | ETA: 1.3min
[Epoch 19] Iter  300/1132 | loss=0.15491 | 0.037s/iter | ETA: 1.2min
[Epoch 19] Iter  400/1132 | loss=0.16586 | 0.037s/iter | ETA: 1.1min
[Epoch 19] Iter  500/1132 | loss=0.18409 | 0.037s/iter | ETA: 1.1min
[Epoch 19] Iter  600/1132 | loss=0.17960 | 0.037s/iter | ETA: 1.0min
[Epoch 19] Iter  700/1132 | loss=0.15121 | 0.037s/iter | ETA: 1.0min
[Epoch 19] Iter  800/1132 | loss=0.17758 | 0.037s/iter | ETA: 0.9min
[Epoch 19] Iter  900/1132 | loss=0.16845 | 0.037s/iter | ETA: 0.8min
[Epoch 19] Iter 1000/1132 | loss=0.18860 | 0.037s/iter | ETA: 0.8min
[Epoch 19] Iter 1100/1132 | loss=0.18276 | 0.037s/iter | ETA: 0.7min

------------------------------------------------------------
[Epoch 19] Summary | Time: 41.8s
Train Loss: 0.179660
Vali  Loss: 0.154672
Test  Loss: 0.179827
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 1.9073486328125e-09
[Epoch 20] Iter  100/1132 | loss=0.20099 | 0.149s/iter | ETA: 2.6min
[Epoch 20] Iter  200/1132 | loss=0.19291 | 0.037s/iter | ETA: 0.6min
[Epoch 20] Iter  300/1132 | loss=0.16533 | 0.037s/iter | ETA: 0.5min
[Epoch 20] Iter  400/1132 | loss=0.16618 | 0.037s/iter | ETA: 0.4min
[Epoch 20] Iter  500/1132 | loss=0.17275 | 0.037s/iter | ETA: 0.4min
[Epoch 20] Iter  600/1132 | loss=0.16122 | 0.037s/iter | ETA: 0.3min
[Epoch 20] Iter  700/1132 | loss=0.19262 | 0.036s/iter | ETA: 0.3min
[Epoch 20] Iter  800/1132 | loss=0.20589 | 0.037s/iter | ETA: 0.2min
[Epoch 20] Iter  900/1132 | loss=0.17345 | 0.037s/iter | ETA: 0.1min
[Epoch 20] Iter 1000/1132 | loss=0.15541 | 0.037s/iter | ETA: 0.1min
[Epoch 20] Iter 1100/1132 | loss=0.18122 | 0.037s/iter | ETA: 0.0min

------------------------------------------------------------
[Epoch 20] Summary | Time: 41.6s
Train Loss: 0.179629
Vali  Loss: 0.154707
Test  Loss: 0.179827
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 9.5367431640625e-10
>>>>>>>testing : electricity_96_192_FreDEA_electricity_ftM_sl96_ll48_pl192_dm64_nh8_el2_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 5069
mse:0.1796707808971405, mae:0.27787351608276367, rmse:0.4238758981227875
✅ 实验完成: MSE=0.179671, MAE=0.277874

================================================================================
运行实验: electricity - Pred 336
================================================================================
命令: python -u run_longExp.py --data electricity --data_path electricity.csv --model FreDEA --model_id electricity_96_336 --seq_len 96 --pred_len 336 --enc_in 321 --dec_in 321 --c_out 321 --d_model 64 --d_ff 128 --e_layers 1 --memory_size 64 --bottleneck_dim 2 --dropout 0.15 --batch_size 16 --learning_rate 0.0005 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4

Args in experiment:
Namespace(is_training=1, train_only=False, model_id='electricity_96_336', model='FreDEA', data='electricity', root_path='./dataset/', data_path='electricity.csv', channel_independence=0, features='M', target='OT', freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=336, individual=False, embed_type=0, enc_in=321, dec_in=321, c_out=321, d_model=64, reg_lambda=0.01, wavelet='haar', n_bins=8, n_heads=8, e_layers=1, d_layers=1, d_ff=128, moving_avg=25, factor=1, distil=True, rev_affine=1, memory_size=64, bottleneck_dim=2, ablation_freq=0, ablation_tea=0, ablation_cea=0, dropout=0.15, embed='timeF', activation='gelu', output_attention=False, do_predict=False, num_workers=4, itr=1, train_epochs=20, batch_size=16, patience=5, learning_rate=0.0005, des='Exp', loss='mse', lradj='type1', use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=False, devices='0,1,2,3', test_flop=False)
Use GPU: cuda:0
>>>>>>>start training : electricity_96_336_FreDEA_electricity_ftM_sl96_ll48_pl336_dm64_nh8_el1_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0>>>>>>>>>>>>>>>>>>>>>>>>>>
train 17981
val 2297
test 4925

============================================================
TRAINING CONFIGURATION
============================================================
Model:           FreDEA
Dataset:         electricity
Seq/Pred Len:    96 -> 336
Batch Size:      16
Learning Rate:   0.0005
Train Epochs:    20
Total Params:    108,376
d_model:         64
n_heads:         8
e_layers:        1
============================================================

[Epoch 01] Iter  100/1123 | loss=0.48617 | 0.036s/iter | ETA: 13.4min
[Epoch 01] Iter  200/1123 | loss=0.31469 | 0.024s/iter | ETA: 9.0min
[Epoch 01] Iter  300/1123 | loss=0.35261 | 0.025s/iter | ETA: 9.1min
[Epoch 01] Iter  400/1123 | loss=0.28596 | 0.024s/iter | ETA: 9.0min
[Epoch 01] Iter  500/1123 | loss=0.32551 | 0.024s/iter | ETA: 8.8min
[Epoch 01] Iter  600/1123 | loss=0.26835 | 0.024s/iter | ETA: 8.8min
[Epoch 01] Iter  700/1123 | loss=0.26648 | 0.024s/iter | ETA: 8.7min
[Epoch 01] Iter  800/1123 | loss=0.26042 | 0.024s/iter | ETA: 8.7min
[Epoch 01] Iter  900/1123 | loss=0.26098 | 0.024s/iter | ETA: 8.6min
[Epoch 01] Iter 1000/1123 | loss=0.26780 | 0.024s/iter | ETA: 8.6min
[Epoch 01] Iter 1100/1123 | loss=0.27396 | 0.024s/iter | ETA: 8.7min

------------------------------------------------------------
[Epoch 01] Summary | Time: 27.8s
Train Loss: 0.327884
Vali  Loss: 0.197778
Test  Loss: 0.226136
Validation loss decreased (inf --> 0.197778).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 02] Iter  100/1123 | loss=0.26019 | 0.118s/iter | ETA: 41.7min
[Epoch 02] Iter  200/1123 | loss=0.26862 | 0.024s/iter | ETA: 8.5min
[Epoch 02] Iter  300/1123 | loss=0.24986 | 0.024s/iter | ETA: 8.4min
[Epoch 02] Iter  400/1123 | loss=0.22737 | 0.024s/iter | ETA: 8.4min
[Epoch 02] Iter  500/1123 | loss=0.25062 | 0.024s/iter | ETA: 8.3min
[Epoch 02] Iter  600/1123 | loss=0.26506 | 0.024s/iter | ETA: 8.3min
[Epoch 02] Iter  700/1123 | loss=0.26955 | 0.024s/iter | ETA: 8.2min
[Epoch 02] Iter  800/1123 | loss=0.21271 | 0.024s/iter | ETA: 8.2min
[Epoch 02] Iter  900/1123 | loss=0.21018 | 0.025s/iter | ETA: 8.4min
[Epoch 02] Iter 1000/1123 | loss=0.22132 | 0.025s/iter | ETA: 8.4min
[Epoch 02] Iter 1100/1123 | loss=0.22881 | 0.025s/iter | ETA: 8.3min

------------------------------------------------------------
[Epoch 02] Summary | Time: 27.4s
Train Loss: 0.241549
Vali  Loss: 0.183892
Test  Loss: 0.209792
Validation loss decreased (0.197778 --> 0.183892).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.00025
[Epoch 03] Iter  100/1123 | loss=0.20855 | 0.117s/iter | ETA: 39.3min
[Epoch 03] Iter  200/1123 | loss=0.24127 | 0.025s/iter | ETA: 8.3min
[Epoch 03] Iter  300/1123 | loss=0.21227 | 0.025s/iter | ETA: 8.2min
[Epoch 03] Iter  400/1123 | loss=0.22077 | 0.025s/iter | ETA: 8.1min
[Epoch 03] Iter  500/1123 | loss=0.22115 | 0.025s/iter | ETA: 8.1min
[Epoch 03] Iter  600/1123 | loss=0.23633 | 0.025s/iter | ETA: 8.0min
[Epoch 03] Iter  700/1123 | loss=0.21628 | 0.025s/iter | ETA: 8.0min
[Epoch 03] Iter  800/1123 | loss=0.22614 | 0.024s/iter | ETA: 7.8min
[Epoch 03] Iter  900/1123 | loss=0.23139 | 0.024s/iter | ETA: 7.7min
[Epoch 03] Iter 1000/1123 | loss=0.21295 | 0.024s/iter | ETA: 7.7min
[Epoch 03] Iter 1100/1123 | loss=0.21415 | 0.024s/iter | ETA: 7.6min

------------------------------------------------------------
[Epoch 03] Summary | Time: 27.8s
Train Loss: 0.229158
Vali  Loss: 0.180807
Test  Loss: 0.206494
Validation loss decreased (0.183892 --> 0.180807).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.000125
[Epoch 04] Iter  100/1123 | loss=0.20539 | 0.109s/iter | ETA: 34.4min
[Epoch 04] Iter  200/1123 | loss=0.23320 | 0.024s/iter | ETA: 7.6min
[Epoch 04] Iter  300/1123 | loss=0.19091 | 0.024s/iter | ETA: 7.5min
[Epoch 04] Iter  400/1123 | loss=0.21195 | 0.024s/iter | ETA: 7.5min
[Epoch 04] Iter  500/1123 | loss=0.25729 | 0.024s/iter | ETA: 7.4min
[Epoch 04] Iter  600/1123 | loss=0.21919 | 0.024s/iter | ETA: 7.4min
[Epoch 04] Iter  700/1123 | loss=0.23592 | 0.024s/iter | ETA: 7.4min
[Epoch 04] Iter  800/1123 | loss=0.23303 | 0.024s/iter | ETA: 7.3min
[Epoch 04] Iter  900/1123 | loss=0.21755 | 0.024s/iter | ETA: 7.2min
[Epoch 04] Iter 1000/1123 | loss=0.23388 | 0.024s/iter | ETA: 7.2min
[Epoch 04] Iter 1100/1123 | loss=0.19794 | 0.024s/iter | ETA: 7.1min

------------------------------------------------------------
[Epoch 04] Summary | Time: 27.1s
Train Loss: 0.224419
Vali  Loss: 0.179598
Test  Loss: 0.205854
Validation loss decreased (0.180807 --> 0.179598).  Saving model ...
------------------------------------------------------------
Updating learning rate to 6.25e-05
[Epoch 05] Iter  100/1123 | loss=0.24538 | 0.109s/iter | ETA: 32.6min
[Epoch 05] Iter  200/1123 | loss=0.25191 | 0.024s/iter | ETA: 7.1min
[Epoch 05] Iter  300/1123 | loss=0.19821 | 0.024s/iter | ETA: 7.1min
[Epoch 05] Iter  400/1123 | loss=0.20791 | 0.024s/iter | ETA: 7.0min
[Epoch 05] Iter  500/1123 | loss=0.22914 | 0.024s/iter | ETA: 7.0min
[Epoch 05] Iter  600/1123 | loss=0.23239 | 0.024s/iter | ETA: 6.9min
[Epoch 05] Iter  700/1123 | loss=0.23793 | 0.024s/iter | ETA: 6.9min
[Epoch 05] Iter  800/1123 | loss=0.23825 | 0.024s/iter | ETA: 6.9min
[Epoch 05] Iter  900/1123 | loss=0.20856 | 0.024s/iter | ETA: 6.8min
[Epoch 05] Iter 1000/1123 | loss=0.21143 | 0.024s/iter | ETA: 6.7min
[Epoch 05] Iter 1100/1123 | loss=0.23424 | 0.024s/iter | ETA: 6.7min

------------------------------------------------------------
[Epoch 05] Summary | Time: 27.1s
Train Loss: 0.222180
Vali  Loss: 0.178905
Test  Loss: 0.205120
Validation loss decreased (0.179598 --> 0.178905).  Saving model ...
------------------------------------------------------------
Updating learning rate to 3.125e-05
[Epoch 06] Iter  100/1123 | loss=0.23977 | 0.110s/iter | ETA: 30.7min
[Epoch 06] Iter  200/1123 | loss=0.17753 | 0.025s/iter | ETA: 6.8min
[Epoch 06] Iter  300/1123 | loss=0.21868 | 0.025s/iter | ETA: 6.8min
[Epoch 06] Iter  400/1123 | loss=0.21467 | 0.025s/iter | ETA: 6.8min
[Epoch 06] Iter  500/1123 | loss=0.23810 | 0.025s/iter | ETA: 6.7min
[Epoch 06] Iter  600/1123 | loss=0.25124 | 0.025s/iter | ETA: 6.7min
[Epoch 06] Iter  700/1123 | loss=0.23946 | 0.025s/iter | ETA: 6.7min
[Epoch 06] Iter  800/1123 | loss=0.23995 | 0.025s/iter | ETA: 6.6min
[Epoch 06] Iter  900/1123 | loss=0.26501 | 0.025s/iter | ETA: 6.6min
[Epoch 06] Iter 1000/1123 | loss=0.22692 | 0.025s/iter | ETA: 6.5min
[Epoch 06] Iter 1100/1123 | loss=0.23610 | 0.025s/iter | ETA: 6.5min

------------------------------------------------------------
[Epoch 06] Summary | Time: 27.9s
Train Loss: 0.221064
Vali  Loss: 0.179147
Test  Loss: 0.204848
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 1.5625e-05
[Epoch 07] Iter  100/1123 | loss=0.29463 | 0.117s/iter | ETA: 30.6min
[Epoch 07] Iter  200/1123 | loss=0.22071 | 0.025s/iter | ETA: 6.5min
[Epoch 07] Iter  300/1123 | loss=0.23424 | 0.025s/iter | ETA: 6.3min
[Epoch 07] Iter  400/1123 | loss=0.20852 | 0.024s/iter | ETA: 6.2min
[Epoch 07] Iter  500/1123 | loss=0.23298 | 0.024s/iter | ETA: 6.1min
[Epoch 07] Iter  600/1123 | loss=0.20893 | 0.024s/iter | ETA: 6.0min
[Epoch 07] Iter  700/1123 | loss=0.19815 | 0.024s/iter | ETA: 6.0min
[Epoch 07] Iter  800/1123 | loss=0.18545 | 0.024s/iter | ETA: 5.9min
[Epoch 07] Iter  900/1123 | loss=0.23655 | 0.024s/iter | ETA: 5.9min
[Epoch 07] Iter 1000/1123 | loss=0.23585 | 0.024s/iter | ETA: 5.9min
[Epoch 07] Iter 1100/1123 | loss=0.22305 | 0.024s/iter | ETA: 5.8min

------------------------------------------------------------
[Epoch 07] Summary | Time: 27.5s
Train Loss: 0.220428
Vali  Loss: 0.179052
Test  Loss: 0.204877
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 7.8125e-06
[Epoch 08] Iter  100/1123 | loss=0.24749 | 0.111s/iter | ETA: 26.8min
[Epoch 08] Iter  200/1123 | loss=0.20348 | 0.025s/iter | ETA: 5.9min
[Epoch 08] Iter  300/1123 | loss=0.19820 | 0.025s/iter | ETA: 5.9min
[Epoch 08] Iter  400/1123 | loss=0.22643 | 0.025s/iter | ETA: 5.8min
[Epoch 08] Iter  500/1123 | loss=0.22285 | 0.025s/iter | ETA: 5.8min
[Epoch 08] Iter  600/1123 | loss=0.19653 | 0.025s/iter | ETA: 5.7min
[Epoch 08] Iter  700/1123 | loss=0.21391 | 0.025s/iter | ETA: 5.7min
[Epoch 08] Iter  800/1123 | loss=0.21089 | 0.025s/iter | ETA: 5.7min
[Epoch 08] Iter  900/1123 | loss=0.20051 | 0.025s/iter | ETA: 5.6min
[Epoch 08] Iter 1000/1123 | loss=0.22530 | 0.025s/iter | ETA: 5.6min
[Epoch 08] Iter 1100/1123 | loss=0.24027 | 0.025s/iter | ETA: 5.5min

------------------------------------------------------------
[Epoch 08] Summary | Time: 27.9s
Train Loss: 0.220184
Vali  Loss: 0.179168
Test  Loss: 0.204840
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 3.90625e-06
[Epoch 09] Iter  100/1123 | loss=0.22444 | 0.108s/iter | ETA: 24.2min
[Epoch 09] Iter  200/1123 | loss=0.23360 | 0.025s/iter | ETA: 5.5min
[Epoch 09] Iter  300/1123 | loss=0.19802 | 0.025s/iter | ETA: 5.4min
[Epoch 09] Iter  400/1123 | loss=0.21967 | 0.025s/iter | ETA: 5.4min
[Epoch 09] Iter  500/1123 | loss=0.21918 | 0.025s/iter | ETA: 5.3min
[Epoch 09] Iter  600/1123 | loss=0.21077 | 0.025s/iter | ETA: 5.3min
[Epoch 09] Iter  700/1123 | loss=0.22025 | 0.025s/iter | ETA: 5.2min
[Epoch 09] Iter  800/1123 | loss=0.20107 | 0.025s/iter | ETA: 5.2min
[Epoch 09] Iter  900/1123 | loss=0.22250 | 0.025s/iter | ETA: 5.2min
[Epoch 09] Iter 1000/1123 | loss=0.22429 | 0.025s/iter | ETA: 5.1min
[Epoch 09] Iter 1100/1123 | loss=0.19436 | 0.025s/iter | ETA: 5.1min

------------------------------------------------------------
[Epoch 09] Summary | Time: 27.9s
Train Loss: 0.220041
Vali  Loss: 0.179173
Test  Loss: 0.204878
EarlyStopping counter: 4 out of 5
------------------------------------------------------------
Updating learning rate to 1.953125e-06
[Epoch 10] Iter  100/1123 | loss=0.22380 | 0.116s/iter | ETA: 23.7min
[Epoch 10] Iter  200/1123 | loss=0.24525 | 0.025s/iter | ETA: 5.1min
[Epoch 10] Iter  300/1123 | loss=0.20888 | 0.025s/iter | ETA: 5.0min
[Epoch 10] Iter  400/1123 | loss=0.23135 | 0.025s/iter | ETA: 4.9min
[Epoch 10] Iter  500/1123 | loss=0.20040 | 0.025s/iter | ETA: 4.9min
[Epoch 10] Iter  600/1123 | loss=0.21776 | 0.025s/iter | ETA: 4.8min
[Epoch 10] Iter  700/1123 | loss=0.20596 | 0.025s/iter | ETA: 4.8min
[Epoch 10] Iter  800/1123 | loss=0.22702 | 0.025s/iter | ETA: 4.8min
[Epoch 10] Iter  900/1123 | loss=0.22854 | 0.025s/iter | ETA: 4.7min
[Epoch 10] Iter 1000/1123 | loss=0.21727 | 0.025s/iter | ETA: 4.7min
[Epoch 10] Iter 1100/1123 | loss=0.19489 | 0.025s/iter | ETA: 4.6min

------------------------------------------------------------
[Epoch 10] Summary | Time: 28.2s
Train Loss: 0.219920
Vali  Loss: 0.179064
Test  Loss: 0.204779
EarlyStopping counter: 5 out of 5
------------------------------------------------------------

************************************************************
EARLY STOPPING TRIGGERED
************************************************************
>>>>>>>testing : electricity_96_336_FreDEA_electricity_ftM_sl96_ll48_pl336_dm64_nh8_el1_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 4925
mse:0.20502005517482758, mae:0.3050096035003662, rmse:0.45279139280319214
✅ 实验完成: MSE=0.205020, MAE=0.305010

================================================================================
运行实验: electricity - Pred 720
================================================================================
命令: python -u run_longExp.py --data electricity --data_path electricity.csv --model FreDEA --model_id electricity_96_720 --seq_len 96 --pred_len 720 --enc_in 321 --dec_in 321 --c_out 321 --d_model 128 --d_ff 256 --e_layers 1 --memory_size 128 --bottleneck_dim 4 --dropout 0.1 --batch_size 16 --learning_rate 0.0005 --lradj type1 --train_epochs 20 --patience 5 --itr 1 --num_workers 4

Args in experiment:
Namespace(is_training=1, train_only=False, model_id='electricity_96_720', model='FreDEA', data='electricity', root_path='./dataset/', data_path='electricity.csv', channel_independence=0, features='M', target='OT', freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=720, individual=False, embed_type=0, enc_in=321, dec_in=321, c_out=321, d_model=128, reg_lambda=0.01, wavelet='haar', n_bins=8, n_heads=8, e_layers=1, d_layers=1, d_ff=256, moving_avg=25, factor=1, distil=True, rev_affine=1, memory_size=128, bottleneck_dim=4, ablation_freq=0, ablation_tea=0, ablation_cea=0, dropout=0.1, embed='timeF', activation='gelu', output_attention=False, do_predict=False, num_workers=4, itr=1, train_epochs=20, batch_size=16, patience=5, learning_rate=0.0005, des='Exp', loss='mse', lradj='type1', use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=False, devices='0,1,2,3', test_flop=False)
Use GPU: cuda:0
>>>>>>>start training : electricity_96_720_FreDEA_electricity_ftM_sl96_ll48_pl720_dm128_nh8_el1_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0>>>>>>>>>>>>>>>>>>>>>>>>>>
train 17597
val 1913
test 4541

============================================================
TRAINING CONFIGURATION
============================================================
Model:           FreDEA
Dataset:         electricity
Seq/Pred Len:    96 -> 720
Batch Size:      16
Learning Rate:   0.0005
Train Epochs:    20
Total Params:    301,272
d_model:         128
n_heads:         8
e_layers:        1
============================================================

[Epoch 01] Iter  100/1099 | loss=0.40956 | 0.060s/iter | ETA: 21.9min
[Epoch 01] Iter  200/1099 | loss=0.31643 | 0.048s/iter | ETA: 17.5min
[Epoch 01] Iter  300/1099 | loss=0.29134 | 0.048s/iter | ETA: 17.5min
[Epoch 01] Iter  400/1099 | loss=0.32172 | 0.048s/iter | ETA: 17.3min
[Epoch 01] Iter  500/1099 | loss=0.34221 | 0.048s/iter | ETA: 17.2min
[Epoch 01] Iter  600/1099 | loss=0.31225 | 0.048s/iter | ETA: 17.2min
[Epoch 01] Iter  700/1099 | loss=0.27670 | 0.048s/iter | ETA: 17.1min
[Epoch 01] Iter  800/1099 | loss=0.26240 | 0.048s/iter | ETA: 17.1min
[Epoch 01] Iter  900/1099 | loss=0.30455 | 0.049s/iter | ETA: 17.0min
[Epoch 01] Iter 1000/1099 | loss=0.27317 | 0.049s/iter | ETA: 17.0min

------------------------------------------------------------
[Epoch 01] Summary | Time: 53.7s
Train Loss: 0.330479
Vali  Loss: 0.216007
Test  Loss: 0.252505
Validation loss decreased (inf --> 0.216007).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.0005
[Epoch 02] Iter  100/1099 | loss=0.30891 | 0.234s/iter | ETA: 81.1min
[Epoch 02] Iter  200/1099 | loss=0.27362 | 0.048s/iter | ETA: 16.5min
[Epoch 02] Iter  300/1099 | loss=0.27431 | 0.048s/iter | ETA: 16.5min
[Epoch 02] Iter  400/1099 | loss=0.27756 | 0.048s/iter | ETA: 16.5min
[Epoch 02] Iter  500/1099 | loss=0.24827 | 0.048s/iter | ETA: 16.4min
[Epoch 02] Iter  600/1099 | loss=0.25179 | 0.048s/iter | ETA: 16.3min
[Epoch 02] Iter  700/1099 | loss=0.23209 | 0.048s/iter | ETA: 16.2min
[Epoch 02] Iter  800/1099 | loss=0.21137 | 0.048s/iter | ETA: 16.1min
[Epoch 02] Iter  900/1099 | loss=0.22594 | 0.048s/iter | ETA: 16.0min
[Epoch 02] Iter 1000/1099 | loss=0.24353 | 0.048s/iter | ETA: 15.9min

------------------------------------------------------------
[Epoch 02] Summary | Time: 53.1s
Train Loss: 0.249246
Vali  Loss: 0.212220
Test  Loss: 0.228947
Validation loss decreased (0.216007 --> 0.212220).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.00025
[Epoch 03] Iter  100/1099 | loss=0.23899 | 0.225s/iter | ETA: 73.7min
[Epoch 03] Iter  200/1099 | loss=0.22496 | 0.048s/iter | ETA: 15.8min
[Epoch 03] Iter  300/1099 | loss=0.21843 | 0.048s/iter | ETA: 15.7min
[Epoch 03] Iter  400/1099 | loss=0.20777 | 0.048s/iter | ETA: 15.6min
[Epoch 03] Iter  500/1099 | loss=0.21866 | 0.048s/iter | ETA: 15.5min
[Epoch 03] Iter  600/1099 | loss=0.21229 | 0.048s/iter | ETA: 15.4min
[Epoch 03] Iter  700/1099 | loss=0.19910 | 0.049s/iter | ETA: 15.4min
[Epoch 03] Iter  800/1099 | loss=0.21690 | 0.049s/iter | ETA: 15.5min
[Epoch 03] Iter  900/1099 | loss=0.23871 | 0.048s/iter | ETA: 15.3min
[Epoch 03] Iter 1000/1099 | loss=0.21849 | 0.048s/iter | ETA: 15.2min

------------------------------------------------------------
[Epoch 03] Summary | Time: 53.6s
Train Loss: 0.221275
Vali  Loss: 0.203381
Test  Loss: 0.223601
Validation loss decreased (0.212220 --> 0.203381).  Saving model ...
------------------------------------------------------------
Updating learning rate to 0.000125
[Epoch 04] Iter  100/1099 | loss=0.21569 | 0.229s/iter | ETA: 70.8min
[Epoch 04] Iter  200/1099 | loss=0.19927 | 0.049s/iter | ETA: 15.1min
[Epoch 04] Iter  300/1099 | loss=0.22807 | 0.049s/iter | ETA: 14.9min
[Epoch 04] Iter  400/1099 | loss=0.22501 | 0.049s/iter | ETA: 14.8min
[Epoch 04] Iter  500/1099 | loss=0.22003 | 0.049s/iter | ETA: 14.8min
[Epoch 04] Iter  600/1099 | loss=0.21532 | 0.049s/iter | ETA: 14.7min
[Epoch 04] Iter  700/1099 | loss=0.20355 | 0.049s/iter | ETA: 14.6min
[Epoch 04] Iter  800/1099 | loss=0.20155 | 0.049s/iter | ETA: 14.5min
[Epoch 04] Iter  900/1099 | loss=0.21506 | 0.049s/iter | ETA: 14.4min
[Epoch 04] Iter 1000/1099 | loss=0.22141 | 0.049s/iter | ETA: 14.4min

------------------------------------------------------------
[Epoch 04] Summary | Time: 53.8s
Train Loss: 0.212464
Vali  Loss: 0.206311
Test  Loss: 0.224172
EarlyStopping counter: 1 out of 5
------------------------------------------------------------
Updating learning rate to 6.25e-05
[Epoch 05] Iter  100/1099 | loss=0.21253 | 0.229s/iter | ETA: 66.9min
[Epoch 05] Iter  200/1099 | loss=0.20702 | 0.048s/iter | ETA: 14.0min
[Epoch 05] Iter  300/1099 | loss=0.19984 | 0.048s/iter | ETA: 14.0min
[Epoch 05] Iter  400/1099 | loss=0.20166 | 0.048s/iter | ETA: 13.8min
[Epoch 05] Iter  500/1099 | loss=0.19714 | 0.048s/iter | ETA: 13.7min
[Epoch 05] Iter  600/1099 | loss=0.21662 | 0.048s/iter | ETA: 13.7min
[Epoch 05] Iter  700/1099 | loss=0.22718 | 0.048s/iter | ETA: 13.6min
[Epoch 05] Iter  800/1099 | loss=0.21052 | 0.048s/iter | ETA: 13.5min
[Epoch 05] Iter  900/1099 | loss=0.21501 | 0.048s/iter | ETA: 13.4min
[Epoch 05] Iter 1000/1099 | loss=0.20973 | 0.048s/iter | ETA: 13.4min

------------------------------------------------------------
[Epoch 05] Summary | Time: 53.4s
Train Loss: 0.208879
Vali  Loss: 0.208355
Test  Loss: 0.224780
EarlyStopping counter: 2 out of 5
------------------------------------------------------------
Updating learning rate to 3.125e-05
[Epoch 06] Iter  100/1099 | loss=0.19859 | 0.229s/iter | ETA: 62.6min
[Epoch 06] Iter  200/1099 | loss=0.22234 | 0.049s/iter | ETA: 13.3min
[Epoch 06] Iter  300/1099 | loss=0.20584 | 0.049s/iter | ETA: 13.2min
[Epoch 06] Iter  400/1099 | loss=0.19307 | 0.049s/iter | ETA: 13.1min
[Epoch 06] Iter  500/1099 | loss=0.21280 | 0.049s/iter | ETA: 13.0min
[Epoch 06] Iter  600/1099 | loss=0.20176 | 0.049s/iter | ETA: 12.9min
[Epoch 06] Iter  700/1099 | loss=0.20764 | 0.049s/iter | ETA: 12.8min
[Epoch 06] Iter  800/1099 | loss=0.23645 | 0.049s/iter | ETA: 12.7min
[Epoch 06] Iter  900/1099 | loss=0.20715 | 0.049s/iter | ETA: 12.7min
[Epoch 06] Iter 1000/1099 | loss=0.20285 | 0.049s/iter | ETA: 12.6min

------------------------------------------------------------
[Epoch 06] Summary | Time: 53.9s
Train Loss: 0.207177
Vali  Loss: 0.208334
Test  Loss: 0.224728
EarlyStopping counter: 3 out of 5
------------------------------------------------------------
Updating learning rate to 1.5625e-05
[Epoch 07] Iter  100/1099 | loss=0.20084 | 0.230s/iter | ETA: 58.5min
[Epoch 07] Iter  200/1099 | loss=0.22072 | 0.048s/iter | ETA: 12.2min
[Epoch 07] Iter  300/1099 | loss=0.19448 | 0.049s/iter | ETA: 12.3min
[Epoch 07] Iter  400/1099 | loss=0.21268 | 0.048s/iter | ETA: 12.1min
[Epoch 07] Iter  500/1099 | loss=0.20133 | 0.049s/iter | ETA: 12.1min
[Epoch 07] Iter  600/1099 | loss=0.21347 | 0.049s/iter | ETA: 12.0min
[Epoch 07] Iter  700/1099 | loss=0.20969 | 0.049s/iter | ETA: 12.0min
[Epoch 07] Iter  800/1099 | loss=0.21140 | 0.049s/iter | ETA: 11.8min
[Epoch 07] Iter  900/1099 | loss=0.20270 | 0.049s/iter | ETA: 11.7min
[Epoch 07] Iter 1000/1099 | loss=0.20210 | 0.049s/iter | ETA: 11.7min

------------------------------------------------------------
[Epoch 07] Summary | Time: 53.7s
Train Loss: 0.206369
Vali  Loss: 0.208280
Test  Loss: 0.224091
EarlyStopping counter: 4 out of 5
------------------------------------------------------------
Updating learning rate to 7.8125e-06
[Epoch 08] Iter  100/1099 | loss=0.21500 | 0.224s/iter | ETA: 53.1min
[Epoch 08] Iter  200/1099 | loss=0.21217 | 0.048s/iter | ETA: 11.3min
[Epoch 08] Iter  300/1099 | loss=0.21043 | 0.048s/iter | ETA: 11.2min
[Epoch 08] Iter  400/1099 | loss=0.20187 | 0.048s/iter | ETA: 11.2min
[Epoch 08] Iter  500/1099 | loss=0.20273 | 0.048s/iter | ETA: 11.1min
[Epoch 08] Iter  600/1099 | loss=0.19881 | 0.048s/iter | ETA: 11.1min
[Epoch 08] Iter  700/1099 | loss=0.19136 | 0.048s/iter | ETA: 10.9min
[Epoch 08] Iter  800/1099 | loss=0.18860 | 0.048s/iter | ETA: 10.8min
[Epoch 08] Iter  900/1099 | loss=0.20307 | 0.048s/iter | ETA: 10.7min
[Epoch 08] Iter 1000/1099 | loss=0.20724 | 0.048s/iter | ETA: 10.7min

------------------------------------------------------------
[Epoch 08] Summary | Time: 53.2s
Train Loss: 0.205955
Vali  Loss: 0.207062
Test  Loss: 0.223552
EarlyStopping counter: 5 out of 5
------------------------------------------------------------

************************************************************
EARLY STOPPING TRIGGERED
************************************************************
>>>>>>>testing : electricity_96_720_FreDEA_electricity_ftM_sl96_ll48_pl720_dm128_nh8_el1_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 4541
mse:0.2235366553068161, mae:0.3172343373298645, rmse:0.47279661893844604
✅ 实验完成: MSE=0.223537, MAE=0.317234
