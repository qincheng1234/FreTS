"""
FreDEA 超参数自动搜索脚本 (使用 Hyperopt)

使用方法:
    python run_hyperopt.py --data ETTm1 --pred_len 96 --max_evals 50
    python run_hyperopt.py --data electricity --pred_len 720 --max_evals 30

依赖安装:
    pip install hyperopt
"""

import argparse
import os
import sys
import json
import time
import random
import numpy as np
import torch
from datetime import datetime

from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials, space_eval
from hyperopt.pyll import scope

from exp.exp_main import Exp_Main

# =============================================================================
# 数据集配置
# =============================================================================

DATASET_CONFIG = {
    'ETTm1': {
        'data_path': 'ETTm1.csv',
        'enc_in': 7, 'dec_in': 7, 'c_out': 7,
    },
    'ETTm2': {
        'data_path': 'ETTm2.csv',
        'enc_in': 7, 'dec_in': 7, 'c_out': 7,
    },
    'ETTh1': {
        'data_path': 'ETTh1.csv',
        'enc_in': 7, 'dec_in': 7, 'c_out': 7,
    },
    'ETTh2': {
        'data_path': 'ETTh2.csv',
        'enc_in': 7, 'dec_in': 7, 'c_out': 7,
    },
    'exchange_rate': {
        'data_path': 'exchange_rate.csv',
        'enc_in': 8, 'dec_in': 8, 'c_out': 8,
    },
    'electricity': {
        'data_path': 'electricity.csv',
        'enc_in': 321, 'dec_in': 321, 'c_out': 321,
    },
    'weather': {
        'data_path': 'weather.csv',
        'enc_in': 21, 'dec_in': 21, 'c_out': 21,
    },
    'traffic': {
        'data_path': 'traffic.csv',
        'enc_in': 862, 'dec_in': 862, 'c_out': 862,
    },
}

# =============================================================================
# 搜索空间定义
# =============================================================================

def get_search_space(dataset_name, pred_len):
    """
    根据数据集和预测长度返回合适的搜索空间
    """
    enc_in = DATASET_CONFIG[dataset_name]['enc_in']
    is_high_dim = enc_in > 100
    is_long_horizon = pred_len >= 336
    
    # 基础搜索空间
    space = {
        # 模型维度
        'd_model': hp.choice('d_model', [64, 128, 256, 512]),
        'd_ff': hp.choice('d_ff', [64, 128, 256]),
        
        # 层数
        'e_layers': hp.choice('e_layers', [1, 2, 3]),
        
        # 正则化
        'dropout': hp.uniform('dropout', 0.05, 0.4),
        
        # 学习率
        'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-3)),
        
        # 批大小
        'batch_size': hp.choice('batch_size', [8, 16, 32]),
        
        # 学习率调度
        'lradj': hp.choice('lradj', ['1', '2', '3']),
    }
    
    # 高维数据集的特殊配置
    if is_high_dim:
        space['batch_size'] = hp.choice('batch_size_highdim', [4, 8, 16])
        space['d_model'] = hp.choice('d_model_highdim', [64, 128, 256])
    
    # 长序列预测的特殊配置
    if is_long_horizon:
        space['dropout'] = hp.uniform('dropout_long', 0.15, 0.5)
    
    return space


# =============================================================================
# 目标函数
# =============================================================================

class HyperoptObjective:
    def __init__(self, base_args, log_dir):
        self.base_args = base_args
        self.log_dir = log_dir
        self.trial_count = 0
        self.best_mse = float('inf')
        self.results_log = []
        
    def __call__(self, params):
        self.trial_count += 1
        trial_start = time.time()
        
        print("\n" + "="*70)
        print(f"  TRIAL {self.trial_count}")
        print("="*70)
        print(f"  Parameters: {json.dumps(params, indent=4, default=str)}")
        print("-"*70)
        
        try:
            # 构建参数
            args = self._build_args(params)
            
            # 设置随机种子
            fix_seed = 2021
            random.seed(fix_seed)
            torch.manual_seed(fix_seed)
            np.random.seed(fix_seed)
            
            # 创建实验
            exp = Exp_Main(args)
            
            # 生成 setting 名称
            setting = self._generate_setting(args)
            
            # 训练
            exp.train(setting)
            
            # 测试并获取 MSE
            mse, mae = self._get_test_metrics(exp, setting)
            
            # 记录结果
            trial_time = time.time() - trial_start
            result = {
                'trial': self.trial_count,
                'params': params,
                'mse': mse,
                'mae': mae,
                'time': trial_time,
                'status': 'success'
            }
            self.results_log.append(result)
            
            # 更新最佳结果
            if mse < self.best_mse:
                self.best_mse = mse
                print(f"\n  ★ NEW BEST! MSE: {mse:.6f}, MAE: {mae:.6f}")
                self._save_best_config(params, mse, mae)
            else:
                print(f"\n  MSE: {mse:.6f}, MAE: {mae:.6f} (Best: {self.best_mse:.6f})")
            
            # 保存日志
            self._save_results_log()
            
            # 清理显存
            del exp
            torch.cuda.empty_cache()
            
            return {'loss': mse, 'mae': mae, 'status': STATUS_OK}
            
        except Exception as e:
            print(f"\n  ✗ TRIAL FAILED: {str(e)}")
            
            result = {
                'trial': self.trial_count,
                'params': params,
                'error': str(e),
                'status': 'failed'
            }
            self.results_log.append(result)
            self._save_results_log()
            
            torch.cuda.empty_cache()
            
            return {'loss': float('inf'), 'status': STATUS_FAIL}
    
    def _build_args(self, params):
        """将超参数合并到基础参数"""
        args = argparse.Namespace(**vars(self.base_args))
        
        # 更新超参数
        args.d_model = params['d_model']
        args.d_ff = params['d_ff']
        args.e_layers = params['e_layers']
        args.dropout = round(params['dropout'], 4)
        args.learning_rate = round(params['learning_rate'], 6)
        args.batch_size = params['batch_size']
        args.lradj = params['lradj']
        
        return args
    
    def _generate_setting(self, args):
        """生成实验设置名称"""
        setting = 'hyperopt_{}_{}_{}_dm{}_df{}_el{}_dr{}_lr{}_bs{}'.format(
            args.model,
            args.data,
            args.pred_len,
            args.d_model,
            args.d_ff,
            args.e_layers,
            int(args.dropout * 100),
            f'{args.learning_rate:.0e}',
            args.batch_size
        )
        return setting
    
    def _get_test_metrics(self, exp, setting):
        """获取测试集指标"""
        from utils.metrics import metric
        
        test_data, test_loader = exp._get_data(flag='test')
        
        preds = []
        trues = []
        
        exp.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(exp.device)
                batch_y = batch_y.float().to(exp.device)
                
                outputs = exp.model(batch_x)
                
                f_dim = -1 if exp.args.features == 'MS' else 0
                outputs = outputs[:, -exp.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -exp.args.pred_len:, f_dim:]
                
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        return mse, mae
    
    def _save_best_config(self, params, mse, mae):
        """保存最佳配置"""
        best_config = {
            'params': params,
            'mse': mse,
            'mae': mae,
            'dataset': self.base_args.data,
            'pred_len': self.base_args.pred_len,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        best_path = os.path.join(self.log_dir, 'best_config.json')
        with open(best_path, 'w') as f:
            json.dump(best_config, f, indent=4, default=str)
        
        # 同时生成可执行命令
        cmd = self._generate_command(params)
        cmd_path = os.path.join(self.log_dir, 'best_command.txt')
        with open(cmd_path, 'w') as f:
            f.write(f"# Best MSE: {mse:.6f}, MAE: {mae:.6f}\n")
            f.write(f"# Dataset: {self.base_args.data}, Pred_len: {self.base_args.pred_len}\n")
            f.write(f"# Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(cmd)
    
    def _generate_command(self, params):
        """生成训练命令"""
        args = self.base_args
        ds = DATASET_CONFIG[args.data]
        
        cmd = f"""python -u run_longExp.py \\
    --data {args.data} \\
    --data_path {ds['data_path']} \\
    --model FreDEA \\
    --model_id {args.data}_96_{args.pred_len} \\
    --seq_len 96 \\
    --pred_len {args.pred_len} \\
    --enc_in {ds['enc_in']} \\
    --dec_in {ds['dec_in']} \\
    --c_out {ds['c_out']} \\
    --d_model {params['d_model']} \\
    --d_ff {params['d_ff']} \\
    --e_layers {params['e_layers']} \\
    --dropout {round(params['dropout'], 4)} \\
    --batch_size {params['batch_size']} \\
    --learning_rate {round(params['learning_rate'], 6)} \\
    --lradj {params['lradj']} \\
    --train_epochs 20 \\
    --patience 8 \\
    --itr 1"""
        return cmd
    
    def _save_results_log(self):
        """保存所有结果日志"""
        log_path = os.path.join(self.log_dir, 'all_trials.json')
        with open(log_path, 'w') as f:
            json.dump(self.results_log, f, indent=2, default=str)


# =============================================================================
# 主函数
# =============================================================================

def create_base_args(dataset_name, pred_len):
    """创建基础参数"""
    ds = DATASET_CONFIG[dataset_name]
    
    args = argparse.Namespace(
        # 基础配置
        is_training=1,
        train_only=False,
        model='FreDEA',
        model_id=f'{dataset_name}_96_{pred_len}',
        
        # 数据配置
        data=dataset_name,
        root_path='./dataset/',
        data_path=ds['data_path'],
        features='M',
        target='OT',
        freq='h',
        checkpoints='./checkpoints/',
        
        # 序列配置
        seq_len=96,
        label_len=48,
        pred_len=pred_len,
        
        # 模型配置 (将被 hyperopt 覆盖)
        enc_in=ds['enc_in'],
        dec_in=ds['dec_in'],
        c_out=ds['c_out'],
        d_model=128,
        d_ff=128,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        dropout=0.1,
        
        # 额外配置
        channel_independence=0,
        individual=False,
        embed_type=0,
        embed='timeF',
        activation='gelu',
        output_attention=False,
        do_predict=False,
        distil=True,
        rev_affine=1,
        moving_avg=25,
        factor=1,
        memory_size=64,
        bottleneck_dim=2,
        reg_lambda=0.01,
        wavelet='haar',
        n_bins=8,
        fusion_init=0.0,
        ablation_freq=0,
        ablation_tea=0,
        ablation_cea=0,
        
        # 训练配置
        num_workers=4,
        itr=1,
        train_epochs=15,  # 搜索时减少 epochs 加速
        batch_size=16,
        patience=5,       # 搜索时减少 patience 加速
        learning_rate=0.0005,
        des='hyperopt',
        loss='mse',
        lradj='3',
        use_amp=False,
        
        # GPU 配置
        use_gpu=torch.cuda.is_available(),
        gpu=0,
        use_multi_gpu=False,
        devices='0',
        device_ids=[0],
        
        # 其他
        test_flop=False,
    )
    
    return args


def main():
    parser = argparse.ArgumentParser(description='FreDEA Hyperparameter Search with Hyperopt')
    parser.add_argument('--data', type=str, required=True, 
                        choices=list(DATASET_CONFIG.keys()),
                        help='Dataset name')
    parser.add_argument('--pred_len', type=int, required=True, 
                        choices=[96, 192, 336, 720],
                        help='Prediction length')
    parser.add_argument('--max_evals', type=int, default=50,
                        help='Maximum number of evaluations')
    parser.add_argument('--output_dir', type=str, default='./hyperopt_results',
                        help='Output directory for results')
    
    cmd_args = parser.parse_args()
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(cmd_args.output_dir, f'{cmd_args.data}_pl{cmd_args.pred_len}_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("  FreDEA Hyperparameter Search (Hyperopt)")
    print("="*70)
    print(f"  Dataset:      {cmd_args.data}")
    print(f"  Pred Length:  {cmd_args.pred_len}")
    print(f"  Max Evals:    {cmd_args.max_evals}")
    print(f"  Output Dir:   {log_dir}")
    print("="*70 + "\n")
    
    # 创建基础参数
    base_args = create_base_args(cmd_args.data, cmd_args.pred_len)
    
    # 获取搜索空间
    space = get_search_space(cmd_args.data, cmd_args.pred_len)
    
    # 保存搜索空间配置
    space_info = {k: str(v) for k, v in space.items()}
    with open(os.path.join(log_dir, 'search_space.json'), 'w') as f:
        json.dump(space_info, f, indent=2)
    
    # 创建目标函数
    objective = HyperoptObjective(base_args, log_dir)
    
    # 创建 Trials 对象用于保存历史
    trials = Trials()
    
    # 运行优化
    print("Starting hyperparameter search...")
    start_time = time.time()
    
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=cmd_args.max_evals,
        trials=trials,
        verbose=True,
        show_progressbar=False
    )
    
    total_time = time.time() - start_time
    
    # 获取最佳参数
    best_params = space_eval(space, best)
    
    print("\n" + "="*70)
    print("  SEARCH COMPLETED")
    print("="*70)
    print(f"  Total Time:   {total_time/60:.1f} minutes")
    print(f"  Total Trials: {cmd_args.max_evals}")
    print(f"  Best MSE:     {objective.best_mse:.6f}")
    print("-"*70)
    print("  Best Parameters:")
    for k, v in best_params.items():
        print(f"    {k}: {v}")
    print("="*70)
    print(f"\n  Results saved to: {log_dir}")
    print(f"  Best command:     {os.path.join(log_dir, 'best_command.txt')}")
    print()


if __name__ == '__main__':
    main()
