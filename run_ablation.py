#!/usr/bin/env python
"""
FreDEA 消融实验运行脚本

消融设置：
1. w/o Freq: --ablation_freq 1 (移除频域处理)
2. w/o TEA: --ablation_tea 1 (移除 Temporal External Attention)
3. w/o CEA: --ablation_cea 1 (移除 Channel External Attention)
4. w/o TEA&CEA: --ablation_tea 1 --ablation_cea 1 (同时移除两者)

数据集：ETTm1, ETTh1, Weather
预测长度：96, 192, 336, 720
"""

import subprocess
import re
import os
import time
from datetime import datetime

# ============ 实验配置 ============

# 1. 数据集配置 (复用最佳优参数)
DATASET_CONFIGS = {
    'ETTm1': {
        'data_path': 'ETTm1.csv',
        'enc_in': 7,
        'configs': {
            96: {'d_model': 128, 'd_ff': 256, 'e_layers': 1, 'memory_size': 64, 
                 'bottleneck_dim': -1, 'dropout': 0.05, 'batch_size': 32, 
                 'learning_rate': 0.001, 'lradj': 'type1'},
            192: {'d_model': 128, 'd_ff': 256, 'e_layers': 1, 'memory_size': 64,
                  'bottleneck_dim': -1, 'dropout': 0.05, 'batch_size': 32,
                  'learning_rate': 0.001, 'lradj': 'type1'},
            336: {'d_model': 128, 'd_ff': 256, 'e_layers': 1, 'memory_size': 64,
                  'bottleneck_dim': -1, 'dropout': 0.1, 'batch_size': 32,
                  'learning_rate': 0.001, 'lradj': 'type1'},
            720: {'d_model': 64, 'd_ff': 128, 'e_layers': 1, 'memory_size': 64,
                  'bottleneck_dim': 2, 'dropout': 0.3, 'batch_size': 128,
                  'learning_rate': 0.0003, 'lradj': 'type1'},
        }
    },
    'ETTh1': {
        'data_path': 'ETTh1.csv',
        'enc_in': 7,
        'configs': {
            96: {'d_model': 64, 'd_ff': 128, 'e_layers': 1, 'memory_size': 64,
                 'bottleneck_dim': -1, 'dropout': 0.0, 'batch_size': 32,
                 'learning_rate': 0.001, 'lradj': 'type1'},
            192: {'d_model': 64, 'd_ff': 128, 'e_layers': 1, 'memory_size': 64,
                  'bottleneck_dim': -1, 'dropout': 0.1, 'batch_size': 32,
                  'learning_rate': 0.001, 'lradj': 'type1'},
            336: {'d_model': 64, 'd_ff': 128, 'e_layers': 1, 'memory_size': 64,
                  'bottleneck_dim': -1, 'dropout': 0.2, 'batch_size': 32,
                  'learning_rate': 0.001, 'lradj': 'type1'},
            720: {'d_model': 64, 'd_ff': 128, 'e_layers': 1, 'memory_size': 64,
                  'bottleneck_dim': 1, 'dropout': 0.4, 'batch_size': 32,
                  'learning_rate': 0.0003, 'lradj': 'type1'},
        }
    },
    'weather': {
        'data_path': 'weather.csv',
        'enc_in': 21,
        'configs': {
            96: {'d_model': 128, 'd_ff': 256, 'e_layers': 2, 'memory_size': 128,
                 'bottleneck_dim': 4, 'dropout': 0.1, 'batch_size': 32,
                 'learning_rate': 0.0005, 'lradj': '3'},
            192: {'d_model': 128, 'd_ff': 256, 'e_layers': 2, 'memory_size': 128,
                  'bottleneck_dim': 4, 'dropout': 0.1, 'batch_size': 32,
                  'learning_rate': 0.0005, 'lradj': '3'},
            336: {'d_model': 64, 'd_ff': 128, 'e_layers': 2, 'memory_size': 128,
                  'bottleneck_dim': 4, 'dropout': 0.1, 'batch_size': 32,
                  'learning_rate': 0.0005, 'lradj': '3'},
            720: {'d_model': 128, 'd_ff': 256, 'e_layers': 2, 'memory_size': 128,
                  'bottleneck_dim': 4, 'dropout': 0.1, 'batch_size': 32,
                  'learning_rate': 0.0005, 'lradj': '3'},
        }
    }
}

# 2. 消融实验配置
ABLATIONS = {
    'wo_Freq': {
        'args': ['--ablation_freq', '1'], 
        'desc': 'w/o Freq (Time Domain Only)'
    },
    'wo_TEA': {
        'args': ['--ablation_tea', '1'], 
        'desc': 'w/o TEA (No Time Attn)'
    },
    'wo_CEA': {
        'args': ['--ablation_cea', '1'], 
        'desc': 'w/o CEA (No Channel Attn)'
    },
    'wo_TEA_CEA': {
        'args': ['--ablation_tea', '1', '--ablation_cea', '1'], 
        'desc': 'w/o TEA & CEA'
    }
}

# 3. 预测长度
PRED_LENS = [96, 192, 336, 720]

# 4. 通用参数
COMMON_PARAMS = {
    'seq_len': 96,
    'train_epochs': 20,
    'patience': 8,
    'itr': 1,
    'num_workers': 4,
    'use_amp': False
}

WEATHER_EPOCHS = 30


# ============ 工具函数 ============

def build_command(dataset, pred_len, ablation_name):
    """构建消融实验命令"""
    data_config = DATASET_CONFIGS[dataset]
    pred_config = data_config['configs'][pred_len]
    ablation_config = ABLATIONS[ablation_name]
    
    # 基础命令
    cmd = [
        'python', '-u', 'run_longExp.py',
        '--data', dataset,
        '--data_path', data_config['data_path'],
        '--model', 'FreDEA',
        '--model_id', f'{dataset}_{COMMON_PARAMS["seq_len"]}_{pred_len}_{ablation_name}',
        '--seq_len', str(COMMON_PARAMS['seq_len']),
        '--pred_len', str(pred_len),
        '--enc_in', str(data_config['enc_in']),
        '--dec_in', str(data_config['enc_in']),
        '--c_out', str(data_config['enc_in']),
        '--d_model', str(pred_config['d_model']),
        '--d_ff', str(pred_config['d_ff']),
        '--e_layers', str(pred_config['e_layers']),
        '--memory_size', str(pred_config['memory_size']),
        '--bottleneck_dim', str(pred_config['bottleneck_dim']),
        '--dropout', str(pred_config['dropout']),
        '--batch_size', str(pred_config['batch_size']),
        '--learning_rate', str(pred_config['learning_rate']),
        '--lradj', pred_config['lradj'],
        '--train_epochs', str(WEATHER_EPOCHS if dataset == 'weather' else COMMON_PARAMS['train_epochs']),
        '--patience', str(COMMON_PARAMS['patience']),
        '--itr', str(COMMON_PARAMS['itr']),
        '--num_workers', str(COMMON_PARAMS['num_workers']),
    ]
    
    # 添加消融参数
    cmd.extend(ablation_config['args'])
    
    # 其他配置
    if 'rev_affine' in pred_config:
        cmd.append('--rev_affine')
        cmd.append(str(pred_config['rev_affine']))
        
    if 'fusion_init' in pred_config:
        cmd.append('--fusion_init')
        cmd.append(str(pred_config['fusion_init']))

    if COMMON_PARAMS['use_amp']:
        cmd.append('--use_amp')
        
    return cmd

def extract_results(output):
    """提取MSE/MAE"""
    mse_pattern = r'mse[:\s]+([0-9.]+)'
    mae_pattern = r'mae[:\s]+([0-9.]+)'
    
    mse_match = re.search(mse_pattern, output, re.IGNORECASE)
    mae_match = re.search(mae_pattern, output, re.IGNORECASE)
    
    mse = float(mse_match.group(1)) if mse_match else None
    mae = float(mae_match.group(1)) if mae_match else None
    return mse, mae

def run_experiment(dataset, pred_len, ablation_name):
    """运行实验"""
    cmd = build_command(dataset, pred_len, ablation_name)
    
    print(f"\n{'='*60}")
    print(f"运行消融: {dataset} - {pred_len} - {ablation_name}")
    print(f"描述: {ABLATIONS[ablation_name]['desc']}")
    print(f"{'='*60}")
    print("Command: " + " ".join(cmd))
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        full_output = []
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.strip())
                full_output.append(line)
        
        process.wait()
        full_output_str = "".join(full_output)
        
        mse, mae = extract_results(full_output_str)
        
        status = 'success' if mse is not None else 'failed'
        print(f"结果: {status.upper()} - MSE: {mse}, MAE: {mae}")
        
        return {
            'dataset': dataset,
            'pred_len': pred_len,
            'ablation': ablation_name,
            'mse': mse,
            'mae': mae,
            'status': status
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return {
            'dataset': dataset,
            'pred_len': pred_len,
            'ablation': ablation_name,
            'mse': None,
            'mae': None,
            'status': 'error'
        }

def main():
    results = []
    
    # 获取需要运行的数据集key
    target_datasets = ['ETTm1', 'ETTh1', 'weather']
    
    print(f"开始执行消融实验...")
    print(f"数据集: {target_datasets}")
    print(f"预测长度: {PRED_LENS}")
    print(f"消融项: {list(ABLATIONS.keys())}")
    
    total_experiments = len(target_datasets) * len(PRED_LENS) * len(ABLATIONS)
    current = 0
    
    for dataset in target_datasets:
        for pred_len in PRED_LENS:
            for ablation_name in ABLATIONS.keys():
                current += 1
                print(f"\n进度: {current}/{total_experiments}")
                res = run_experiment(dataset, pred_len, ablation_name)
                results.append(res)
    
    # 生成报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f'Ablation_Res_{timestamp}.md'
    
    lines = [
        f"# FreDEA 消融实验报告",
        f"时间: {timestamp}",
        f"\n## 汇总表格 (MSE)",
        f"\n| Dataset | Pred | Baseline | w/o Freq | w/o TEA | w/o CEA | w/o TEA&CEA |",
        f"|---|---|---|---|---|---|---|"
    ]
    
    # 这里我们只在这个脚本里跑了消融，Baseline数据不在这个脚本里
    # 如果要对比，需要手动填入Baseline，这里只列出消融结果
    
    for dataset in target_datasets:
        for pred_len in PRED_LENS:
            row = [dataset, str(pred_len), "N/A"] # Baseline placeholder
            
            # 按顺序提取消融结果
            for ab_name in ['wo_Freq', 'wo_TEA', 'wo_CEA', 'wo_TEA_CEA']:
                res = next((r for r in results if r['dataset'] == dataset and r['pred_len'] == pred_len and r['ablation'] == ab_name), None)
                val = f"{res['mse']:.4f}" if res and res['mse'] else "Fail"
                row.append(val)
            
            lines.append("| " + " | ".join(row) + " |")

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"\n报告已保存: {report_file}")

if __name__ == "__main__":
    main()
