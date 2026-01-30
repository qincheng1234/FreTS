#!/usr/bin/env python
"""
FreDEA 批量实验运行脚本

功能：
1. 在多个数据集上测试FreDEA性能
2. 自动提取MSE/MAE/RMSE结果
3. 生成Markdown格式的汇总表格

数据集：ETTm1, ETTm2, ETTh1, ETTh2, Exchange, Weather, Electricity
预测长度：96, 192, 336, 720
"""

import subprocess
import re
import os
from datetime import datetime

# ============ 配置部分 ============

# 数据集配置（针对每个数据集的最优超参数）
# 数据集配置（针对每个数据集的最优超参数 - 优化版 V2）
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
            # [优化] 720步: bottleneck 1->2，增加一点容量
            720: {'d_model': 64, 'd_ff': 128, 'e_layers': 1, 'memory_size': 64,
                  'bottleneck_dim': 2, 'dropout': 0.3, 'batch_size': 128,
                  'learning_rate': 0.0003, 'lradj': 'type1'},
        }
    },
    'ETTm2': {
        'data_path': 'ETTm2.csv',
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
                  'bottleneck_dim': 1, 'dropout': 0.3, 'batch_size': 128,
                  'learning_rate': 0.0003, 'lradj': 'type1'},
        }
    },
    'ETTh1': {
        'data_path': 'ETTh1.csv',
        'enc_in': 7,
        'configs': {
            # [优化 V4] 96步: d_model=64 (保持较优), dropout->0.0 (消灭欠拟合)
            96: {'d_model': 64, 'd_ff': 128, 'e_layers': 1, 'memory_size': 64,
                 'bottleneck_dim': -1, 'dropout': 0.0, 'batch_size': 32,
                 'learning_rate': 0.001, 'lradj': 'type1'},
            192: {'d_model': 64, 'd_ff': 128, 'e_layers': 1, 'memory_size': 64,
                  'bottleneck_dim': -1, 'dropout': 0.1, 'batch_size': 32,
                  'learning_rate': 0.001, 'lradj': 'type1'},
            336: {'d_model': 64, 'd_ff': 128, 'e_layers': 1, 'memory_size': 64,
                  'bottleneck_dim': -1, 'dropout': 0.2, 'batch_size': 32,
                  'learning_rate': 0.001, 'lradj': 'type1'},
            # [V7] 720步: dropout 0.25->0.4 (0124中间结果显示0.4表现更好)
            720: {'d_model': 64, 'd_ff': 128, 'e_layers': 1, 'memory_size': 64,
                  'bottleneck_dim': 1, 'dropout': 0.4, 'batch_size': 32,
                  'learning_rate': 0.0003, 'lradj': 'type1'},
        }
    },
    'ETTh2': {
        'data_path': 'ETTh2.csv',
        'enc_in': 7,
        'configs': {
            # [V27] ETTh2 优化: rev_affine=0 (关闭可学习仿射，减少非平稳过拟合)
            # 96/192: fusion_init=3.0 偏向趋势
            96: {'d_model': 64, 'd_ff': 128, 'e_layers': 1, 'memory_size': 64,
                 'bottleneck_dim': -1, 'dropout': 0.15, 'batch_size': 32,
                 'learning_rate': 0.001, 'lradj': 'type1', 'fusion_init': 3.0, 'rev_affine': 0},
            192: {'d_model': 64, 'd_ff': 128, 'e_layers': 1, 'memory_size': 64,
                  'bottleneck_dim': -1, 'dropout': 0.15, 'batch_size': 32,
                  'learning_rate': 0.001, 'lradj': 'type1', 'fusion_init': 3.0, 'rev_affine': 0},
            # 336/720: 默认平衡，关闭 rev_affine
            336: {'d_model': 64, 'd_ff': 128, 'e_layers': 1, 'memory_size': 64,
                  'bottleneck_dim': -1, 'dropout': 0.2, 'batch_size': 32,
                  'learning_rate': 0.001, 'lradj': 'type1', 'rev_affine': 0},
            720: {'d_model': 64, 'd_ff': 128, 'e_layers': 1, 'memory_size': 64,
                  'bottleneck_dim': 1, 'dropout': 0.45, 'batch_size': 32,
                  'learning_rate': 0.0003, 'lradj': 'type1', 'rev_affine': 0},
        }
    },
    'exchange': {
        'data_path': 'exchange_rate.csv',
        'enc_in': 8,
        'configs': {
            # [V7] 96/192步: 恢复 dropout=0.5
            96: {'d_model': 64, 'd_ff': 128, 'e_layers': 1, 'memory_size': 32,
                 'bottleneck_dim': -1, 'dropout': 0.5, 'batch_size': 32,
                 'learning_rate': 0.0001, 'lradj': '3'},
            192: {'d_model': 64, 'd_ff': 128, 'e_layers': 1, 'memory_size': 32,
                  'bottleneck_dim': -1, 'dropout': 0.5, 'batch_size': 32,
                  'learning_rate': 0.0001, 'lradj': '3'},
            # [V10] 336/720步: 关闭 CEA (`ablation_cea=1`)
            # [V12] 336步: 保持 rev_affine=0 (最佳MSE 0.33)
            336: {'d_model': 32, 'd_ff': 64, 'e_layers': 1, 'memory_size': 32,
                  'bottleneck_dim': 1, 'dropout': 0.65, 'batch_size': 32,
                  'learning_rate': 0.0005, 'lradj': 'type1', 'rev_affine': 0},
            # [V14] 720步: 恢复 dropout=0.6 (历史最佳 0.8558)
            # V13 (0.7) 导致欠拟合 (0.897)
            720: {'d_model': 32, 'd_ff': 64, 'e_layers': 1, 'memory_size': 32,
                  'bottleneck_dim': 1, 'dropout': 0.6, 'batch_size': 32,
                  'learning_rate': 0.0005, 'lradj': 'type1', 'rev_affine': 1},
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
    },
    'electricity': {
        'data_path': 'electricity.csv',
        'enc_in': 321,
        'configs': {
            # [V9] 深度优先策略 (Depth > Width)
            # 解决 V8 (d=128, layers=1) 失败的问题: 复杂序列需要深度
            # 保持 d_model=64 以配合 e_layers=2 避免 OOM
            96: {'d_model': 128, 'd_ff': 128, 'e_layers': 2, 'memory_size': 256,
                 'bottleneck_dim': 2, 'dropout': 0.1, 'batch_size': 16,
                 'learning_rate': 0.005, 'lradj': '3'},
            192: {'d_model': 128, 'd_ff': 128, 'e_layers': 2, 'memory_size': 256,
                  'bottleneck_dim': 2, 'dropout': 0.1, 'batch_size': 16,
                  'learning_rate': 0.005, 'lradj': '3'},
            # [V14] 336步: 回退 e_layers=1, d_model=64 (历史最佳 0.1930)
            # V12 (layers=2) 导致性能下降 (0.1999)
            336: {'d_model': 128, 'd_ff': 128, 'e_layers': 1, 'memory_size': 256,
                  'bottleneck_dim': 2, 'dropout': 0.1, 'batch_size': 16,
                  'learning_rate': 0.005, 'lradj': '3'},
            # [V14] 720步: 回退 e_layers=1, 但 d_model=128 (历史最佳 0.2267)
            # V12 (layers=2, d=64) 性能较差 (0.2372)
            720: {'d_model': 128, 'd_ff': 128, 'e_layers': 1, 'memory_size': 256,
                  'bottleneck_dim': 4, 'dropout': 0.1, 'batch_size': 16,
                  'learning_rate': 0.005, 'lradj': '3'},
        }
    },
}

# 预测长度
PRED_LENS = [96, 192, 336, 720]

# 通用参数
COMMON_PARAMS = {
    'seq_len': 96,
    'train_epochs': 20,
    'patience': 8,  # [关闭早停] 设为极大值
    'itr': 1,
    'num_workers': 8,  # 8-10 for 14 vCPUs
    'use_amp': False,   # [用户请求] 关闭混合精度，防止副作用
}

# Weather特殊参数
WEATHER_EPOCHS = 30


# ============ 工具函数 ============

def build_command(dataset, pred_len, config):
    """构建训练命令"""
    data_config = DATASET_CONFIGS[dataset]
    pred_config = config
    
    # 针对RTX 5090的 Batch Size 优化策略 V3 (回归精度优先)
    # [用户反馈] Batch Size=128 导致 ETTm1 性能下降 (0.318 -> 0.328)
    # 策略 Adjustment: 
    # 1. 小数据集 (ETT, Exchange) 回归 Batch Size = 32，确保迭代次数足够，优先保证精度。
    # 2. 只有大数据集 (Weather, Electricity) 才适当增大 Batch Size。
    
    base_batch_size = pred_config['batch_size']
    
    if dataset in ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', 'exchange']:
        # 强制回归到 32 (最佳 Baseline 配置)
        # 虽然 GPU 利用率低，但在小数据集上能获得更好的泛化性能
        optimized_batch_size = 32
    elif dataset == 'weather':
        # Weather 可以适当大一点
        optimized_batch_size = 64
    elif dataset == 'electricity':
        # [重要] Electricity 321维，显存压力大，使用配置中的 batch_size (16)
        optimized_batch_size = base_batch_size
    else:
        optimized_batch_size = base_batch_size * 2
        
    # 基础命令
    cmd = [
        'python', '-u', 'run_longExp.py',
        '--data', dataset,
        '--data_path', data_config['data_path'],
        '--model', 'FreDEA',
        '--model_id', f'{dataset}_{COMMON_PARAMS["seq_len"]}_{pred_len}',
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
        '--batch_size', str(optimized_batch_size), # 使用优化后的 Batch Size
        '--learning_rate', str(pred_config['learning_rate']),
        '--lradj', pred_config['lradj'],
        '--train_epochs', str(WEATHER_EPOCHS if dataset == 'weather' else COMMON_PARAMS['train_epochs']),
        '--patience', str(COMMON_PARAMS['patience']),
        '--itr', str(COMMON_PARAMS['itr']),
        '--num_workers', '4', # [优化] 降低 worker 数量，减少小数据集的 overhead
    ]
    
    # [新增] 支持 CEA 消融实验配置
    if 'ablation_cea' in pred_config and pred_config['ablation_cea']:
        cmd.append('--ablation_cea')
        cmd.append('1')
        
    # [新增 V11] 支持 RevIN Affine 配置
    if 'rev_affine' in pred_config:
        cmd.append('--rev_affine')
        cmd.append(str(pred_config['rev_affine']))
    
    # [新增 V26] 支持 Fusion Init 配置 (ETTh2 优化)
    if 'fusion_init' in pred_config:
        cmd.append('--fusion_init')
        cmd.append(str(pred_config['fusion_init']))
    
    if COMMON_PARAMS['use_amp']:
        cmd.append('--use_amp')
    
    return cmd


def extract_results(output):
    """从命令输出中提取MSE/MAE/RMSE结果"""
    # 匹配模式: mse:0.xxx, mae:0.xxx
    mse_pattern = r'mse[:\s]+([0-9.]+)'
    mae_pattern = r'mae[:\s]+([0-9.]+)'
    
    mse_match = re.search(mse_pattern, output, re.IGNORECASE)
    mae_match = re.search(mae_pattern, output, re.IGNORECASE)
    
    mse = float(mse_match.group(1)) if mse_match else None
    mae = float(mae_match.group(1)) if mae_match else None
    
    return mse, mae


def run_experiment(dataset, pred_len):
    """运行单个实验"""
    config = DATASET_CONFIGS[dataset]['configs'][pred_len]
    cmd = build_command(dataset, pred_len, config)
    
    print(f"\n{'='*80}")
    print(f"运行实验: {dataset} - Pred {pred_len}")
    print(f"{'='*80}")
    print("命令: " + ' '.join(cmd))
    print()
    
    try:
        # 使用 Popen 来实时流式输出
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # 行缓冲
            universal_newlines=True
        )
        
        full_output = []
        
        # 实时读取输出
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.strip())  # 实时打印到控制台
                full_output.append(line)
        
        # 等待进程结束
        return_code = process.wait()
        full_output_str = "".join(full_output)
        
        if return_code != 0:
            print(f"❌ 实验失败，返回码: {return_code}")
            return {
                'dataset': dataset,
                'pred_len': pred_len,
                'mse': None,
                'mae': None,
                'status': 'failed',
                'error': f'Process exited with code {return_code}'
            }
        
        # 提取结果
        mse, mae = extract_results(full_output_str)
        
        if mse is not None and mae is not None:
            print(f"✅ 实验完成: MSE={mse:.6f}, MAE={mae:.6f}")
            return {
                'dataset': dataset,
                'pred_len': pred_len,
                'mse': mse,
                'mae': mae,
                'status': 'success',
                'command': ' \\\n  '.join([cmd[i] if i == 0 else f'--{cmd[i]} {cmd[i+1]}' if i % 2 == 1 and i < len(cmd)-1 else '' for i in range(0, len(cmd), 2)])
            }
        else:
            print(f"❌ 结果提取失败")
            return {
                'dataset': dataset,
                'pred_len': pred_len,
                'mse': None,
                'mae': None,
                'status': 'failed',
                'error': 'Result extraction failed'
            }
            
    except KeyboardInterrupt:
        print("\n❌ 用户中断实验")
        return {
            'dataset': dataset,
            'pred_len': pred_len,
            'mse': None,
            'mae': None,
            'status': 'interrupted'
        }
    except Exception as e:
        print(f"❌ 实验执行错误: {str(e)}")
        return {
            'dataset': dataset,
            'pred_len': pred_len,
            'mse': None,
            'mae': None,
            'status': 'error',
            'error': str(e)
        }


def generate_summary_table(results):
    """生成Markdown格式的汇总表格"""
    
    # 按数据集分组
    datasets = ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', 'exchange', 'weather', 'electricity']
    
    summary = []
    summary.append("# FreDEA 实验结果汇总")
    summary.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 主结果表格
    summary.append("## 主要结果 (MSE)")
    summary.append("\n| 数据集 | 96 | 192 | 336 | 720 |")
    summary.append("|--------|---------|---------|---------|---------|")
    
    for dataset in datasets:
        row = [dataset]
        for pred_len in PRED_LENS:
            result = next((r for r in results if r['dataset'] == dataset and r['pred_len'] == pred_len), None)
            if result and result['mse'] is not None:
                row.append(f"{result['mse']:.4f}")
            else:
                row.append("N/A")
        summary.append("| " + " | ".join(row) + " |")
    
    # MAE表格
    summary.append("\n## 主要结果 (MAE)")
    summary.append("\n| 数据集 | 96 | 192 | 336 | 720 |")
    summary.append("|--------|---------|---------|---------|---------|")
    
    for dataset in datasets:
        row = [dataset]
        for pred_len in PRED_LENS:
            result = next((r for r in results if r['dataset'] == dataset and r['pred_len'] == pred_len), None)
            if result and result['mae'] is not None:
                row.append(f"{result['mae']:.4f}")
            else:
                row.append("N/A")
        summary.append("| " + " | ".join(row) + " |")
    
    # 详细结果
    summary.append("\n## 详细结果")
    
    for result in results:
        if result['status'] == 'success':
            summary.append(f"\n### {result['dataset']} - Pred {result['pred_len']}")
            summary.append(f"- **MSE**: {result['mse']:.6f}")
            summary.append(f"- **MAE**: {result['mae']:.6f}")
            summary.append(f"- **状态**: ✅ 成功")
        else:
            summary.append(f"\n### {result['dataset']} - Pred {result['pred_len']}")
            summary.append(f"- **状态**: ❌ {result['status']}")
            if 'error' in result:
                summary.append(f"- **错误**: {result['error']}")
    
    # 配置表格
    summary.append("\n## 超参数配置")
    summary.append("\n| 数据集 | Pred | d_model | e_layers | bottleneck | dropout | batch_size | lr |")
    summary.append("|--------|------|---------|----------|------------|---------|------------|-----|")
    
    for dataset in datasets:
        for pred_len in PRED_LENS:
            if pred_len in DATASET_CONFIGS[dataset]['configs']:
                config = DATASET_CONFIGS[dataset]['configs'][pred_len]
                summary.append(
                    f"| {dataset} | {pred_len} | {config['d_model']} | "
                    f"{config['e_layers']} | {config['bottleneck_dim']} | "
                    f"{config['dropout']} | {config['batch_size']} | {config['learning_rate']} |"
                )
    
    return "\n".join(summary)


# ============ 主函数 ============

def main():
    """主函数"""
    print("="*80)
    print("FreDEA 批量实验运行脚本")
    print("="*80)
    print(f"数据集数量: {len(DATASET_CONFIGS)}")
    print(f"预测长度: {PRED_LENS}")
    print(f"总实验数: {len(DATASET_CONFIGS) * len(PRED_LENS)}")
    print("="*80)
    
    results = []
    
    # 运行所有实验
    for dataset in DATASET_CONFIGS.keys():
        for pred_len in PRED_LENS:
            if pred_len in DATASET_CONFIGS[dataset]['configs']:
                result = run_experiment(dataset, pred_len)
                results.append(result)
    
    # 生成汇总报告
    print("\n" + "="*80)
    print("生成汇总报告...")
    print("="*80)
    
    summary = generate_summary_table(results)
    
    # 保存到文件
    output_file = f'FreDEA_Results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"\n✅ 汇总报告已保存到: {output_file}")
    
    # 打印到控制台
    print("\n" + "="*80)
    print("汇总报告预览:")
    print("="*80)
    print(summary)
    
    # 统计信息
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = len(results) - success_count
    
    print("\n" + "="*80)
    print(f"实验完成统计:")
    print(f"  成功: {success_count}/{len(results)}")
    print(f"  失败: {failed_count}/{len(results)}")
    print("="*80)


if __name__ == '__main__':
    main()
