#!/usr/bin/env python
"""
FreDEA 快速测试脚本 - 智能同步版

直接复用 run_all_experiments.py 的配置和逻辑，确保命令完全一致。
只测试核心数据集和步长。
"""

import subprocess
import re
from datetime import datetime
import sys
import os

# 导入配置和工具函数
try:
    from run_all_experiments import DATASET_CONFIGS, build_command, extract_results
except ImportError:
    print("❌ 错误: 无法导入 run_all_experiments.py，请确保文件在同一目录下")
    sys.exit(1)

# 定义要测试的子集 (按优先级排序)
# 格式: '数据集名称': [预测步长列表]
# [V15 验证] DPEM 风格 Feature-Time Fusion 架构全面验证
from collections import OrderedDict
TEST_TARGETS = OrderedDict([
    # 核心验证: 全数据集全步长
    ('ETTh1', [96, 192, 336, 720]),
    ('ETTh2', [96, 192, 336, 720]),
    ('ETTm1', [96, 192, 336, 720]),
    ('ETTm2', [96, 192, 336, 720]),
    ('exchange', [96, 192, 336, 720]),
    ('weather', [96, 192, 336, 720]),
    ('electricity', [96, 192, 336, 720]),
])

def main():
    print("="*80)
    print("FreDEA 快速测试脚本 (Sync with All Experiments)")
    print("="*80)
    
    tasks = []
    for dataset, pred_lens in TEST_TARGETS.items():
        for pl in pred_lens:
            tasks.append((dataset, pl))
            
    print(f"计划执行任务数: {len(tasks)}")
    print("="*80)
    
    results = []
    
    for dataset, pred_len in tasks:
        # 动态构建命令，确保与主脚本一致
        try:
            config = DATASET_CONFIGS[dataset]['configs'][pred_len]
            cmd_list = build_command(dataset, pred_len, config)
            # 转为字符串用于打印和执行
            cmd_str = ' '.join(cmd_list)
            
            task_name = f"{dataset}_{pred_len}"
            
            print(f"\n{'='*80}")
            print(f"运行: {task_name}")
            print(f"{'='*80}")
            print(f"Command: {cmd_str}")
            
            # 使用 Popen 来实时流式输出
            process = subprocess.Popen(
                cmd_list,
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
                # 记录失败但继续执行后续实验
                continue

            mse, mae = extract_results(full_output_str)
            
            if mse is not None and mae is not None:
                results.append({
                    'name': task_name,
                    'mse': mse,
                    'mae': mae,
                    'cmd': cmd_str
                })
                print(f"✅ 完成: MSE={mse:.6f}, MAE={mae:.6f}")
            else:
                print(f"❌ 结果提取失败")
                
        except KeyboardInterrupt:
            print("\n❌ 用户中断实验")
            sys.exit(0)
        except Exception as e:
            print(f"❌ 错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 生成汇总
    print("\n" + "="*80)
    print("汇总结果")
    print("="*80)
    
    output_lines = []
    output_lines.append(f"# FreDEA 快速测试结果\n")
    output_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    for r in results:
        output_lines.append(f"## {r['name']}")
        output_lines.append(f"```bash")
        output_lines.append(r['cmd'])
        output_lines.append(f"```")
        output_lines.append(f"**结果**: mse:{r['mse']}, mae:{r['mae']}\n")
    
    summary = "\n".join(output_lines)
    
    # 保存
    filename = f'Quick_Test_Results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(summary)
    print(f"\n✅ 结果已保存到: {filename}")


if __name__ == '__main__':
    main()
