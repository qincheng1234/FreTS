import pandas as pd
import numpy as np

def format_value(val, is_best, is_second):
    """根据是否是最优或次优结果格式化 LaTeX 字符串"""
    if pd.isna(val):
        return "-"
    
    # 格式化数字，保留3位小数
    val_str = f"{val:.3f}"
    
    if is_best:
        # 红色 + 加粗
        return f"\\color{{red}}\\textbf{{{val_str}}}"
    elif is_second:
        # 蓝色 + 下划线
        return f"\\color{{blue}}\\underline{{{val_str}}}"
    else:
        return val_str

def generate_latex_table(file_path):
    # 读取 Excel，header=[0, 1] 处理双层表头
    # 第一行是模型名 (FreDEA, DPEM...), 第二行是指标名 (MSE, MAE...)
    try:
        df = pd.read_excel(file_path, header=[0, 1])
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{file_path}'，请确认文件路径。")
        return

    # 预处理：填充 Dataset 列的合并单元格 (NaN -> 前值)
    # 假设第一列是 Dataset，第二列是 Length
    df.iloc[:, 0] = df.iloc[:, 0].ffill()
    
    # 获取所有模型列（排除前两列 Dataset 和 Length）
    # 假设从第3列开始是数据 (索引为2)
    metric_data = df.iloc[:, 2:]
    
    # 初始化一个同样大小的 DataFrame 来存放 LaTeX 字符串
    latex_df = pd.DataFrame(index=df.index, columns=metric_data.columns)

    # 遍历每一行进行比较
    for idx, row in metric_data.iterrows():
        # 分离 MSE 和 MAE
        # 假设列顺序是: Model1_MSE, Model1_MAE, Model2_MSE, Model2_MAE ...
        # 使用切片：从0开始步长为2是MSE，从1开始步长为2是MAE
        mses = row.iloc[0::2]
        maes = row.iloc[1::2]

        # --- 处理 MSE ---
        # 找到最小值和次小值
        sorted_mse = mses.sort_values(ascending=True).values
        best_mse = sorted_mse[0]
        second_mse = sorted_mse[1] if len(sorted_mse) > 1 else np.inf

        # 格式化这一行的 MSE
        for i, val in enumerate(mses):
            col_idx = i * 2 # 原表中的列索引
            is_best = np.isclose(val, best_mse)
            is_second = np.isclose(val, second_mse)
            latex_df.iloc[idx, col_idx] = format_value(val, is_best, is_second)

        # --- 处理 MAE ---
        sorted_mae = maes.sort_values(ascending=True).values
        best_mae = sorted_mae[0]
        second_mae = sorted_mae[1] if len(sorted_mae) > 1 else np.inf

        for i, val in enumerate(maes):
            col_idx = i * 2 + 1 # 原表中的列索引
            is_best = np.isclose(val, best_mae)
            is_second = np.isclose(val, second_mae)
            latex_df.iloc[idx, col_idx] = format_value(val, is_best, is_second)

    # --- 开始构建 LaTeX 字符串 ---
    
    # 获取模型名称列表 (用于表头)
    # df.columns 类似 [('Dataset', 'Unnamed'), ('Length', 'Unnamed'), ('FreDEA', 'MSE'), ('FreDEA', 'MAE')...]
    # 我们需要提取模型名
    cols = df.columns[2:]
    model_names = []
    for i in range(0, len(cols), 2):
        model_names.append(cols[i][0]) # 获取模型名，如 FreDEA

    # 1. 表头部分
    num_models = len(model_names)
    latex_code = []
    latex_code.append("\\begin{table*}[t]")
    latex_code.append("  \\centering")
    latex_code.append("  \\resizebox{\\textwidth}{!}{")
    
    # 动态生成列格式: l|c|cc|cc...
    col_format = "l|c|" + "cc|" * (num_models - 1) + "cc"
    latex_code.append(f"  \\begin{{tabular}}{{{col_format}}}")
    latex_code.append("    \\toprule")
    
    # 生成第一行表头: Dataset & Len & Model1 & Model2 ...
    header_row1 = "\\multirow{2}{*}{Dataset} & \\multirow{2}{*}{Len} & "
    header_row1 += " & ".join([f"\\multicolumn{{2}}{{c|}}{{\\textbf{{{name}}}}}" if i < num_models-1 else f"\\multicolumn{{2}}{{c}}{{\\textbf{{{name}}}}}" for i, name in enumerate(model_names)])
    header_row1 += " \\\\"
    latex_code.append("    " + header_row1)
    
    # 生成 cmidrule 线
    cmidrules = ""
    start_col = 3
    for _ in range(num_models):
        cmidrules += f"\\cmidrule(lr){{{start_col}-{start_col+1}}} "
        start_col += 2
    latex_code.append("    " + cmidrules)
    
    # 生成第二行表头: & & MSE & MAE & MSE & MAE ...
    header_row2 = "    & " + " & ".join(["MSE & MAE"] * num_models) + " \\\\"
    latex_code.append(header_row2)
    latex_code.append("    \\midrule")

    # 2. 数据体部分
    # 按 Dataset 分组处理
    datasets = df.iloc[:, 0].unique()
    
    for ds_idx, dataset in enumerate(datasets):
        ds_rows = df[df.iloc[:, 0] == dataset]
        latex_rows_segment = latex_df.loc[ds_rows.index]
        
        # 每一组的第一行，打印 Dataset 名字
        first_row = True
        num_rows = len(ds_rows)
        
        for i in range(num_rows):
            original_row = ds_rows.iloc[i]
            formatted_values = latex_rows_segment.iloc[i].tolist()
            length_val = original_row.iloc[1] # Length 列
            
            # 这一行的数据字符串
            data_str = " & ".join(formatted_values)
            
            if first_row:
                # 第一行显示 Dataset 名字，使用 multirow
                latex_code.append(f"    \\multirow{{{num_rows}}}{{*}}{{{dataset}}} & {length_val} & {data_str} \\\\")
                first_row = False
            else:
                latex_code.append(f"     & {length_val} & {data_str} \\\\")
        
        # 每组 Dataset 结束后加分割线 (如果是最后一组则不加，由 bottomrule 处理)
        if ds_idx < len(datasets) - 1:
            latex_code.append(f"    \\cmidrule{{1-{2 + num_models*2}}}")

    # 3. 结尾部分
    latex_code.append("    \\bottomrule")
    latex_code.append("  \\end{tabular}")
    latex_code.append("  }")
    latex_code.append("  \\caption{Multivariate time series forecasting results. \\color{red}\\textbf{Red Bold} indicates best, \\color{blue}\\underline{Blue Underline} indicates second best.}")
    latex_code.append("  \\label{tab:main_results}")
    latex_code.append("\\end{table*}")

    # 输出结果
    print("\n" + "="*20 + " LaTeX Code " + "="*20 + "\n")
    final_latex = "\n".join(latex_code)
    print(final_latex)
    
    # 也可以写入文件
    with open("table_output.tex", "w", encoding="utf-8") as f:
        f.write(final_latex)
    print("\n" + "="*50)
    print("代码已生成并保存为 'table_output.tex'")

# 运行函数
if __name__ == "__main__":
    generate_latex_table("实验数据.xlsx")