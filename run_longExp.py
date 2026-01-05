import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Linear family for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, default=1, help='status') # 状态：1为训练模式，0为测试/预测模式
parser.add_argument('--train_only', type=bool, default=False, help='perform training on full input dataset without validation and testing') # 仅训练：是否用全部数据训练而不进行验证和测试
parser.add_argument('--model_id', type=str, default='ETTm1', help='model id') # 模型ID：用于标识实验，通常包含数据集名称
parser.add_argument('--model', type=str, default='FreLinear',
                    help='model name, options: [NLinear, DLinear, FreLinear]') # 模型名称：选择要运行的模型，如 FreLinear, DLinear 等

# data loader
parser.add_argument('--data', type=str, default='ETTm1', help='dataset type') # 数据集类型：对应 data_factory 中的 key，指定使用哪种数据集类
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file') # 数据根目录：数据集文件所在的文件夹路径
parser.add_argument('--data_path', type=str, default='ETTm1.csv', help='data file') # 数据文件：具体的数据集文件名（csv格式）
parser.add_argument('--channel_independence', type=int, default=0, help='1: channel dependence 0: channel independence') # 通道独立性：1为开启通道依赖（学习变量间关系），0为独立（各变量单独预测，默认）
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate') # 预测任务类型：M=多变量预测多变量，S=单变量预测单变量，MS=多变量预测单变量（用多个特征预测目标列）
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task') # 目标列：在 S 或 MS 任务中，指定要去预测的那一列的名字
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h') # 数据频率：时间特征编码的频率，如 'h' 代表按小时。用于生成时间相关的 embedding
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints') # 模型保存路径：训练好的模型权重保存位置

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length') # 输入序列长度：模型往回看多少个时间步（Lookback window）
parser.add_argument('--label_len', type=int, default=48, help='start token length') # 标签长度：Informer类模型专用的 Decoder 启动 token 长度（FreTS其实用不到，但为了兼容保留）
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length') # 预测序列长度：需要预测未来多少个时间步

# DLinear
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually') # DLinear专用：是否对每个变量单独使用一个 Linear 层（互不共享权重）
# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding') # Embedding类型：0是默认时间特征编码
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # Encoder输入维度：即输入数据的通道数/变量数（如 ETTm1 有7个变量）
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size') # Decoder输入维度：同上，通常等于 enc_in
parser.add_argument('--c_out', type=int, default=7, help='output size') # 输出维度：需要预测的变量数量
parser.add_argument('--d_model', type=int, default=512, help='dimension of model') # 模型隐藏层维度：内部特征向量的大小（在FreTS中对应 embed_size）
# === [新增] 频域正则化参数 ===
parser.add_argument('--reg_lambda', type=float, default=0.01, help='frequency regularization weight') # 频域正则化权重：控制频域损失的强度
parser.add_argument('--wavelet', type=str, default='haar', help='wavelet type: haar or db2') # 小波类型：haar 或 db2（保留兼容性）
parser.add_argument('--n_bins', type=int, default=8, help='number of bins for BSP Loss') # BSP分箱数：建议8
# ===========================
parser.add_argument('--n_heads', type=int, default=8, help='num of heads') # 多头注意力头数：Transformer类模型用，FreTS不用
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers') # Encoder层数：Transformer类模型用
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers') # Decoder层数：Transformer类模型用
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn') # FFN维度：FeedForward层的大小
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average') # 移动平均窗口：DLinear等分解模型中用于提取趋势的窗口大小
parser.add_argument('--factor', type=int, default=1, help='attn factor') # Attention因子：Informer中ProbSparse Attention的采样因子
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True) # 蒸馏操作：是否在 Encoder 层之间使用卷积池化减少维度（默认开启）
parser.add_argument('--dropout', type=float, default=0.05, help='dropout') # Dropout比率：防止过拟合
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]') # 时间编码方式：timeF为基于频率通过模型生成，fixed为正弦余弦编码，learned为可学习Embedding
parser.add_argument('--activation', type=str, default='gelu', help='activation') # 激活函数
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder') # 输出Attention：是否在输出中包含注意力权重矩阵（用于可视化）
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data') # 执行预测：训练测试完是否由模型直接输出对未知未来的预测文件

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers') # 数据加载线程数：Windows下建议设为 0
parser.add_argument('--itr', type=int, default=1, help='experiments times') # 实验重复次数：跑几次取平均
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs') # 训练轮数：总共训练多少 Epoch
parser.add_argument('--batch_size', type=int, default=8, help='batch size of train input data') # 批次大小：显存够可以调大
parser.add_argument('--patience', type=int, default=3, help='early stopping patience') # 早停耐心值：验证集Loss多少个Epoch不下降就停止训练
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate') # 学习率
parser.add_argument('--des', type=str, default='Exp', help='exp description') # 实验描述：会记录在日志里
parser.add_argument('--loss', type=str, default='mse', help='loss function') # 损失函数：默认均方误差 MSE
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate') # 学习率衰减策略：type1/type2等不同的衰减方式
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False) # 混合精度训练：是否开启 AMP 加速

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu') # 使用GPU：是否使用显卡
parser.add_argument('--gpu', type=int, default=0, help='gpu') # GPU编号：指定使用哪块显卡
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False) # 多卡训练：是否使用多块显卡
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus') # 多卡ID：指定多块显卡的ID列表
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage') # 计算FLOPs：是否计算模型浮点运算量

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        if not args.train_only:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                  args.model,
                                                                                                  args.data,
                                                                                                  args.features,
                                                                                                  args.seq_len,
                                                                                                  args.label_len,
                                                                                                  args.pred_len,
                                                                                                  args.d_model,
                                                                                                  args.n_heads,
                                                                                                  args.e_layers,
                                                                                                  args.d_layers,
                                                                                                  args.d_ff,
                                                                                                  args.factor,
                                                                                                  args.embed,
                                                                                                  args.distil,
                                                                                                  args.des, ii)

    exp = Exp(args)  # set experiments

    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)
    else:
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
    torch.cuda.empty_cache()
