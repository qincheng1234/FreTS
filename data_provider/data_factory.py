from data_provider.data_loader import Dataset_Covid, Dataset_Custom, Dataset_Pred, Dataset_Custom_, Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Solar
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,   # 使用标准 ETT 小时数据集类
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute, # 使用标准 ETT 分钟数据集类 (StandardScaler + 12/4/4 月划分)
    'ETTm2': Dataset_ETT_minute,
    'traffic': Dataset_Custom,
    'electricity': Dataset_Custom_,
    'exchange': Dataset_Custom_,
    'weather': Dataset_Custom_,
    'solar': Dataset_Solar,    # Solar-Energy: 137变量太阳能数据
    'covid': Dataset_Covid,
    'ECG': Dataset_Custom_,
    'metr': Dataset_Custom_,
}


def register_dataset(name: str, dataset_cls):
    """在运行时注册自定义数据集类，解决白盒审查卡点 D。

    传入 data_factory.data_dict 中没有的数据集名称时，
    可通过本函数在实验脚本中动态注册，而无需修改源文件。

    Parameters
    ----------
    name : str
        数据集标识符，与 --data 命令行参数对应。
    dataset_cls : type
        实现了与 Dataset_Custom 相同接口的 PyTorch Dataset 类，
        即构造函数接受 (root_path, data_path, flag, size,
        features, target, timeenc, freq, train_only) 参数。

    Raises
    ------
    TypeError
        当 dataset_cls 不是 type（类）时抛出。

    Examples
    --------
    >>> from data_provider.data_factory import register_dataset
    >>> from data_provider.data_loader import Dataset_Custom_
    >>>
    >>> # 注册新数据集（复用已有 loader）
    >>> register_dataset('my_dataset', Dataset_Custom_)
    >>>
    >>> # 或注册完全自定义的 Dataset
    >>> class MyDataset(torch.utils.data.Dataset):
    ...     def __init__(self, root_path, data_path, flag, size,
    ...                  features, target, timeenc, freq, train_only):
    ...         ...
    >>> register_dataset('my_dataset', MyDataset)
    """
    if not isinstance(dataset_cls, type):
        raise TypeError(f"dataset_cls must be a class, got {dataset_cls!r}")
    data_dict[name] = dataset_cls


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    train_only = args.train_only

    if flag == 'test':
        shuffle_flag = False
        drop_last = False  # 修复: 不丢弃最后一个不完整的 batch，确保小测试集能被正确评估
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        train_only=train_only
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
