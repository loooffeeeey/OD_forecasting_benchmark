import numpy as np
import torch
# 计算pearson相关系数
def cal_pearson_vec(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_std = np.std(x)
    y_std = np.std(y)
    x = (x - x_mean) / x_std
    y = (y - y_mean) / y_std
    pearson = np.sum(x * y) / ((len(x) - 1)+1e-6)
    return pearson


def cal_pearson(data):
    # data: (T, 27, O, D)
    # return: (T, O, O)
    # data 为张量，在dim = -1上求和  交换dim = 1和dim = 2的位置
    data = torch.sum(data, dim=-1).transpose(1, 2)
    # (T, O, 27)
    T, O, _ = data.shape
    # 计算矩阵各行的pearson相关系数
    pearson_mat = torch.zeros((T, O, O))
    for t in range(T):
        # 换为张量重写上式
        pearson_mat[t] = torch.exp(torch.abs(torch.corrcoef(data[t]))-1)
    return pearson_mat
