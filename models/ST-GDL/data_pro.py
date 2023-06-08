# %%
from tqdm import tqdm

# %%
data_path = 'data_ori/'
data_path_pro = 'data_pro/'
import os
if not os.path.exists(data_path_pro):
    os.makedirs(data_path_pro)
import numpy as np
import dgl

# Path: data_pro.ipynb
def load_data(data_path, data_path_pro):
    od_matrix = np.load(data_path + 'od_matrix_700.npy')[:220]
    od_matrix = od_matrix.astype(np.float32)
    # 读取eoGraph.dgl
    geoGraph = dgl.load_graphs(data_path + 'geoGraph.dgl')[0][0]
    return od_matrix, geoGraph

# Path: data_pro.ipynb
def get_adj(geoGraph):
    adj = geoGraph.adjacency_matrix()
    adj = adj.to_dense()
    adj = adj.numpy()
    return adj

print('loading data...')
od_matrix, geoGraph = load_data(data_path, data_path_pro)
print('loading data finished')


# %%
trend_len = 7*24
period_len = 24
closeness_len = 3
prediction_len = 12
slot_num = 12

# %%
# 将OD矩阵处理成三通道输入
# Path: data_pro.ipynb
def process_od(od_matrix):
    data_closeness = []
    data_trend = []
    data_period = []
    num_hour = od_matrix.shape[0]
    batch_size = num_hour - trend_len - prediction_len + 1
    data_closeness = np.zeros((batch_size, closeness_len, od_matrix.shape[1], od_matrix.shape[2]))
    data_trend = np.zeros((batch_size, slot_num, od_matrix.shape[1], od_matrix.shape[2]))
    data_period = np.zeros((batch_size, slot_num, od_matrix.shape[1], od_matrix.shape[2]))
    data_prediction = np.zeros((batch_size, prediction_len, od_matrix.shape[1], od_matrix.shape[2]))
    for curh in tqdm(range(num_hour)):
        if (curh - trend_len<0) or (curh + prediction_len>num_hour):
            continue
        data_closeness[curh-trend_len] = od_matrix[curh-closeness_len:curh]
        data_trend[curh-trend_len] = od_matrix[curh-trend_len:curh-trend_len+slot_num]
        data_period[curh-trend_len] = od_matrix[curh-period_len:curh-period_len+slot_num]
        data_prediction[curh-trend_len] = od_matrix[curh:curh+prediction_len]
    return data_closeness, data_trend, data_period,data_prediction
data_closeness, data_trend, data_period,data_prediction = process_od(od_matrix)


# %%
# 保存为 data_closeness, data_trend, data_period,data_prediction 为npz
# np.savez(data_path_pro + 'data.npz', closeness = data_closeness, trend = data_trend, period = data_period, prediction = data_prediction)
np.save(data_path_pro + 'data_closeness.npy', data_closeness)
np.save(data_path_pro + 'data_trend.npy', data_trend)
np.save(data_path_pro + 'data_period.npy', data_period)
np.save(data_path_pro + 'data_prediction.npy', data_prediction)

# %%


# %%



