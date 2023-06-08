import numpy as np
import os
import datetime
from libs import utils, graph_common
import pickle
import pandas as pd
'''
函数说明：分三类函数
一、原始数据的预处理
1-1 
二、所有模型共享的数据处理
2-1 
三、模型独自所需的数据处理
3-1 
'''
######################################## Data process for original ########################################

def load_original_data(args, log):
    data_file = os.path.join(args.dataset_dir, 'original', args.data_file)
    M = np.load(data_file)["matrix"]
    M = M.astype(np.float32)
    # Days, T1, N, _, T2 = 30, 20, 10, 10, 20
    # M = np.random.normal(size=(Days, T1, N, _, T2)).astype(np.float32)
    utils.log_string(log, 'original data shape: ' + str(M.shape) + ' dtype: ' + str(
        M.dtype))
    return M

'''
产生周特征
'''
def week_transform(args,log):
    station_txt_file = os.path.join(args.dataset_dir, 'original', args.day_index)
    day_list = []
    with open(station_txt_file, "r") as fin:
        for line in fin.readlines():
            line = line[:-1]  # 去掉\n
            day_list.append(line)
    dayofweek = np.array([int(datetime.datetime.strptime(day_list[i], '%Y-%m-%d').strftime("%w")) for i in
                 range(len(day_list))])
    utils.log_string(log, 'week features shape: ' + str(dayofweek.shape) + ' dtype: ' + str(
        dayofweek.dtype))
    return dayofweek

'''
dayofweek: list
'''
def add_time(T1, N, dayofweek):
    Days = len(dayofweek)
    Dayoftime = np.tile(np.reshape(np.array([t for t in range(T1)]), (1, T1, 1, 1)),(Days, 1, N, 1)) #(Days,T1,N,1)
    Dayofweek = np.tile(np.reshape(np.array(dayofweek), (Days, 1, 1, 1)),(1, T1, N, 1)) # (Days,T1,N,1)
    Dayofyear = np.tile(np.reshape(np.arange(len(dayofweek)), (Days, 1, 1, 1)),(1, T1, N, 1)) # (Days,T1,N,1)
    output = np.concatenate([Dayoftime, Dayofweek, Dayofyear], axis=-1) # (Days,T1,N,3)
    return output

'''
列表聚合
'''
def process_all(args, all_data, data_names,data_mode):
    dir_name, data_path, log = utils.path(args,data_mode)
    for i in range(len(all_data)):
        # print(i,data_names[i])
        all_data[i] = np.stack(all_data[i], axis=0).astype(np.float32)

        message = '%s_%s_%s shape: %s dtype: %s' % (data_names[i], args.data_type, dir_name, str(all_data[i].shape), str(all_data[i].dtype))
        utils.log_string(log, message)
    return all_data

'''
功能：保存指定输入模式的数据，跟模型无关
data: 被保存的数据列表
data_names: 数据名称
data_path: 保存路径
'''
def save_feature_data(args, all_data, data_names,data_mode):
    data_path = os.path.join(args.dataset_dir, args.model, data_mode, utils.input_name(args)) # 路径
    print("data_path:",data_path)
    for i in range(len(data_names)):
        save_name = os.path.join(data_path, data_names[i] + '_' + args.data_type + '.npz')
        if all_data[i].shape[1]!=0:
            np.savez_compressed(save_name, all_data[i])

'''
功能：装载指定的数据
返回：一个列表，元素是x,x1,..,y
'''
def load(args, log, dir_name):
    path = os.path.join(args.dataset_dir, 'all_models', args.data_mode, dir_name) # 存放的文件夹路径
    results = []
    data_names_list = args.data_names.split("-")
    for data_name in data_names_list:
        file_name = data_name + '_' + args.data_type + '.npz'
        file_path = os.path.join(path, file_name)
        result = np.load(file_path)['arr_0']
        utils.log_string(log, '%s %s' % (file_name, str(result.shape)))
        results.append(result)
    return results


# Time = add_time(T1,N,dayofweek)  # (D,T1,N,T)
def sequence_OD_data_PDW3(args, M_OD, Time, dow_num=7,data_mode="data1"):
    # 前P个时间段,前D天,前W周
    P = int(args.input_steps[0])
    D = int(args.input_steps[1])
    W = int(args.input_steps[2])

    T1 = M_OD.shape[1]
    M = np.sum(M_OD, axis=-1)  # (D,T1,N,N)
    M1 = np.sum(M_OD[...,:T1], axis=1)  # (D,N_in,N_out,T1)
    M1 = np.transpose(M1,[0,3,2,1])# (D,T1,N_out,N_in)

    flow_in = np.sum(M, axis=-1)  # (D,T1,N,N)
    flow_out = np.sum(M1, axis=-1)  # (D,T1,N)

    Days, T1, N, _ = M.shape

    if args.data_type == "In" or args.data_type == "Out":
        M = np.expand_dims(M, axis=-1)  # (Days, T1, N, 1)
        M1 = np.expand_dims(M1, axis=-1)  # (Days, T1, N, 1)
    M = np.concatenate([M, Time], axis=-1)  # (Days, T1, N, F+T)
    # M1 = np.concatenate([M1, Time], axis=-1)  # (Days, T1, N, F+T)

    # all_data = [[], [], [], [], [], []]
    all_data = [[] for i in range(9)]
    for j in range(Days):
        if j - dow_num * W < 0: continue
        weeks = [j - dow_num * w for w in range(1, W + 1)]  # [j-7,...,j-7W]
        if j - D < 0: continue
        for i in range(T1):
            if i - P < 0: continue
            y = M[j, i, ...]  # 第j天第i个时间段，(N,F)
            x_D = M[j - D:j, i, ...]  # 前D天第i个时间段，(D,N,F)
            x_W = M[weeks, i, ...]  # 前W周同一天第i个时间段(W,N,F)
            x_P = M[j, i - P:i, ...]  # 同一天，前P个时间段，全天候出闸，(P,N,F)
            in_flow = flow_in[j, i - P:i, ...] # 入站客流
            all_data[0].append(x_P)  # (M,P,N,F)
            all_data[1].append(x_D)  # (M,D,N,F)
            all_data[2].append(x_W)  # (M,W,N,F)
            all_data[3].append(y)  # (M,N,F)
            all_data[4].append(in_flow)  # (M,N)


            x_P_before = np.sum(M_OD[j, i - P:i, ..., :i], axis=-1)  # 同一天，前P个时间段，预测时间段前出闸，(P,N,N)
            x_P_after = np.sum(M_OD[j, i - P:i, ..., i:], axis=-1)  # 同一天，前P个时间段，预测时间段后出闸，(P,N,N)
            time = Time[j, i - P:i, ...]  # (P,N,3)
            x_P_before = np.concatenate([x_P_before, time], axis=-1)  # (P,N,N+3)
            x_P_after = np.concatenate([x_P_after, time], axis=-1)  # (P,N,N+3)
            out_flow = flow_out[j, i - P:i, ...] # 出站客流

            # x_P_before_out = np.transpose(np.sum(M_OD[j, :i, ..., i - P:i], axis=0),[2,1,0])  # 同一天，当天进站，在[i-p, i]出站的人数，(P,N,N)
            x_P_before_out = np.transpose(np.sum(M_OD[j, :i, ..., i - P:i], axis=0),
                                          [2, 0, 1])  # 同一天，当天进站，在[i-p, i]出站的人数，(P,N,N)
            all_data[5].append(x_P_before)
            all_data[6].append(x_P_after)
            all_data[7].append(out_flow)
            all_data[8].append(x_P_before_out)

    data_names = ['samples_all', 'samples_D', 'samples_W', 'labels','in_flow', 'samples_P', 'samples_after','out_flow','od_out']
    all_data = process_all(args, all_data, data_names, data_mode)
    save_feature_data(args, all_data, data_names, data_mode)
    return all_data



'''
功能：求其元素级别的均值与方差
保存好均值与方差,npz格式，key是"mean"，"std"；
有OD级别和总流量级别
'''
def mean_std_save(args,log, M, data_mode,mode_name):
    M_OD = np.sum(M, axis=-1)  # (D,T1,N,N)
    mean, std = np.mean(M_OD), np.std(M_OD) #(M,T,N,N)
    path = os.path.join(args.dataset_dir, args.model, data_mode,mode_name, 'mean_std_OD.npz')
    np.savez_compressed(path, mean=mean, std=std)
    utils.log_string(log, 'OD=>mean:{},std:{}'.format(mean, std))
    M_IN = np.sum(M_OD,axis=-1) #(M,T,N)
    mean, std = np.mean(M_IN), np.std(M_IN)
    path = os.path.join(args.dataset_dir, args.model, data_mode, mode_name, 'mean_std_Flow.npz')
    np.savez_compressed(path, mean=mean, std=std)
    utils.log_string(log, 'OD=>mean:{},std:{}'.format(mean, std))
#################################### Data process for Models : Comomon#############################################
'''
功能：装载均值和标准差
'''
def load_mean_std(args):
    # 装载均值和标准差
    mode_name = utils.input_name(args)
    path = os.path.join(args.dataset_dir, 'all_models', args.data_mode,mode_name, 'mean_std_OD.npz')
    statistic = np.load(path)
    mean, std = statistic['mean'], statistic['std']
    return mean, std

'''
参数：标准化数据
'''
def standarize(args, samples, normalize=True):
    mean, std = load_mean_std(args)
    # 标准化
    if normalize:
        if args.model not in ["MVPF"]:
            samples = (samples - mean) / std
        else:
            samples[..., :-2] = (samples[..., :-2] - mean) / std # 最后三维是时间特征，避开
    return samples

'''
功能：打乱数据集(固定了seed)
输入一个list，list中每个元素是一个数据集，形状是多维数组，按照第一个维度将数据集打乱
'''
def shuffle(data):
    results = []
    sample_num = data[0].shape[0]
    #per = list(np.random.RandomState(seed=42).permutation(sample_num)) # 固定seed
    per = list(np.random.permutation(sample_num)) # 随机划分
    for i in range(len(data)):
        results.append(data[i][per,...])
    return results

'''
'''
def proportion(log, data, pro=0.1):
    num_sample = int(np.floor(data[0].shape[0] * pro))
    num_data = len(data)
    Data = []
    for i in range(num_data):
        Data.append(np.copy(data[i]))
    for i in range(num_data):
        Data[i] = Data[i][:num_sample,...]
        utils.log_string(log, '%s=>shape: %s' % (i, Data[i].shape))
    return Data

'''
划分训练集/验证集/测试集
'''
def data_split(args, log, data):
    batch_num = data[0].shape[0]
    train_num = round(args.train_ratio * batch_num)
    val_num = round(args.val_ratio * batch_num)
    x = len(data)
    trains = []
    vals = []
    tests = []
    for i in range(x):
        train = data[i][:train_num, ...]
        val = data[i][train_num:train_num+val_num, ...]
        test = data[i][train_num+val_num:, ...]
        trains.append(train)
        vals.append(val)
        tests.append(test)
        utils.log_string(log, '%s=>train: %s\tval: %s\ttest: %s' % (i, train.shape, val.shape, test.shape))
    return [trains, vals, tests]

'''
功能：划分batch
对data列表中每个元素的第一个维度进行batch划分，不够一个batch的丢弃掉
'''
def batch_split(args, data):
    x = len(data)
    sample_num = data[0].shape[0]
    results=[]
    for i in range(x):
        batch_num = sample_num // args.Batch_Size
        sample_num = sample_num - sample_num % args.Batch_Size
        t = data[i][:sample_num, ...]
        t = np.stack(np.split(t, batch_num, axis=0), axis=0)
        t = t.astype(np.float32)
        results.append(t)
    return results

'''
3- 保存训练集/验证集/测试集
'''
def save_dataslipt(args, data_dir, data):
    # for i in range(len(data)):
    #     data[i] = data[i].astype(np.float32)
    types = ['train', 'val', 'test']
    for i in range(len(types)):
        file_name = os.path.join(data_dir, types[i])
        if args.model in ["MVPF"]:
            np.savez_compressed(file_name, data[i][0], data[i][1], data[i][2],data[i][3], data[i][4])

        else:
            print("No such model! => ",args.model)
            raise ValueError

'''
根据模型去掉时间特征
'''
def filter_time(args, data, save_time=False):

    if args.model in [ "MVPF"]:
        data[1] = data[1][...,:-1] # 去掉输入的最后一维时间特征

    else:
        print("Wrong!")
        raise ValueError
    if save_time:
        time = data[0][..., -3:]
        data.append(time)
    data[0] = data[0][..., :-3]  # 去掉输出的时间特征
    return data
'''
3- 数据处理全流程
'''
def data_process(args):
    print(args)
    # 按输入模式，产生文件夹
    log_dir, save_dir, data_dir,  dir_name = utils.create_dir_PDW(args)
    log = utils.create_log(args, args.data_log)
    # 装载数据：[(M,N,N),(M,T1,N,N),(M,T2,N,N),...]
    data = load(args, log, dir_name)
    # 根据模型去掉时间特征
    data = filter_time(args, data)
    # 打乱数据集
    data = shuffle(data)

    if args.model in ["MVPF"]:
        # 产生距离图
        path = os.path.join(args.dataset_dir, 'original', 'graph_geo.npz')
        # print(path)
        A = np.load(path)['arr_0'].astype(np.float32)  # (N,N)
        graph_geo = graph_common.distance_graph_GEML(A)  # (N,N)
        path = os.path.join(data_dir, 'graph_geo.npz')
        # print(data_dir)
        np.savez_compressed(path, graph=graph_geo)






    # 标准化数据:只标准化输入X
    data[1] = standarize(args, data[1], normalize=args.Normalize)


    # 检查多维数组是否有nan,inf
    for d in data:
        utils.check_inf_nan(d)


    # 划分batch
    data = batch_split(args, data) # M=>(V,B)
    # 划分训练集/验证集/测试集
    data = data_split(args, log, data)
    # 保存数据
    save_dataslipt(args, data_dir, data)
    utils.log_string(log, 'data_type:%s, prediction type:%s'%(args.data_type, args.output_type))
    utils.log_string(log, 'Finish\n')
    return data

'''
3- 主模型装载数据
装载训练集/测试集/验证集，标准差/均值
'''
def load_data(args, log, data_dir, yes=True):
    utils.log_string(log, 'loading data...')
    types = ['train','val','test']
    results = []
    for type in types:
        path = os.path.join(data_dir, '%s.npz'%(type))
        data = np.load(path)
        dict = {}
        for j in range(len(data)):
            name = "arr_%s" % j
            dict[name] = data[name]
            utils.log_string(log, '%s=>%s shape: %s,type:%s' % (type, j, dict[name].shape,dict[name].dtype))
        results.append(dict)
    # 装载标准差和均值
    if yes:
        mean, std = load_mean_std(args)
    else:
        mean, std = 0.0, 1.0 # 数据不标准化
    utils.log_string(log,'mean:{},std:{}'.format(mean, std))
    return results, mean, std
