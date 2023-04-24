# -*- coding: UTF-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

# (M,P,N)
def multi_targets(X, std, mean, F_out):
    OD = inverse_positive(X, std, mean,F_out)
    In_preds = tf.reduce_sum(OD, axis=-1) #(M,P,N)
    Out_preds = tf.reduce_sum(OD, axis=-2) #(M,P,N)
    return [OD, In_preds, Out_preds]
'''
逆标准化层
加入一个线性变换
'''
def inverse_positive(X, std, mean, units):
    inputs = X * std + mean
    inputs = fc_layer(inputs, units=units)
    inputs = activation_layer(inputs, activation='relu')
    return inputs


def choose_RNNs(unit,type='GRU'):
    if type=="GRU":
        cell = tf.nn.rnn_cell.GRUCell(num_units=unit)
    elif type=="LSTM":
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=unit, state_is_tuple=False)
    elif type =="RNN":
        cell = tf.nn.rnn_cell.BasicRNNCell(num_units=unit)
    else:
        print("Wrong type")
        cell = None
    return cell

'''
RNN 层
输入：(B,P,F)
输出: (B,F)
'''
def lstm_layer(X, unit,type='GRU'):
    cell  = choose_RNNs(unit=unit, type=type)
    outputs, last_states = tf.nn.dynamic_rnn(cell=cell, inputs=X, dtype=tf.float32)
    output = outputs[:, -1, :] #(B,F)
    return output

def multi_lstm(X, units, type='GRU'):
    cells = [choose_RNNs(unit=unit, type=type) for unit in units]
    stacked_cells = tf.nn.rnn_cell.MultiRNNCell(cells)
    outputs, last_states = tf.nn.dynamic_rnn(cell=stacked_cells, inputs=X, dtype=tf.float32)
    output = outputs[:, -1, :] #(B,F)
    return output

'''
多层FC
'''
def multi_fc(X, activations=['relu'], units=[64], drop_rate=None, bn=False, dims=None, is_training=True):
    num_layer = len(units)
    inputs = X
    for i in range(num_layer):
        if drop_rate is not None:
            inputs = dropout_layer(inputs, drop_rate=drop_rate, is_training=is_training)
        inputs = fc_layer(inputs, units=units[i])
        inputs = activation_layer(inputs, activation=activations[i], bn=bn, dims=dims, is_training=dims)
    return inputs

'''
多层GCN
'''
def multi_gcn(graph, X, activations=["relu"], units=[128], Ks = [None],drop_rate=None, bn=False, dims=None, is_training=True):
    num_layer = len(units)
    inputs = X
    for i in range(num_layer):
        if drop_rate is not None:
            inputs = dropout_layer(inputs, drop_rate=drop_rate, is_training=is_training)
        inputs = gcn_layer(graph, inputs, K=Ks[i])
        inputs = fc_layer(inputs,units=units[i])
        inputs = activation_layer(inputs, activation=activations[i], bn=bn, dims=dims, is_training=dims)
    return inputs

# def multi_gcn_test(L, X, activations=["relu"], units=[128], Ks = [None],drop_rate=None, bn=False, dims=None, is_training=True):
#     num_layer = len(units)
#     inputs = X
#     for i in range(num_layer):
#         inputs = tf.matmul(L, inputs)
#         inputs = fc_layer(inputs,units=units[i])
#         inputs = activation_layer(inputs, activation=activations[i], bn=bn, dims=dims, is_training=dims)
#     return inputs

'''
空间静态嵌入
针对(B,T,N,N)，给予每个点一个spatial embedding
输入：(N,Fs)
输出: (1,1,N,D),D是对齐维度
'''
def s_embbing_static(SE, D, activations=['relu',None], drop_rate=None, bn=False, dims=None, is_training=True):
    SE = tf.expand_dims(tf.expand_dims(SE, axis = 0), axis = 0) # (1,1,N,Fs)
    SE = multi_fc(SE, activations=activations, units=[D,D], drop_rate=drop_rate, bn=bn, dims=dims, is_training=is_training)
    return SE # (1,1,N,D)

'''
时间静态嵌入
针对(B,T,N,N)，给予每个点一个spatial embedding
输入：(N,Fs)
输出: (B,T,N,D),D是对齐维度
'''
def t_embbing_static(t, TE, T, D, activations=['relu',None], drop_rate=None, bn=False, dims=None, is_training=True):
    if t==0:
        TE = tf.one_hot(TE[..., 0], depth=7)  # (B,T,N,7),dayofweek
    elif t==1:
        TE = tf.one_hot(TE[..., 1], depth=T)  # (B,T,N,T)
    elif t==2:
        dayofweek = tf.one_hot(TE[..., 0], depth=7)
        timeofday = tf.one_hot(TE[..., 1], depth=T)  # (B,T,N,T)
        TE = tf.concat((dayofweek, timeofday), axis=-1)  # (B,T,N,T+7)
    else:
        print("No time feature")
    TE = multi_fc(TE, activations=activations, units=[D,D], drop_rate=drop_rate, bn=bn, dims=dims, is_training=is_training)
    return TE

'''
特征拼接
X:(B,P,N,N)=>(B,P,N,D)
SE:(1,1,N,D)
TE:(B,P,N,D)
输出：(B,P,N,2D)
'''
def x_embedding(X, D, activations=['relu',None],drop_rate=None, bn=False, dims=None, is_training=True):
    X = multi_fc(X, activations=activations, units=[D,D], drop_rate=drop_rate, bn=bn, dims=dims, is_training=is_training)
    return X

def x_SE_TE(X, SE, TE, is_X=True, is_SE=True, is_TE=True):
    # print("X ",X)
    # print("SE ",SE)
    # print("TE ",TE)

    type = [is_X, is_SE, is_TE]
    if type == [True, True, True]:
        STE = tf.add(SE, TE)  # (B,P,N,D)
        output = tf.concat((X, STE), axis=-1)  # (B,P,N,2D)
    elif type == [False, True, True]:
        STE = tf.add(SE, TE)  # (B,P,N,D)
        output = STE
    elif type == [True, True, False]:
        # print("no TE")
        tmp = tf.zeros_like(X)
        SE = tf.add(SE, tmp)  # (B,P,N,D)
        output = tf.concat((X, SE), axis=-1)  # (B,P,N,2D)

    elif type == [True, False, True]:
        # print("no SE")
        output = tf.concat((X, TE), axis=-1)  # (B,P,N,2D)

    elif type == [True, False, False]:
        output = X
    else:
        raise ValueError
        print("Wrong Type!")
    print("output",output)
    # exit()
    return output

def x_spatio_temporal(X, SE, TE, activations=['relu',None],drop_rate=None, bn=False, dims=None, is_training=True):
    D = SE.shape[-1]
    STE =tf.add(SE, TE) # (B,P,N,D)
    X = multi_fc(X, activations=activations, units=[D,D], drop_rate=drop_rate, bn=bn, dims=dims, is_training=is_training)
    X = tf.concat((X, STE), axis=-1) #(B,P,N,2D)
    return X

'''
conv2d 层(无激活)
输入:[..., input_dim], 输出[..., output_dim]
'''
def conv2d_layer(x, output_dims, kernel_size, stride=[1, 1],padding='SAME'):
    input_dims = x.get_shape()[-1].value
    kernel_shape = kernel_size + [input_dims, output_dims]
    # 卷积核用glorot_uniform初始化
    kernel = tf.Variable(
        tf.glorot_uniform_initializer()(shape=kernel_shape),
        dtype=tf.float32, trainable=True, name='kernel')
    # 卷积
    x = tf.nn.conv2d(x, kernel, [1] + stride + [1], padding=padding)
    bias = tf.Variable(tf.zeros_initializer()(shape=[output_dims]),dtype=tf.float32, trainable=True, name='bias')
    x = tf.nn.bias_add(x, bias)
    return x

'''
线性变换层
输入(...,F)，输出(...,units)
'''
def fc_layer(X, units=128):
    W = tf.Variable(tf.glorot_uniform_initializer()(shape = [X.shape[-1],units]),
                    dtype = tf.float32, trainable = True, name='kernel') #(F, F1)
    b = tf.Variable(tf.zeros_initializer()(shape = [units]),
                    dtype = tf.float32, trainable = True, name = 'bias') #(F1,)
    Y = tf.matmul(X, W) + b #(...,F)*(F,F1)+(F1,)=>(...,F1)
    return Y

''''
功能：可是输入静态矩阵，也可以输入动态矩阵，进行图卷积
K = None，表示无切比雪夫公式；K = 4 表示4阶切比雪夫公式
L=(N,N),X=(B,N,F),Output=(B,N,F1)
L=(B,T,N,N),X=(B,T,N,F),Output=(B,T,N,F1)
L=(N,N),X=(B,T,N,F),Output=(B,T,N,F1)
'''
def gcn_layer(L, X, K = None):
    y1 = X
    y2 = tf.matmul(L,y1)
    if K:
        x0, x1 = y1, y2
        total = [x0, x1]
        for k in range(3, K + 1):
            x2 = 2 * tf.matmul(L,x1) - x0
            total.append(x2)
            x1, x0 = x2, x1
        total = tf.concat(total, axis=-1)
        y2 = total
    return y2


'''
激活层
带有norm层
'''
def activation_layer(X, activation='relu', bn=False, dims=None, is_training=True):
    inputs = X
    if activation!=None:
        if bn:
            inputs = batch_norm(inputs, dims, is_training)
        if activation == 'relu':
            inputs = tf.nn.relu(inputs)
        elif activation == 'sigmoid':
            inputs = tf.nn.sigmoid(inputs)
        elif activation == 'tanh':
            inputs = tf.nn.tanh(inputs)
    return inputs

'''
Dropout 层
训练时，采用dropout，测试不采用
'''
def dropout_layer(x, drop_rate, is_training):
    x = tf.cond(
        is_training,
        lambda: tf.nn.dropout(x, rate=drop_rate),
        lambda: x)
    return x

'''
标准化层
'''
def batch_norm(x, dims, is_training):
    # 形状
    shape = x.get_shape().as_list()[-dims:]
    # 偏置系数，初始值为0
    beta = tf.Variable(
        tf.zeros_initializer()(shape=shape),
        dtype=tf.float32, trainable=True, name='beta')
    # 缩放系数，初始值为1
    gamma = tf.Variable(
        tf.ones_initializer()(shape=shape),
        dtype=tf.float32, trainable=True, name='gamma')
    # 计算均值和方差
    moment_dims = list(range(len(x.get_shape()) - dims))
    batch_mean, batch_var = tf.nn.moments(x, moment_dims, name='moments')
    # 滑动平均
    ema = tf.train.ExponentialMovingAverage(0.9)
    # Operator that maintains moving averages of variables.
    ema_apply_op = tf.cond(
        is_training,
        lambda: ema.apply([batch_mean, batch_var]),
        lambda: tf.no_op())
    # Update moving average and return current batch's avg and var.
    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    # ema.average returns the Variable holding the average of var.
    mean, var = tf.cond(
        is_training,
        mean_var_with_update,
        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    # 标准化:x输入，mean均值，var方差，beta=偏移值，gama=缩放系数，1e-3=防止除零
    x = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return x

'''
所有可训练参数，加入正则化
'''
def regularizer(Loss):
    tf.add_to_collection(name="losses", value=Loss)
    for variable in tf.trainable_variables():
        tf.add_to_collection(name='losses', value=tf.nn.l2_loss(variable))
    Loss_l2 = tf.add_n(tf.get_collection("losses"))
    return Loss_l2
