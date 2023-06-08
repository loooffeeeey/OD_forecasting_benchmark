import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from libs import utils, metrics, main_common, data_common, model_common
import numpy as np


# samples=(M,B,T,N,F_in), labels=(M,B,N,F_out)
def placeholder(T, N, F_in, F_out):
    labels = tf.compat.v1.placeholder(shape = (None, N, F_out), dtype = tf.float32,name="labels")
    samples = tf.compat.v1.placeholder(shape=(None, T, N, F_in), dtype=tf.float32, name="samples")
    flow_in = tf.compat.v1.placeholder(shape=(None, T, N), dtype=tf.float32, name="flow_in")
    od_out = tf.compat.v1.placeholder(shape=(None, T, N, N), dtype=tf.float32, name="od_out")
    flow_out = tf.compat.v1.placeholder(shape=(None, T, N), dtype=tf.float32, name="flow_out")

    return [labels, samples,flow_in,od_out,flow_out]




conv1d = tf.layers.conv1d

def attn_head(seq, bias_mat, out_sz, activation, residual=False):
    # with tf.name_scope('my_attn'):
    seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

    # simplest self-attention possible
    f_1 = tf.layers.conv1d(seq_fts, 1, 1)
    f_2 = tf.layers.conv1d(seq_fts, 1, 1)
    logits = f_1 + tf.transpose(f_2, [0, 2, 1])
    # coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)
    coefs = tf.nn.softmax(tf.nn.leaky_relu(logits))


    vals = tf.matmul(coefs, seq_fts)
    tmp_bias =  tf.Variable(tf.zeros_initializer()(shape = [vals.get_shape().as_list()[-1]]),
                dtype = tf.float32, trainable = True, name = 'bias') #(F1,)

    # ret = tf.contrib.layers.bias_add(vals)
    ret = tf.nn.bias_add(vals,tmp_bias)

    # residual connection
    if residual:
        if seq.shape[-1] != ret.shape[-1]:
            ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
        else:
            ret = ret + seq

    return activation(ret)  # activation



def inference(inputs, bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
    attns = []
    for _ in range(n_heads[0]):
        attns.append(attn_head(inputs, bias_mat=bias_mat,
                                      out_sz=hid_units[0], activation=activation,
                                      residual=False))
    h_1 = tf.concat(attns, axis=-1)
    for i in range(1, len(hid_units)):
        attns = []
        for _ in range(n_heads[i]):
            attns.append(attn_head(h_1, bias_mat=bias_mat,
                                          out_sz=hid_units[i], activation=activation,
                                          residual=residual))
        h_1 = tf.concat(attns, axis=-1)
    return h_1
    # out = []

    # for i in range(n_heads[-1]):
    #     out.append(layers.attn_head(h_1, bias_mat=bias_mat,
    #                                 out_sz=nb_classes, activation=lambda x: x,
    #                                 residual=False))
    # logits = tf.add_n(out) / n_heads[-1]
    #
    # return logits

def GAT_network_OD(i, input, args, GCN_activations, GCN_units,Ks, graph_geo, SE_embedding, TE, RNN_units):
    N = args.N
    name_scope_str = 'dynamic_feature_learning_' + str(i)
    with tf.name_scope(name_scope_str):
        # input = inputs[i]
        # # tmp = tf.concat([samples,TE,tf_ave_time],axis=-1)
        # tmp = tf.concat([input, TE], axis=-1)
        # # tmp = tf.concat([samples, tf_ave_time], axis=-1)
        # # tmp = samples
        #
        # X = model_common.multi_fc(tmp, activations=['relu'], units=[N])
        #
        # # X = model_common.x_embedding(X, N, activations=['relu', None])
        #
        # # (B,P,N,2D)=> (B,P,N,D)
        # query = model_common.multi_fc(X, activations=['relu'], units=[N])
        # key = model_common.multi_fc(X, activations=['relu'], units=[N])
        # value = model_common.multi_fc(X, activations=['relu'], units=[N])
        #
        #
        # # (BH,P,N,d)*(BH,P,N,d)=>(BH,P,N,N)
        # attention = tf.matmul(query, key, transpose_b=True)
        #
        # attention /= (N ** 0.5)
        # attention = tf.nn.softmax(attention, axis=-1)
        #
        # # (BH,P,N,N) * (BH,P,N,d)=> (BH,P,N,d)
        # attention = tf.matmul(attention, value)
        # # tf_graph_flow_in_od = tf.matmul(tf_graph_flow_in_od, graph_geo)
        # attention_resnet = attention+input
        # print(attention_resnet)

        # input = inputs[i]  # (B,P,N,N)

        # f_all = []
        # for index_i in range(N):
        #     f_i = input[...,index_i:index_i+1,:] #(B,P,1,N)
        #     f_i = tf.tile(f_i,[1,1,N,1])#(B,P,N,N)
        #     # f_i = tf.concat([f_i,input],axis=-1)#(B,P,N,N*2)
        #     f_i = model_common.multi_fc(tf.concat([f_i, input], axis=-1), activations=['relu'], units=[1])  # (B,P,N,1)
        #     f_all.append(f_i)
        # f = tf.stack(f_all,axis=3)#(B,P,N,N,1)

        # 这个参数调大了结果变差
        hid_units = 1  # numbers of hidden units per each attention head in each layer
        # n_heads = [8, 1]  # additional entry for the output layer
        # 1-3都没有什么变化,
        n_heads = 1

        if args.test_mode == 18:
            F_ = int(N / 4)
        else:
            F_ = int(N / n_heads)  # 向下取证

        #
        head_output = []
        with tf.name_scope("GAT"):
            for head in range(n_heads):
                layer_out = input
                last_input = input
                for k in range(hid_units):
                    with tf.name_scope("kernel"):
                        kernel = tf.Variable(tf.glorot_uniform_initializer()(shape=[N, F_]),
                                             dtype=tf.float32, trainable=True,
                                             name='kernel_{}_{}_{}'.format(i, head, k))  # (F, F1)
                    # print("layer_out", layer_out)
                    features = tf.matmul(layer_out, kernel)
                    with tf.name_scope("self_attention_kernel"):
                        self_attention_kernel = tf.Variable(tf.glorot_uniform_initializer()(shape=[F_, 1]),
                                                            dtype=tf.float32, trainable=True,
                                                            name='self_attention_{}_{}_{}'.format(i, head,
                                                                                                  k))  # (F, F1)
                    with tf.name_scope("neigh_attention_kernel"):
                        neigh_attention_kernel = tf.Variable(tf.glorot_uniform_initializer()(shape=[F_, 1]),
                                                             dtype=tf.float32, trainable=True,
                                                             name='neigh_attention_{}_{}_{}'.format(i, head,
                                                                                                    k))  # (F, F1)
                    attn_for_self = tf.matmul(features, self_attention_kernel)  # (N x 1), [a_1]^T [Wh_i] # (B,P,N,1)
                    attn_for_neighs = tf.matmul(features, neigh_attention_kernel)  # (N x 1), [a_2]^T [Wh_j] # (B,P,N,1)
                    dense = attn_for_self + tf.transpose(attn_for_neighs,
                                                         [0, 1, 3, 2])  # (N x N) via broadcasting # (B,P,N,N)
                    # dense = dense+layer_out
                    dense = tf.nn.softmax(tf.nn.softmax(dense, axis=-1), axis=-2)  # (B,P,N,N)
                    # geo_out = model_common.multi_gcn(graph_geo, input, activations=GCN_activations, units=GCN_units,
                    #                                  Ks=Ks)  # (B,T,N,128)

                    # inputs = dense
                    # num_layer = len(GCN_units)
                    # for n_i in range(num_layer):
                    #     inputs = tf.matmul(inputs,input)
                    #     inputs = model_common.fc_layer(inputs, units=GCN_units[n_i])
                    #     inputs = model_common.activation_layer(inputs, activation=GCN_activations[n_i])
                    # dense = inputs
                    with tf.name_scope("multi_gcn"):
                        dense = model_common.multi_gcn(dense, last_input, activations=GCN_activations, units=GCN_units,
                                                       Ks=Ks)  # (B,T,N,128)

                    # dense = dense + tf.matmul(f_ave)
                    # dense = tf.concat([dense, f_ave], axis=-1)
                    # w_ = tf.Variable(tf.glorot_uniform_initializer()(shape=[3, N, N]),
                    #                         dtype=tf.float32, trainable=True, name='w_{}_{}_{}'.format(i, head,k))  # (P, N, N)
                    # dense = dense + graph_geo * w_
                    # dense = dense + tf.matmul(f_ave, w_) # (B,P,N,N)
                    # dense = dense + f_ave * w_  # (B,P,N,N)
                    # print(dense)
                    # dense = tf.nn.leaky_relu(dense,alpha=0.2)

                    with tf.name_scope("multi_fc"):
                        dense = model_common.multi_fc(dense, activations=['relu'], units=[N])  # (B,P,N,N)
                    # dense = tf.nn.softmax(tf.nn.softmax(dense,axis=-1),axis=-2)
                    # dense = tf.nn.softmax(dense, axis=-1)
                    # dense = tf.concat([dense,geo_out], axis=-1)
                    layer_out = dense
                    # layer_out = model_common.multi_fc(layer_out, activations=['relu'], units=[int(N)])  # (B,P,N,N)
                    # layer_out = layer_out+last_input
                    last_input = layer_out
                head_output.append(layer_out)

        gat_out = tf.concat(head_output, axis=-1)  # (B,P,N,int(N/head)*head)

        with tf.name_scope('GCN'):
            gcn_out = model_common.multi_gcn(graph_geo, input, activations=GCN_activations, units=GCN_units,
                                             Ks=Ks)  # (B,T,N,128)
            with tf.name_scope('multi_fc'):
                gcn_out = model_common.multi_fc(gcn_out, activations=['relu', None], units=[64, N])  # (B,P,N,N)

        # 节点特征和图注意力网络特征拼接
        if args.test_mode == 8:
            V = tf.concat([SE_embedding, gat_out], axis=-1)  # (B,P,N,D) +(B,P,N,?)
        elif args.test_mode == 9:
            V = tf.concat([SE_embedding, gat_out, TE], axis=-1)  # (B,P,N,D) +(B,P,N,?)
        elif args.test_mode == 10:
            V = tf.concat([SE_embedding, gcn_out, TE], axis=-1)  # (B,P,N,D) +(B,P,N,?)
        elif args.test_mode == 17:
            V = tf.concat([SE_embedding, gcn_out, gat_out], axis=-1)  # (B,P,N,D) +(B,P,N,?)
        else:
            # 增加了GCN后结果变差了?
            V = tf.concat([SE_embedding, gcn_out, gat_out, TE], axis=-1)  # (B,P,N,D) +(B,P,N,?)



        # V_o = tf.concat([tf_graph_flow_in_od,tf_graph_flow_in],axis=-1)
        V = tf.transpose(V, perm=[0, 2, 1, 3])  # (B,N,T,256)
        V = tf.reshape(V, shape=(-1, V.shape[-2], V.shape[-1]))  # (BN,T,256)

        if args.test_mode == 11:
            H = tf.reshape(V, shape=(-1, V.shape[-2] * V.shape[-1]))  # (BN P,256) -> (BN,P*256)
        else:
            with tf.variable_scope(name_scope_str + '/RNN{}'.format(i)):
                # with tf.name_scope('RNN{}'.format(i)):
                H = model_common.multi_lstm(V, RNN_units, args.RNN_type)  # (BN P,256) -> (BN,128)

        H = tf.reshape(H, shape=[-1, N, H.shape[-1]])  # (BN,128) =>(B,N,128)
        # print(TE.shape)
        # exit()
        # H = tf.concat([H, TE], axis=-1)
        # H = model_common.fc_layer(tf.concat([H, TE], axis=-1), 2*N)
        # H = model_common.fc_layer(H, 2 * N)
        with tf.name_scope('RNN{}'.format(i)):
            with tf.name_scope('fc_layer'):
                H = model_common.fc_layer(H, N)
        # if args.test_mode == 8:

        # # 加入时间后变差
        # if 0:
        #     # H = tf.concat([H, TE], axis=-1)
        #     H = tf.concat([H, tf.nn.softmax(TE,axis=-1)], axis=-1)
        #     H = model_common.fc_layer(H, 2 * N)
        # H_res.append(H)
        return H

def Model(args, mean, std, samples, od_out,flow_i,flow_o,ave_time,graph_geo, SE, features):
    # print("graph_geo ",graph_geo)

    N = ave_time.shape[0]
    # print("N = ",N)
    T = 32 #时间段
    # D = 4 #D是对齐维度
    D = int(args.D)

    P = od_out.get_shape().as_list()[1]

    TE = tf.cast(samples[..., -2:], tf.int32)  # (B,P,N,2)

    with tf.name_scope("prepare_data"):
        day_of_week_and_time_of_day = 2
        with tf.name_scope("t_embbing_static"):
            TE = model_common.t_embbing_static(day_of_week_and_time_of_day, TE, T, D, activations=['relu', None])  # (B,P,N,D)

        # TE = tf.transpose(TE, [0, 2, 1, 3])  # (B,P,N,D) -> (B,N,P,D)
        # shape = TE.get_shape().as_list()
        # TE = tf.reshape(TE, shape=[-1, shape[1], shape[2] * shape[3]])  # (B,P,N,D)  -> (B,N,P*D)


        # TE = tf.tile(tf.expand_dims(TE,axis=3), [1, 1, 1,N, 1])  # (B,P,N,N,D)
        # TE = model_common.multi_fc(TE, activations=['relu',None],units=[64,1])
        # print(TE)

        # remove time data
        samples= samples[...,:-2]
        # P = TE.shape[1].value

        with tf.name_scope("SE_embedding"):
            SE_embedding = model_common.multi_fc(SE, activations=['relu', None], units=[64, N])
            SE_embedding = tf.add(SE_embedding, tf.zeros_like(samples))  # (B,P,N,D)

        GCN_activations = args.GCN_activations.split("-")
        GCN_units = [int(i) for i in args.GCN_units.split("-")]
        Ks = [None] * len(GCN_units)
        RNN_units = [int(i) for i in args.RNN_units.split("-")]


        # 边的特征为节点i和节点j拼接后的结果
        # edge_feature = np.zeros(shape=[N, N, (features.shape[-1]) * 2], dtype=np.float32)  # (N,N,F)
        # for i in range(N):
        #     for j in range(N):
        #         edge_feature[i, j] = np.concatenate([features[i:i + 1], features[j:j + 1]],axis=-1)

        # # node = model_common.multi_fc(SE, activations=['relu', None], units=[64, 1])# 将D维降到1维
        # node = model_common.multi_fc(edge_feature, activations=['relu', None], units=[64, 1])  # 将D维降到1维
        # node = tf.squeeze(node, axis=-1)  # (N,N)
        # node = tf.add(node, tf.zeros_like(samples))  # (B,P,N,N)
        #
        # # node_concat_edge = tf.concat([samples,node],axis=-1)
        # node_concat_edge = node
        # node_concat_edge = tf.nn.softmax(node_concat_edge)

        # tmp = tf.matmul(flow_i, graph_geo)
        # print(GCN_units)

        # # GCN_units = [N] * len(GCN_units)
        # # print(GCN_units)
        # # exit()
        # tmp = model_common.multi_gcn(graph_geo, flow_i, activations=GCN_activations, units=GCN_units,
        #                                  Ks=Ks)  # (B,T,N,128)
        # print(tmp)
        # exit()
        with tf.name_scope("inputs"):
            if args.test_mode == 16:
                inputs = [model_common.multi_fc(samples, activations=['relu'], units=[N]),
                          model_common.multi_fc(tf.expand_dims(flow_i, axis=-1), activations=['relu'], units=[N]),
                          model_common.multi_fc(od_out, activations=['relu'], units=[N]),
                          model_common.multi_fc(tf.expand_dims(flow_o, axis=-1), activations=['relu'], units=[N])]
            else:
                inputs = [samples,
                          model_common.multi_fc(tf.expand_dims(flow_i, axis=-1), activations=['relu'], units=[N]),
                          od_out,
                          model_common.multi_fc(tf.expand_dims(flow_o, axis=-1), activations=['relu'], units=[N])]


    H_res = [1] * len(inputs)
    # # f_ave = tf.expand_dims(tf.add(ave_time, tf.zeros_like(inputs[0])), axis=-1)  # (B,P,N,N)
    # f_ave = tf.add(ave_time, tf.zeros_like(inputs[0]))  # (B,P,N,N)
    # for i in range(4):

    for i in range(len(inputs)):
        H_res[i] = GAT_network_OD(i, inputs[i], args, GCN_activations, GCN_units, Ks, graph_geo, SE_embedding, TE, RNN_units)


    # 模型降解
    # 1.去掉入站客流和出站客流
    if args.test_mode == 1:
        H_i = H_res[1]
        H_o = H_res[3]
    # 2.去掉入站客流OD和出站客流OD
    elif args.test_mode == 2:
        H_i = H_res[0]
        H_o = H_res[2]
    # 3.入站作为OD特征
    elif args.test_mode == 3:
        H_i = tf.concat([H_res[0], H_res[1]], axis=-1)
        H_o = H_i
    # 4.出站作为OD特征
    elif args.test_mode == 4:
        H_o = tf.concat([H_res[2], H_res[3]], axis=-1)
        H_i = H_o
    elif args.test_mode == 12:
        H_i = H_res[1]
        H_o = tf.concat([H_res[2], H_res[3]], axis=-1)
    elif args.test_mode == 13:
        H_i = H_res[0]
        H_o = tf.concat([H_res[2], H_res[3]], axis=-1)
    elif args.test_mode == 14:
        H_i = tf.concat([H_res[0], H_res[1]], axis=-1)
        H_o = H_res[3]
    elif args.test_mode == 15:
        H_i = tf.concat([H_res[0], H_res[1]], axis=-1)
        H_o = H_res[3]
    else:
        H_i = tf.concat([H_res[0], H_res[1]], axis=-1)
        H_o = tf.concat([H_res[2], H_res[3]], axis=-1)


    # V_i = tf.concat([SE_embedding]+com_res[0]+com_res[1], axis=-1)
    #
    # # V_o = tf.concat([tf_graph_flow_in_od,tf_graph_flow_in],axis=-1)
    # V_i = tf.transpose(V_i, perm=[0, 2, 1, 3])  # (B,N,T,256)
    # V_i = tf.reshape(V_i, shape=(-1, V_i.shape[-2], V_i.shape[-1]))  # (BN,T,256)
    # with tf.variable_scope('RNN0'):
    #     H_i = model_common.multi_lstm(V_i, RNN_units, args.RNN_type)  # (BN P,256) -> (BN,128)
    # H_i = tf.reshape(H_i, shape=[-1, N, H_i.shape[-1]])  # (BN,128) =>(B,N,128)
    #
    # V_o = tf.concat([SE_embedding] + com_res[2] + com_res[3], axis=-1)
    #
    # # V_o = tf.concat([tf_graph_flow_in_od,tf_graph_flow_in],axis=-1)
    # V_o = tf.transpose(V_o, perm=[0, 2, 1, 3])  # (B,N,T,256)
    # V_o = tf.reshape(V_o, shape=(-1, V_o.shape[-2], V_o.shape[-1]))  # (BN,T,256)
    # with tf.variable_scope('RNN1'):
    #     H_o = model_common.multi_lstm(V_o, RNN_units, args.RNN_type)  # (BN P,256) -> (BN,128)
    # H_o = tf.reshape(H_o, shape=[-1, N, H_o.shape[-1]])  # (BN,128) =>(B,N,128)

    # H_m = model_common.fc_layer(H, units=H.get_shape().as_list()[-1])  # (B,N,128)=>(B,N,128)
    # 这里需要使用其他技术进行融合

    with tf.name_scope('conversion_relationship_learning'):
        # 这里实际上是将O的特征和D的特征结合起来，得到从O到D的客流
        if args.test_mode == 5:
            w_i = tf.Variable(tf.glorot_uniform_initializer()(shape=[H_i.shape[-1], H_i.shape[-1]]),
                            dtype=tf.float32, trainable=True, name='kernel')  # (F, F1)
            # w_i 与时间，平均到达时间相关
            H_tmp = tf.matmul(H_i,w_i)
            # w_o = tf.Variable(tf.glorot_uniform_initializer()(shape=[H_o.shape[-1], H_o.shape[-1]]),
            #                   dtype=tf.float32, trainable=True, name='kernel')  # (F, F1)
            # H_o = tf.matmul(H_o,w_o)
            M_pred = tf.matmul(H_tmp, H_o, transpose_b=True)  # (B,N,128)*(B,N,128)=>(B,N,N)
        elif args.test_mode == 6:# 残差
            w_i = tf.Variable(tf.glorot_uniform_initializer()(shape=[H_i.shape[-1], H_i.shape[-1]]),
                              dtype=tf.float32, trainable=True, name='kernel')  # (F, F1)
            H_tmp = tf.matmul(H_i, w_i)
            # w_o = tf.Variable(tf.glorot_uniform_initializer()(shape=[H_o.shape[-1], H_o.shape[-1]]),
            #                   dtype=tf.float32, trainable=True, name='kernel')  # (F, F1)
            # H_o = tf.matmul(H_o,w_o)
            M_pred1 = tf.matmul(H_tmp, H_o, transpose_b=True)  # (B,N,128)*(B,N,128)=>(B,N,N)

            M_pred1 = model_common.fc_layer(M_pred1, units=N)  # (B,N,128)=>(B,N,1)
            M_pred2 = tf.matmul(H_i, H_o, transpose_b=True)  # (B,N,128)*(B,N,128)=>(B,N,N)
            M_pred = M_pred1+M_pred2

        elif args.test_mode == 7: # 没有转移矩阵
            M_pred = tf.matmul(H_i, H_o, transpose_b=True)  # (B,N,128)*(B,N,128)=>(B,N,N)

        elif args.test_mode == 19:

            H_i = model_common.fc_layer(H_i, units=N)  # (B,N,128)=>(B,N,N)
            H_o = model_common.fc_layer(H_o, units=N)  # (B,N,128)=>(B,N,N)

            w_i = tf.Variable(tf.glorot_uniform_initializer()(shape=[H_i.shape[-1], H_i.shape[-1]]),
                              dtype=tf.float32, trainable=True, name='kernel')  # (F, F1)
            H_tmp = tf.matmul(H_i, w_i)
            M_pred = tf.matmul(H_tmp, H_o, transpose_b=True)  # (B,N,128)*(B,N,128)=>(B,N,N)
            w_landa_1 = tf.Variable(tf.glorot_uniform_initializer()(shape=[N, N]),
                                    dtype=tf.float32, trainable=True, name='kernel')  # (F, F1)

            # w_landa_2 = tf.Variable(tf.glorot_uniform_initializer()(shape=[N, N]),
            #                         dtype=tf.float32, trainable=True, name='kernel')  # (F, F1)
            # weight = w_landa_2 * tf.exp(w_landa_1 * (1 / (ave_time + np.eye(N) * 3)))
            weight = w_landa_1

            M_pred = model_common.fc_layer(M_pred * weight, N)
        elif args.test_mode == 20:

            H_i = model_common.fc_layer(H_i, units=N)  # (B,N,128)=>(B,N,N)
            H_o = model_common.fc_layer(H_o, units=N)  # (B,N,128)=>(B,N,N)

            w_i = tf.Variable(tf.glorot_uniform_initializer()(shape=[H_i.shape[-1], H_i.shape[-1]]),
                              dtype=tf.float32, trainable=True, name='kernel')  # (F, F1)
            H_tmp = tf.matmul(H_i, w_i)
            M_pred = tf.matmul(H_tmp, H_o, transpose_b=True)  # (B,N,128)*(B,N,128)=>(B,N,N)
            w_landa_1 = tf.Variable(tf.glorot_uniform_initializer()(shape=[N, N]),
                                    dtype=tf.float32, trainable=True, name='kernel')  # (F, F1)

            # w_landa_2 = tf.Variable(tf.glorot_uniform_initializer()(shape=[N, N]),
            #                         dtype=tf.float32, trainable=True, name='kernel')  # (F, F1)
            # weight = w_landa_2 * tf.exp(w_landa_1 * (1 / (ave_time + np.eye(N) * 3)))
            # weight = w_landa_1
            weight = tf.exp(w_landa_1 * (1 / (ave_time + np.eye(N) * 3)))
            M_pred = model_common.fc_layer(M_pred * weight, N)

        elif args.test_mode == 21:
            # 考虑平均时间(N,N)和当前时间(B,P,N,2)得到一个转移矩阵(d,d)
            # 其实OD就是一个O到D的转移，可以参考graph enbeding
            H_i = model_common.fc_layer(H_i, units=N)  # (B,N,128)=>(B,N,N)
            H_o = model_common.fc_layer(H_o, units=N)  # (B,N,128)=>(B,N,N)

            w_i = tf.Variable(tf.glorot_uniform_initializer()(shape=[H_i.shape[-1], H_i.shape[-1]]),
                              dtype=tf.float32, trainable=True, name='kernel')  # (F, F1)

            # TE = tf.transpose(TE, [0, 2, 1, 3])  # (B,P,N,D) -> (B,N,P,D)
            # shape = TE.get_shape().as_list()
            # TE = tf.reshape(TE, shape=[-1, shape[1], shape[2] * shape[3]])  # (B,P,N,D)  -> (B,N,P*D) -> (B,N,N)
            # TE = model_common.fc_layer(TE, units=N)  # (B,N,P*D) -> (B,N,N)

            # w_i = tf.matmul(w_i,np.log(ave_time))
            # w_i = model_common.fc_layer(w_i, units=N)
            # w_i = tf.matmul(w_i, ave_time)
            # w_i 与时间，平均到达时间相关

            H_tmp = tf.matmul(H_i, w_i)

            # print(shape)
            # exit()
            # H_tmp = tf.matmul(H_tmp, ave_time)
            # f_ave = tf.add(ave_time, tf.zeros_like(H_tmp))  # (B,P,N,N)
            # H_tmp = tf.matmul(H_tmp, f_ave)
            # H_tmp = model_common.fc_layer(H_tmp, units=N)
            # w_o = tf.Variable(tf.glorot_uniform_initializer()(shape=[H_o.shape[-1], H_o.shape[-1]]),
            #                   dtype=tf.float32, trainable=True, name='kernel')  # (F, F1)
            # H_o = tf.matmul(H_o,w_o)

            M_pred = tf.matmul(H_tmp, H_o, transpose_b=True)  # (B,N,128)*(B,N,128)=>(B,N,N)

            w_landa_1 = tf.Variable(tf.glorot_uniform_initializer()(shape=[N, N]),
                                    dtype=tf.float32, trainable=True, name='kernel')  # (F, F1)

            w_landa_2 = tf.Variable(tf.glorot_uniform_initializer()(shape=[N, N]),
                                    dtype=tf.float32, trainable=True, name='kernel')  # (F, F1)
            # 加入exp后梯度爆炸或者梯度消失可能会更严重
            # weight = w_landa_2*tf.exp(w_landa_1 * (-ave_time))
            # weight = w_landa_2 * tf.exp(w_landa_1*ave_time)
            # 这里使用点乘
            weight = tf.exp(w_landa_1 * (1 / (ave_time + np.eye(N) * 3)))
            weight = tf.matmul(w_landa_2,weight)
            # weight = tf.matmul(w_landa_2,tf.exp(w_landa_1*(1/(ave_time+1))))
            # weight = tf.matmul(w_landa_2, tf.exp(w_landa_1 * (1 / (ave_time + np.eye(N)*3))))
            # weight = tf.matmul(w_landa_2, tf.exp(w_landa_1))
            # weight = w_landa_2
            # M_pred = tf.matmul(M_pred,weight)
            # M_pred = M_pred * weight, N + M_pred
            # 这里使用点乘
            M_pred = model_common.fc_layer(M_pred * weight, N)
            # print("zero Num: ", np.sum(np.not_equal(ave_time, 0)))
            # exit()

        else:#老师提出的方法
            # 考虑平均时间(N,N)和当前时间(B,P,N,2)得到一个转移矩阵(d,d)
            # 其实OD就是一个O到D的转移，可以参考graph enbeding
            H_i = model_common.fc_layer(H_i, units=N)  # (B,N,128)=>(B,N,N)
            H_o = model_common.fc_layer(H_o, units=N)  # (B,N,128)=>(B,N,N)

            w_i = tf.Variable(tf.glorot_uniform_initializer()(shape=[H_i.shape[-1], H_i.shape[-1]]),
                              dtype=tf.float32, trainable=True, name='kernel')  # (F, F1)

            # TE = tf.transpose(TE, [0, 2, 1, 3])  # (B,P,N,D) -> (B,N,P,D)
            # shape = TE.get_shape().as_list()
            # TE = tf.reshape(TE, shape=[-1, shape[1], shape[2] * shape[3]])  # (B,P,N,D)  -> (B,N,P*D) -> (B,N,N)
            # TE = model_common.fc_layer(TE, units=N)  # (B,N,P*D) -> (B,N,N)

            # w_i = tf.matmul(w_i,np.log(ave_time))
            # w_i = model_common.fc_layer(w_i, units=N)
            # w_i = tf.matmul(w_i, ave_time)
            # w_i 与时间，平均到达时间相关

            H_tmp = tf.matmul(H_i, w_i)

            # print(shape)
            # exit()
            # H_tmp = tf.matmul(H_tmp, ave_time)
            # f_ave = tf.add(ave_time, tf.zeros_like(H_tmp))  # (B,P,N,N)
            # H_tmp = tf.matmul(H_tmp, f_ave)
            # H_tmp = model_common.fc_layer(H_tmp, units=N)
            # w_o = tf.Variable(tf.glorot_uniform_initializer()(shape=[H_o.shape[-1], H_o.shape[-1]]),
            #                   dtype=tf.float32, trainable=True, name='kernel')  # (F, F1)
            # H_o = tf.matmul(H_o,w_o)

            M_pred = tf.matmul(H_tmp, H_o, transpose_b=True)  # (B,N,128)*(B,N,128)=>(B,N,N)

            w_landa_1 = tf.Variable(tf.glorot_uniform_initializer()(shape=[N, N]),
                                    dtype=tf.float32, trainable=True, name='kernel')  # (F, F1)

            w_landa_2 = tf.Variable(tf.glorot_uniform_initializer()(shape=[N, N]),
                                    dtype=tf.float32, trainable=True, name='kernel')  # (F, F1)
            # 加入exp后梯度爆炸或者梯度消失可能会更严重
            # weight = w_landa_2*tf.exp(w_landa_1 * (-ave_time))
            # weight = w_landa_2 * tf.exp(w_landa_1*ave_time)
            # 这里使用点乘
            # weight = w_landa_2 * tf.exp(w_landa_1 * (1 / (ave_time + np.eye(N) * 3)))
            weight = tf.exp(w_landa_1 * (1 / (ave_time + np.eye(N) * 3)))

            # weight = tf.matmul(w_landa_2,tf.exp(w_landa_1*(1/(ave_time+1))))
            # weight = tf.matmul(w_landa_2, tf.exp(w_landa_1 * (1 / (ave_time + np.eye(N)*3))))
            # weight = tf.matmul(w_landa_2, tf.exp(w_landa_1))
            # weight = w_landa_2
            # M_pred = tf.matmul(M_pred,weight)
            # M_pred = M_pred * weight, N + M_pred
            # 这里使用点乘
            # M_pred = model_common.fc_layer(M_pred * weight, N)
            M_pred = M_pred * weight
            # M_pred = tf.nn.dropout(M_pred, 0.9)
            # print("zero Num: ", np.sum(np.not_equal(ave_time, 0)))
            # exit()

    with tf.name_scope('others'):
        P_pred = model_common.fc_layer(H_i, units=1) #(B,N,128)=>(B,N,1)
        Q_pred = model_common.fc_layer(H_i,units=1) #(B,N,128)=>(B,N,1)
        with tf.name_scope('M_pred'):
            M_pred = model_common.inverse_positive(M_pred, std, mean, M_pred.shape[-1])



        # print("M_pred ",M_pred)
        P_pred = model_common.inverse_positive(P_pred, std, mean, P_pred.shape[-1])
        Q_pred = model_common.inverse_positive(Q_pred, std, mean, Q_pred.shape[-1])

    print("model_analyzer")
    # import tensorflow.contrib.slim as slim
    import tf_slim as slim
    # slim.model_analyzer.analyze_vars(M_pred, print_info=True)


    # slim.model_analyzer.analyze_ops(tf.get_default_graph(), print_info=True)

    # variables = tf.model_variables()
    # variables = tf.global_variables()
    # slim.model_analyzer.analyze_vars(variables, print_info=True)
    # print("model_analyzer")

    # exit()
    return [M_pred,P_pred,Q_pred]



