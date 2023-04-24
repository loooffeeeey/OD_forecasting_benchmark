import numpy as np
import pickle
import scipy.sparse as sp
from scipy.sparse import linalg
import os

'''
功能：去掉对角线
输入：矩阵(..., N,N)
'''
def del_diag(A):
    M = np.copy(A)
    N = A.shape[-1]
    for i in range(N):
        M[...,i,i]=0.0
    return M

'''
功能：加入自环
输入：矩阵(..., N,N)
'''
def self_loop(A):
    N = A.shape[-1]
    I = np.identity(N)
    A_loop = A + I
    A_loop = A_loop.astype(np.float32)
    return A_loop



'''
功能：实现通过静态特征或动态特征计算的注意力
输入：特征矩阵(...,N,F)
'''
def grap_attention(A):
    A_T =A.swapaxes(-1, -2)  # (...,F,N)
    attention = np.matmul(A, A_T) #(...,N,F)(...,F,N)=>(...,N,N)
    return attention


'''
功能：按行比例归一化(0)，或按比例列归一化(1)
输入：多维矩阵(...,N,N)
'''
def normalize_attention(A, axis=0):
    if axis == 0:
        T = np.expand_dims(np.sum(A, axis=-1), axis=-1) #(...,N)
        attention = A/T
    elif axis == 1:
        T = np.expand_dims(np.sum(A, axis=-2), axis=-2)
        attention = A / T
    else:
        print("axis should be 0 or 1")
    return attention


'''
功能：按行softmax归一化(0)，或按softmax列归一化(1)
输入：多维矩阵(...,N,N)
备注：会极大地放大差距
'''
def softmax_attention(A, axis=0):
    if axis == 0:
        A_max = np.expand_dims(np.max(A, axis=-1), axis=-1) #(...,N)
        T = np.exp(A - A_max)
        L = np.expand_dims(np.sum(T, axis=-1), axis=-1) #(...,N)
        attention = T / L
    elif axis == 1:
        A_max = np.expand_dims(np.max(A, axis=-2), axis=-2)
        T = np.exp(A - A_max)
        L = np.expand_dims(np.sum(T, axis=-2), axis=-2) #(...,N)
        attention = T / L
    else:
        print("axis should be 0 or 1")
    return attention




# adj_mx:ndarray, L:ndarray
def transform(adj_mx, filter_type="dual_random_walk"):
    if filter_type == "laplacian":
        L = calculate_scaled_laplacian(adj_mx, lambda_max=None)
    elif filter_type == "random_walk":
        L = calculate_random_walk_matrix(adj_mx).T
    elif filter_type == "dual_random_walk":
        L = calculate_random_walk_matrix(adj_mx.T).T
    elif filter_type == "scaled_laplacian":
        L = calculate_scaled_laplacian(adj_mx)
    else:
        L = adj_mx
    return L

# matrices:ndarray (B,P,N,N)
def all_transform(matrices, filter_type='random_walk'):
    B = matrices.shape[0]
    P = matrices.shape[1]
    Matrices = np.zeros_like(matrices)
    for i in range(B):
        for j in range(P):
            adj_mx = matrices[i, j, ...]  # (N,N), ndarray
            Matrices[i:i + 1, j:j + 1, ...] = transform(adj_mx, filter_type=filter_type)
    return Matrices

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    normalized_laplacian = normalized_laplacian.todense().astype(np.float32)
    return normalized_laplacian

def calculate_random_walk_matrix(adj_mx):
    Adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(Adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(Adj_mx).tocoo()
    random_walk_mx = random_walk_mx.todense().astype(np.float32)
    return random_walk_mx

def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        Adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(Adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    L = L.todense().astype(np.float32)
    return L

def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


##############################  Each Model #########################
'''
功能：构建semantic graph (GEML)
      度向量+邻居矩阵，按行归一化，加入自环
输入：矩阵(...,N,N)
'''
def one_graph(A):
    N,N = A.shape
    Ms = np.zeros_like(A).astype(np.float32)
    D_In = np.sum(A, axis=-1)  # (N,)
    D_Out = np.sum(A, axis=-2)  # (N,)
    for i in range(N):
        for j in range(i,N):
            if i == j: continue
            if A[i, j] > 0.0 or A[j, i] > 0.0:
                Ms[j, i] = Ms[i, j] = D_In[j] + D_Out[j]
    for i in range(N):
        row_sum = np.sum(Ms[i,...])
        if row_sum==0.0:continue
        else:
            Ms[i, ...] = Ms[i,...]/row_sum
    return Ms

def semantic_graph_GEML(A):
    M, P, N, N = A.shape
    Ms = np.copy(A).astype(np.float32)
    for i in range(M):
        for j in range(P):
            Ms[i,j] = one_graph(Ms[i,j])
    return Ms


##############################  Each Model #########################
'''
功能：构建semantic graph (GEML)
      度向量+邻居矩阵，按行归一化，加入自环
输入：矩阵(...,N,N)
'''
def xx_one_graph(A):
    N,_ = A.shape
    Ms = np.zeros_like(A).astype(np.float32)
    D_In = np.sum(A, axis=-1)  # (N,)
    D_Out = np.sum(A, axis=-2)  # (N,)
    for i in range(N):
        for j in range(i,N):
            if i == j: continue
            if A[i, j] > 0.0 :
                Ms[i, j] = D_Out[j]
            if A[j, i] > 0.0:
                Ms[j, i] = D_In[j]

    for i in range(N):
        row_sum = np.sum(A[i,...])
        if row_sum==0.0:continue
        else:
            Ms[i, ...] = A[i,...]/row_sum
    return Ms

def xx_graph_GEML(A):
    M, P, N, N = A.shape
    Ms = np.copy(A).astype(np.float32)
    for i in range(M):
        for j in range(P):
            Ms[i,j] = xx_one_graph(Ms[i,j])
    return Ms

'''
功能：构建distance graph (GEML)
      距离图（无自环），按行归一化，加入自环
输入：矩阵(...,N,N)
'''
def distance_graph_GEML(A):
    M = np.copy(A).astype(np.float32)
    M = del_diag(M) # 去掉自环
    M = normalize_attention(M, axis=0) #按行归一化
    M = self_loop(M).astype(np.float32) #加入自环
    return M


##################################### model graph ##########################
def GCN_graph(args,data_dir):
    path = os.path.join(args.dataset_dir, 'original', args.graph_name)
    graph = np.load(path)['arr_0']#(N,N)
    path = os.path.join(data_dir, args.graph_name)
    np.savez_compressed(path, graph=graph)