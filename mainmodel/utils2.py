from __future__ import print_function
import scipy.sparse as sp
import numpy as np
import networkx as nx
from sklearn import preprocessing
from keras.utils import to_categorical
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
#from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="data/cora/", dataset="cora",use_feature=True):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(path))

    idx_features_labels=np.loadtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))

    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    g=nx.read_edgelist("{}{}.cites".format(path, dataset))
    N=len(g)
    adj=nx.to_numpy_array(g,nodelist=idx_features_labels[:, 0])
    adj = sp.coo_matrix(adj)

    if use_feature:
        features = np.array(idx_features_labels[:, 1:-1], dtype=np.float32)
    else:
        features=np.identity(N,dtype=np.float32)
    
    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], g.size(), features.shape[1]))

    return features, adj, labels


def normalize_adj(adj, symmetric=True):
    #adj是csr_array吗
    # 如果邻接矩阵为对称矩阵，得到对称归一化邻接矩阵
    # D^(-1/2) * A * D^(-1/2)
    #是这里的问题
    if symmetric:
        # 计算每一行非0元素的个数
        # 获取非零元素的索引
        row_idx, col_idx = adj.nonzero()
        # 计算每行非零元素的数量
        unique_row_idx, counts = np.unique(row_idx, return_counts=True)
        nonzero_counts = np.zeros(adj.shape[0])
        nonzero_counts[unique_row_idx] = counts
        # 将结果转换为二维数组
        nonzero_counts_2d = np.reshape(nonzero_counts, (-1, 1))
        d_inv_sqrt = np.power(nonzero_counts_2d, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d = sp.diags(d_inv_sqrt, 0)
        a_norm = adj.dot(d).transpose().dot(d).tocoo()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def get_splits(y):
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    
    y_train = np.zeros(y.shape, dtype=np.int32) #y:label
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)

    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]

    train_mask = sample_mask(idx_train, y.shape[0])
    val_mask = sample_mask(idx_val, y.shape[0])
    test_mask = sample_mask(idx_test, y.shape[0])

    return y_train, y_val, y_test, train_mask,val_mask,test_mask


def normalized_laplacian(adj, symmetric=True):
    # 对称归一化的邻接矩阵，D ^ (-1/2) * A * D ^ (-1/2)
    adj_normalized = normalize_adj(adj, symmetric)
    # # 得到对称规范化的图拉普拉斯矩阵，L = I - D ^ (-1/2) * A * D ^ (-1/2)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    return laplacian


def rescale_laplacian(laplacian):
    try:
        print('Calculating largest eigenvalue of normalized graph Laplacian...')
        largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    except ArpackNoConvergence:
        print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2
    # 调整后的对称归一化图拉普拉斯矩阵，L~ = 2 / Lambda * L - I
    scaled_laplacian = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
    return scaled_laplacian


def chebyshev_polynomial(X, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))
    # 返回一个稀疏矩阵列表
    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        """
                :param T_k_minus_one: T(k-1)(L~)
                :param T_k_minus_two: T(k-2)(L~)
                :param X: L~
                :return: Tk(L~)
                """
        # 将输入转化为csr矩阵（压缩稀疏行矩阵）
        X_ = sp.csr_matrix(X, copy=True)
        #  # 递归公式：Tk(L~) = 2L~ * T(k-1)(L~) - T(k-2)(L~)
        XX = 2*X_.dot(T_k_minus_one)-T_k_minus_two
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k+1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    return T_k


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features
