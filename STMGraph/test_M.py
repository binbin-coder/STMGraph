import numpy as np
from sklearn.neighbors import NearestNeighbors

# 假设我们有一个样本集X
# X = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
# A = np.array([[1],[1],[1]])
#
# print(X+X.T)
#
# # 构建最近邻居模型
# n_neighbors = 2
# knn_model = NearestNeighbors(n_neighbors=n_neighbors)
# knn_model.fit(X)
#
# # 计算每个样本的最近邻居
# distances, indices = knn_model.kneighbors(X)
#
# # 构建邻接矩阵
# adjacency_matrix = np.zeros((len(X), len(X)), dtype=int)
#
# for i in range(len(indices)):
#     for j in range(1, n_neighbors):
#         adjacency_matrix[i, indices[i, j]] = 1
#
# print(adjacency_matrix)
# import torch
# from torch.nn.parameter import Parameter
# # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # print(device)
# X = torch.FloatTensor(X)
#
# print(X.unsqueeze(1))
#
# print(Parameter(torch.Tensor(5, 10)))
# - mu) ** 2
# def ccc(in_features: int, out_features: int, n_heads: int):
#     print(out_features)
# ccc(3,'3',3)
# for lay1,lay2 in zip(range(4-1,-1,-1),range(4-1)):
#     print(lay1,lay2)
#
# for lay1 in range(2):
#     print(lay1)

# import numpy as np
# def apply_mask(matrix, mask):
#     return np.matmul(matrix, mask)
# #
# # 示例输入矩阵 X
# X = np.array([[1, 2, 3,3,5],
#               [4, 5, 6,3,9],
#               [7, 8, 9,3,0]])
#
# A = np.array([[1, 2, 3,2,3],
#               [4, 5, 6,2,3],
#               [7, 8, 92,2,8],
#               [7, 8, 92,2,8],
#               [4, 5, 6,2,3]])
#
# # 创建掩码矩阵
# mask_matrix = create_mask_matrix(A.shape)
# # mask_matrix = create_masked_tensor(X, A, 0.5)
# print(mask_matrix)
# # 应用掩码矩阵
# masked_matrix = apply_mask(X, mask_matrix)
#
# print("X:")
# print(X)
# print("\nMask Matrix:")
# print(mask_matrix)
# print("\nMasked Matrix:")
# print(masked_matrix)

# def create_masked_tensor(X, drop=0.5):
#     mask = tf.eye(X.shape[-1])
#     mask = tf.dtypes.cast(mask, tf.float32)
#     mask = tf.where(tf.random.uniform(X.shape) < drop, tf.zeros_like(mask), mask)
#     print(X)
#     print(mask)
#     masked_X = tf.matmul(X, mask)
#     return masked_X
#
#
# # 示例代码
# input_shape = (5, 5)  # 假设输入张量的形状为 10x10
# X = tf.constant(np.random.random(input_shape).astype(np.float32))
# masked_X = create_masked_tensor(X, drop=0.1)
# print(masked_X)
# import tensorflow.compat.v1 as tf
# def create_mask_matrix(matrix_shape, drop=0.5):
#     mask = np.eye(matrix_shape[0],matrix_shape[1])
#     n = min(matrix_shape)
#     num_ones = int(n * drop)  # 计算对角线上1的数量
#     indices = np.random.choice(np.arange(n), size=num_ones, replace=False)  # 随机选择一部分索引
#     mask[indices, indices] = 0  # 将选定的索引位置上的元素设为0
#     mask = tf.constant(mask, dtype=tf.float32)  # 转换为 TensorFlow 张量
#     return mask

# def create_mask_matrix(matrix_shape, drop=0.5):
#     mask = tf.eye(matrix_shape[0], matrix_shape[1])  # 使用 TensorFlow 的 eye 函数创建单位矩阵
#     n = tf.minimum(matrix_shape[0], matrix_shape[1])  # 使用 TensorFlow 的 minimum 函数
#     n_float = tf.cast(n, tf.float32)
#     float_num_ones = n_float * tf.constant(drop, dtype=tf.float32)
#     num_ones = tf.cast(float_num_ones, tf.int32)  # 强制类型转换为整数
#     indices = tf.random.shuffle(tf.range(n))[:num_ones]  # 使用 TensorFlow 的随机函数
#     expanded_updates = tf.zeros([num_ones], dtype=mask.dtype)
#     mask = tf.tensor_scatter_nd_update(mask, indices[:, tf.newaxis], expanded_updates)
#     # mask = tf.tensor_scatter_nd_update(mask, indices[:, tf.newaxis], tf.zeros(num_ones))  # 更新选定的位置为0
#     return mask
#
# # 示例代码
# mask_shape = (10, 10)  # 假设掩码的形状为 10x10
# A = tf.constant([[1, 2, 3, 2, 3],
#                 [4, 5, 6, 2, 3],
#                 [7, 8, 92, 2, 8],
#                 [7, 8, 92, 2, 8],
#                 [4, 5, 6, 2, 3]], dtype=tf.float32)
# print(A.shape[0],A.shape[1])
# mask = create_mask_matrix(mask_shape, drop=0.5)
# print(mask.shape)
# print(mask)

# def create_mask_matrix(input_dim, drop=0.2):
#     # 构造单位矩阵
#     unit_matrix = np.eye(input_dim, dtype=np.float32)
#     mask = tf.convert_to_tensor(unit_matrix)
#
#     # 根据 drop 的比例生成新的 mask 矩阵
#     num_zeros = int(input_dim * drop)  # 需要置零的元素数量
#     mask_indices = tf.random_shuffle(tf.range(input_dim))[:num_zeros]  # 随机选择要置零的索引
#     print(mask_indices)
#     mask = tf.tensor_scatter_nd_update(mask, indices=tf.expand_dims(mask_indices, axis=1), updates=tf.zeros(num_zeros))  # 将选定的位置上的元素设为0
#     return mask

# import tensorflow.compat.v1 as tf
# def create_mask_matrix(matrix_shape, drop=0.9):
#     mask = tf.eye(matrix_shape[0], matrix_shape[1])
#     n = min(matrix_shape)
#     num_ones = int(n * drop)  # 计算对角线上1的数量
#     indices = np.random.choice(np.arange(n), size=num_ones, replace=False)  # 随机选择一部分索引
#     print(indices)
#     updates = tf.zeros(num_ones, dtype=tf.float32)
#     mask = tf.tensor_scatter_nd_update(mask, indices[:, None] * matrix_shape[1] + indices[None, :], updates)
#     return mask
#
# mask_shape = (5, 5)  # 假设掩码的形状为 10x10

# print(A.shape[0],A.shape[1])
# mask = create_mask_matrix(mask_shape, drop=0.5)
# print(mask.shape)
# print(mask)


import tensorflow.compat.v1 as tf
import numpy as np
A = tf.constant([[1, 2, 3, 2, 3,4],
                [4, 5, 6, 2, 3,8],
                [1, 2, 3, 2, 3,3],
                [7, 8, 92, 2, 8,0],
                [1, 2, 3, 2, 3,4],
                [1, 2, 3, 2, 3,2],
                [7, 8, 92, 2, 8,1],
                [4, 5, 6, 2, 3,1],
                [7, 8, 92, 2, 8,6],
                [4, 5, 6, 2, 3,4]], dtype=tf.float32)

B = tf.constant([[1, 2, 3, 2, 3],
                [4, 5, 6, 2, 3],
                [1, 2, 3, 2, 3],
                [7, 8, 92, 2, 8],
                [1, 2, 3, 2, 3],
                [1, 2, 3, 2, 3],
                [7, 8, 92, 2, 8],
                [4, 5, 6, 2, 3],
                [7, 8, 92, 2, 8],
                [4, 5, 6, 2, 3]], dtype=tf.float32)
# 假设 total_rows 为行数，drop_ratio 为要置零的比例

# 生成行屏蔽的掩码
def create_mask_matrix(X, drop_ratio=0.5,noise_ratio=0.2):
    total_rows = tf.shape(X)[0]
    total_rows = tf.cast(total_rows, tf.float32)

    num_drops = tf.cast(total_rows * drop_ratio, tf.float32)  # 需要置零的行数
    noise_num_drops = tf.cast(num_drops * noise_ratio, tf.float32)  # 需要置替换噪声的行数
    noise_num_drops = tf.cast(noise_num_drops, tf.int32)

    total_rows = tf.cast(total_rows, tf.int32)
    indices = tf.random.shuffle(tf.range(total_rows))  # 随机打乱行索引
    num_drops = tf.cast(num_drops, tf.int32)

    drop_indices = tf.gather(indices, tf.range(num_drops), axis=0)  # 选取需要置零的行的索引
    shuffled_indices = tf.random.shuffle(drop_indices)  # 随机打乱行索引
    # 创建掩码
    mask = tf.ones(total_rows, dtype=tf.float32)
    mask = tf.tensor_scatter_nd_update(mask, tf.expand_dims(drop_indices, axis=1), tf.zeros(num_drops))
    # 对 X 矩阵进行屏蔽
    masked_X = X * tf.expand_dims(mask, axis=1)  # 使用 broadcasting 将掩码应用到整个矩阵的每一行
    # 获取需要替换噪声的行的索引
    noise_indices = shuffled_indices[:noise_num_drops]
    # 从 X 矩阵中随机选择与 noise_indices 相同行数的行，并替换到 masked_X 中对应的行
    selected_noise = tf.gather(X, noise_indices)
    masked_X = tf.tensor_scatter_nd_update(masked_X, tf.expand_dims(noise_indices, axis=1), selected_noise)
    # 创建一个可学习的参数，形状与 drop_indices 的长度相同
    learnable_param = tf.Variable(tf.zeros((1,)), trainable=True)
    # learnable_param = tf.get_variable("learnable_param",tf.zeros((1, )))
    # 获取需要置零的行的子矩阵
    masked_rows = tf.gather(masked_X, drop_indices)
    # 将 learnable_param 加到 masked_rows 中
    updated_rows = masked_rows + tf.expand_dims(learnable_param, axis=1)
    # 使用 tf.tensor_scatter_nd_add 将更新后的子矩阵写回 masked_X 中
    masked_X = tf.tensor_scatter_nd_add(masked_X, tf.expand_dims(drop_indices, axis=1), updated_rows)
    return masked_X, drop_indices, num_drops
# print(create_mask_matrix(A,0.5,0.2))

# def create_mask_matrix2(X, filiter_num=5):
#     total_cols = tf.shape(X)[1]
#     if total_cols==filiter_num:
#         return X
#     total_cols = tf.cast(total_cols, tf.int32)
#     shuffled_indices = tf.random.shuffle(tf.range(total_cols))  # 随机打乱行索引
#     # shuffled_indices = tf.cast(shuffled_indices, tf.int32)
#     filiter_indices = shuffled_indices[:filiter_num]
#     selected_filiter = tf.gather(X, filiter_indices,axis=1)
#     return selected_filiter
# 使用 tf.concat 合并两个矩阵
def create_X_matrix(X, X0, dropout=0):
    total_cols = tf.shape(X)[1]
    total_cols0 = tf.shape(X0)[1]
    if dropout == 0:
        return X
    else:
        total_cols = tf.cast(total_cols, tf.float32)
        filiter_num = tf.cast(total_cols * dropout, tf.float32)  # 需要置零的行数
        total_cols = tf.cast(total_cols, tf.int32)
        filiter_num = tf.cast(filiter_num, tf.int32)
        filiter_num0 = total_cols - filiter_num
        total_cols = tf.cast(total_cols, tf.int32)
        total_cols0 = tf.cast(total_cols0, tf.int32)
        shuffled_indices = tf.random.shuffle(tf.range(total_cols))  # 随机打乱行索引
        # shuffled_indices = tf.cast(shuffled_indices, tf.int32)
        filiter_indices = shuffled_indices[:filiter_num]
        filiter_indices = tf.sort(filiter_indices)

        selected_filiter = tf.gather(X, filiter_indices, axis=1)
        shuffled_indices0 = tf.random.shuffle(tf.range(total_cols0))  # 随机打乱行索引

        filiter_indices0 = shuffled_indices0[:filiter_num0]
        filiter_indices0 = tf.sort(filiter_indices0)
        selected_filiter0 = tf.gather(X0, filiter_indices0, axis=1)

        merged_matrix = tf.concat([selected_filiter, selected_filiter0], axis=1)  # 假设你想按行合并
        return merged_matrix

merged_matrix = create_X_matrix(A,B,0.5)
print(merged_matrix)