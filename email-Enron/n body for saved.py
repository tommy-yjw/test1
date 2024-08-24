import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from scipy.sparse import lil_matrix, csr_matrix, diags
from tqdm import tqdm
import random

# 读取CSV文件
df1 = pd.read_csv('size-2-unique-sorted.csv')

# 获取节点列表
nodes_1 = set(df1['node_1']).union(set(df1['node_2']))
node_index_1 = {node: idx for idx, node in enumerate(nodes_1)}

# 初始化稀疏邻接矩阵
num_nodes_1 = len(nodes_1)
adj_matrix = lil_matrix((num_nodes_1, num_nodes_1), dtype='float32')

# 填充稀疏邻接矩阵
for _, row in tqdm(df1.iterrows(), total=df1.shape[0], desc="Filling adjacency matrix"):
    node1 = node_index_1[row['node_1']]
    node2 = node_index_1[row['node_2']]
    adj_matrix[node1, node2] = 1
    adj_matrix[node2, node1] = 1

# 转换为 CSR 格式
adj_matrix = adj_matrix.tocsr()

# 计算稀疏度矩阵
degree_matrix = diags(np.array(adj_matrix.sum(axis=1)).flatten(), 0)

# 计算稀疏拉普拉斯矩阵
laplacian_matrix = degree_matrix - adj_matrix

# 初始条件
x0 = np.random.rand(num_nodes_1)

# 读取CSV文件
df_list = [pd.read_csv(f'size-{i}-unique-sorted.csv') for i in range(3, 9)]

# 获取节点列表
nodes_list = [set(df.iloc[:, :df.shape[1]].values.flatten()) for df in df_list]
node_index_list = [{node: idx for idx, node in enumerate(nodes)} for nodes in nodes_list]

# 创建 result_i 字典
result_3 = {i: np.zeros(num_nodes_1, dtype='float32') for i in range(6)}
result_4 = {i: np.zeros(num_nodes_1, dtype='float32') for i in range(5)}
result_5 = {i: np.zeros(num_nodes_1, dtype='float32') for i in range(4)}
result_6 = {i: np.zeros(num_nodes_1, dtype='float32') for i in range(3)}
result_7 = {i: np.zeros(num_nodes_1, dtype='float32') for i in range(2)}
result_8 = {i: np.zeros(num_nodes_1, dtype='float32') for i in range(1)}

# 定义更新结果的函数
def update_result_3(x0, df, node_index, result):
    for i in range(6):
        result_3[i][:] = 0
        for _, row in df.iterrows():
            nodes = [node_index[row[f'node_{i}']] for i in range(1, 4)]
            product = np.prod([x0[node] for node in nodes[1:]])
            for j in range(len(nodes)):
                result_3[i][nodes[j]] += product - x0[nodes[j]] ** 2
    return result

def update_result_4(x0, df, node_index, result):
    for i in range(5):
        result_4[i][:] = 0
        for _, row in df.iterrows():
            nodes = [node_index[row[f'node_{i}']] for i in range(1, 5)]
            product = np.prod([x0[node] for node in nodes[1:]])
            for j in range(len(nodes)):
                result[i][nodes[j]] += product - x0[nodes[j]] ** 3
    return result

def update_result_5(x0, df, node_index, result):
    for i in range(4):
        result_5[i][:] = 0
        for _, row in df.iterrows():
            nodes = [node_index[row[f'node_{i}']] for i in range(1, 6)]
            product = np.prod([x0[node] for node in nodes[1:]])
            for j in range(len(nodes)):
                result[i][nodes[j]] += product - x0[nodes[j]] ** 4
    return result

def update_result_6(x0, df, node_index, result):
    for i in range(3):
        result_6[i][:] = 0
        for _, row in df.iterrows():
            nodes = [node_index[row[f'node_{i}']] for i in range(1, 7)]
            product = np.prod([x0[node] for node in nodes[1:]])
            for j in range(len(nodes)):
                result[i][nodes[j]] += product - x0[nodes[j]] ** 5
    return result

def update_result_7(x0, df, node_index, result):
    for i in range(2):
        result_7[i][:] = 0
        for _, row in df.iterrows():
            nodes = [node_index[row[f'node_{i}']] for i in range(1, 8)]
            product = np.prod([x0[node] for node in nodes[1:]])
            for j in range(len(nodes)):
                result[i][nodes[j]] += product - x0[nodes[j]] ** 6
    return result

def update_result_8(x0, df, node_index, result):
    for i in range(1):
        result_8[i][:] = 0
        for _, row in df.iterrows():
            nodes = [node_index[row[f'node_{i}']] for i in range(1, 9)]
            product = np.prod([x0[node] for node in nodes[1:]])
            for j in range(len(nodes)):
                result[i][nodes[j]] += product - x0[nodes[j]] ** 7
    return result

# 定义动力学方程
def dynamics2(x, t, laplacian_matrix, df_list, node_index_list, lambda_coeffs):
    final_result = np.zeros_like(x)
    return -laplacian_matrix.dot(x) + final_result

def dynamics3(x, t, laplacian_matrix, df_list, node_index_list, lambda_coeffs):
    final_result = np.zeros_like(x)
    final_result += 0.9 * update_result_3(x, df_list[0], node_index_list[0], result_3[0])
    return -0.1 * laplacian_matrix.dot(x) + final_result

def dynamics4(x, t, laplacian_matrix, df_list, node_index_list, lambda_coeffs):
    final_result = np.zeros_like(x)
    final_result += lambda_coeffs[0] * update_result_3(x, df_list[0], node_index_list[0], result_3[1])
    final_result += (0.9 -lambda_coeffs[0]) * update_result_4(x, df_list[1], node_index_list[1], result_4[0])
    return -0.1 * laplacian_matrix.dot(x) + final_result

def dynamics5(x, t, laplacian_matrix, df_list, node_index_list, lambda_coeffs):
    final_result = np.zeros_like(x)
    final_result += lambda_coeffs[0] * update_result_3(x, df_list[0], node_index_list[0], result_3[2])
    final_result += lambda_coeffs[1] * update_result_4(x, df_list[1], node_index_list[1], result_4[1])
    final_result += (0.9 - lambda_coeffs[0] - lambda_coeffs[1]) * update_result_5(x, df_list[2], node_index_list[2], result_5[0])
    return -0.1 * laplacian_matrix.dot(x) + final_result

def dynamics6(x, t, laplacian_matrix, df_list, node_index_list, lambda_coeffs):
    final_result = np.zeros_like(x)
    final_result += lambda_coeffs[0] * update_result_3(x, df_list[0], node_index_list[0], result_3[3])
    final_result += lambda_coeffs[1] * update_result_4(x, df_list[1], node_index_list[1], result_4[2])
    final_result += lambda_coeffs[2] * update_result_5(x, df_list[2], node_index_list[2], result_5[1])
    final_result += (0.9 - lambda_coeffs[0] - lambda_coeffs[1] - lambda_coeffs[2]) * update_result_6(x, df_list[3], node_index_list[3], result_6[0])
    return -0.1 * laplacian_matrix.dot(x) + final_result

def dynamics7(x, t, laplacian_matrix, df_list, node_index_list, lambda_coeffs):
    final_result = np.zeros_like(x)
    final_result += lambda_coeffs[0] * update_result_3(x, df_list[0], node_index_list[0], result_3[4])
    final_result += lambda_coeffs[1] * update_result_4(x, df_list[1], node_index_list[1], result_4[3])
    final_result += lambda_coeffs[2] * update_result_5(x, df_list[2], node_index_list[2], result_5[2])
    final_result += lambda_coeffs[3] * update_result_6(x, df_list[3], node_index_list[3], result_6[1])
    final_result += lambda_coeffs[4] * update_result_7(x, df_list[4], node_index_list[4], result_7[0])
    final_result += (0.9 - lambda_coeffs[0] - lambda_coeffs[1] - lambda_coeffs[2] - lambda_coeffs[3]) * update_result_7(x, df_list[4], node_index_list[4], result_7[0])
    return -0.1 * laplacian_matrix.dot(x) + final_result

def dynamics8(x, t, laplacian_matrix, df_list, node_index_list, lambda_coeffs):
    final_result = np.zeros_like(x)
    final_result += lambda_coeffs[0] * update_result_3(x, df_list[0], node_index_list[0], result_3[5])
    final_result += lambda_coeffs[1] * update_result_4(x, df_list[1], node_index_list[1], result_4[4])
    final_result += lambda_coeffs[2] * update_result_5(x, df_list[2], node_index_list[2], result_5[3])
    final_result += lambda_coeffs[3] * update_result_6(x, df_list[3], node_index_list[3], result_6[2])
    final_result += lambda_coeffs[4] * update_result_7(x, df_list[4], node_index_list[4], result_7[1])
    final_result += lambda_coeffs[5] * update_result_8(x, df_list[5], node_index_list[5], result_8[0])
    return -0.1 * laplacian_matrix.dot(x) + final_result

# 时间范围
t_max = 35
t_steps = 3500
t = np.linspace(0, t_max, t_steps)

# 系数之和为1
lambda_coeffs = np.array([0.1, 0.1, 0.2, 0.15, 0.15, 0.1, 0.1])
x_sol_list = []
# 求解ODE

for i in range(2, 9):
    with tqdm(total=t_steps, desc=f"Solving dynamics{i}") as pbar:
        if i == 2:
            df = pd.read_csv(f'size-{i}-unique-sorted.csv')
            nodes = set(df.iloc[:, :df.shape[1]].values.flatten())
            node_index = {node: idx for idx, node in enumerate(nodes)}
            x_sol = odeint(globals()[f'dynamics{i}'], x0, t, args=(laplacian_matrix, [df], [node_index], [lambda_coeffs[i-2]]))
        else:
            x_sol = odeint(globals()[f'dynamics{i}'], x0, t, args=(laplacian_matrix, df_list[:i-2], node_index_list[:i-2], lambda_coeffs[:i-2]))
        pbar.update(t_steps)  # 一次性更新进度条
    x_sol_list.append(x_sol)
# with tqdm(total=7, desc="Solving ODEs") as pbar:
#     x_sol_1 = odeint(dynamics2, x0, t, args=(laplacian_matrix, df_list, node_index_list, lambda_coeffs))
#     pbar.update(1)
#     x_sol_2 = odeint(dynamics3, x0, t, args=(laplacian_matrix, df_list, node_index_list, lambda_coeffs))
#     pbar.update(1)
#     x_sol_3 = odeint(dynamics4, x0, t, args=(laplacian_matrix, df_list, node_index_list, lambda_coeffs))
#     pbar.update(1)
#     x_sol_4 = odeint(dynamics5, x0, t, args=(laplacian_matrix, df_list, node_index_list, lambda_coeffs))
#     pbar.update(1)
#     x_sol_5 = odeint(dynamics6, x0, t, args=(laplacian_matrix, df_list, node_index_list, lambda_coeffs))
#     pbar.update(1)
#     x_sol_6 = odeint(dynamics7, x0, t, args=(laplacian_matrix, df_list, node_index_list, lambda_coeffs))
#     pbar.update(1)
#     x_sol_7 = odeint(dynamics8, x0, t, args=(laplacian_matrix, df_list, node_index_list, lambda_coeffs))
#     pbar.update(1)
# x_sol_list = [x_sol_1, x_sol_2, x_sol_3, x_sol_4, x_sol_5, x_sol_6, x_sol_7]

# 绘制结果
plt.figure(figsize=(10, 6))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
for i, x_sol in enumerate(x_sol_list):
    color = colors[i]
    for j in range(x_sol.shape[1]):
        plt.plot(t, x_sol[:, j], color=color)
plt.xlabel('Time')
plt.ylabel('x')
plt.title('Multi-body dynamics')
plt.legend()
plt.show()
