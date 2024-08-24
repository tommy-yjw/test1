import random

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from scipy.sparse import lil_matrix, csr_matrix, diags
from tqdm import tqdm
from numba import njit

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
    # 随机选择边的方向
    if random.choice([True, False]):
        adj_matrix[node1, node2] = 1  # 从 node1 到 node2 的有向边
    else:
        adj_matrix[node2, node1] = 1  # 从 node2 到 node1 的有向边

# 转换为 CSR 格式
adj_matrix = adj_matrix.tocsr()

# 全联通验证
out_degrees = adj_matrix.sum(axis=1).A1
in_degrees = adj_matrix.sum(axis=0).A1

if 0 in out_degrees or 0 in in_degrees:
    raise ValueError("Graph is not fully connected. There are nodes with no outgoing or incoming edges.")
adj_matrix_1 = adj_matrix
degree_matrix_1 = diags(np.array(adj_matrix.sum(axis=1)).flatten(), 0)

# 计算稀疏拉普拉斯矩阵
laplacian_matrix_1 = degree_matrix_1 - adj_matrix
# 转换为密集邻接矩阵
dense_adj_matrix = adj_matrix_1.toarray()

# 随机增加一些边
num_add_edges = 20  # 你可以根据需要调整这个值
for _ in range(num_add_edges):
    node1, node2 = random.sample(range(num_nodes_1), 2)
    adj_matrix[node1, node2] = 1

# 随机减少一些边
num_remove_edges = 20  # 你可以根据需要调整这个值
for _ in range(num_remove_edges):
    node1, node2 = random.sample(range(num_nodes_1), 2)
    adj_matrix[node1, node2] = 0

adj_matrix = adj_matrix.tocsr()  # 转换回 csr_matrix
# # 再次全联通验证
# out_degrees = adj_matrix.sum(axis=1).A1
# in_degrees = adj_matrix.sum(axis=0).A1
#
# if 0 in out_degrees or 0 in in_degrees:
#     raise ValueError("Graph is not fully connected after adding and removing edges.")

degree_matrix_2 = diags(np.array(adj_matrix.sum(axis=1)).flatten(), 0)

# 计算稀疏拉普拉斯矩阵
laplacian_matrix_2 = degree_matrix_2 - adj_matrix
# 转换为密集邻接矩阵
dense_adj_matrix = adj_matrix.toarray()

# 读取CSV文件
df2 = pd.read_csv('size-3-unique-sorted.csv')

# 获取节点列表
nodes_2 = set(df2['node_1']).union(set(df2['node_2'])).union(set(df2['node_3']))
node_index_2 = {node: idx for idx, node in enumerate(nodes_2)}

# 初始化稀疏拉普拉斯矩阵列表
num_nodes_2 = len(nodes_2)
laplacian_matrices_1 = [lil_matrix((num_nodes_1, num_nodes_1), dtype='float32') for _ in range(num_nodes_1)]
laplacian_matrices_2 = [lil_matrix((num_nodes_1, num_nodes_1), dtype='float32') for _ in range(num_nodes_1)]

for _, row in tqdm(df2.iterrows(), total=df2.shape[0], desc="Filling Laplacian matrices_1"):
    node1 = node_index_2[row['node_1']]
    node2 = node_index_2[row['node_2']]
    node3 = node_index_2[row['node_3']]

    # 更新稀疏拉普拉斯矩阵
    if adj_matrix[node2, node1] > 0 and adj_matrix[node3, node1] > 0:
        laplacian_matrices_1[node1] = laplacian_matrices_1[node1].tolil()  # 转换为 lil_matrix
        laplacian_matrices_1[node1][node2, node3] += 1
        laplacian_matrices_1[node1] = laplacian_matrices_1[node1].tocsr()  # 转换回 csr_matrix
    if adj_matrix[node1, node2] > 0 and adj_matrix[node3, node2] > 0:
        laplacian_matrices_1[node2] = laplacian_matrices_1[node2].tolil()
        laplacian_matrices_1[node2][node1, node3] += 1
        laplacian_matrices_1[node2] = laplacian_matrices_1[node2].tocsr()
    if adj_matrix[node1, node3] > 0 and adj_matrix[node2, node3] > 0:
        laplacian_matrices_1[node3] = laplacian_matrices_1[node3].tolil()
        laplacian_matrices_1[node3][node1, node2] += 1
        laplacian_matrices_1[node3] = laplacian_matrices_1[node3].tocsr()

# 转换为 CSR 格式
laplacian_matrices_1 = [mat.tocsr() for mat in laplacian_matrices_1]

# 填充稀疏拉普拉斯矩阵
for _, row in tqdm(df2.iterrows(), total=df2.shape[0], desc="Filling Laplacian matrices_2"):
    node1 = node_index_2[row['node_1']]
    node2 = node_index_2[row['node_2']]
    node3 = node_index_2[row['node_3']]

    if adj_matrix_1[node2, node1] > 0 and adj_matrix_1[node3, node1] > 0:
        laplacian_matrices_2[node1] = laplacian_matrices_2[node1].tolil()
        # 更新稀疏拉普拉斯矩阵
        laplacian_matrices_2[node1][node2, node3] -= 1
        laplacian_matrices_2[node1][node1, node1] += 1  # 每个三元组对node1的影响
        laplacian_matrices_2[node1] = laplacian_matrices_2[node1].tocsr()
    if adj_matrix_1[node1, node2] > 0 and adj_matrix_1[node3, node2] > 0:
        laplacian_matrices_2[node2] = laplacian_matrices_2[node2].tolil()
        laplacian_matrices_2[node2][node1, node3] -= 1
        laplacian_matrices_2[node2][node2, node2] += 1  # 每个三元组对node2的影响
        laplacian_matrices_2[node2] = laplacian_matrices_2[node2].tocsr()
    if adj_matrix_1[node2, node3] > 0 and adj_matrix_1[node1, node3] > 0:
        laplacian_matrices_2[node3] = laplacian_matrices_2[node3].tolil()
        laplacian_matrices_2[node3][node1, node2] -= 1
        laplacian_matrices_2[node3][node3, node3] += 1  # 每个三元组对node3的影响
        laplacian_matrices_2[node3] = laplacian_matrices_2[node3].tocsr()

# 转换为 CSR 格式
laplacian_matrices_2 = [mat.tocsr() for mat in laplacian_matrices_2]

# 初始条件（列矩阵x和o）
x0 = np.random.rand(num_nodes_1)  # 确保x0是一维数组
o0 = np.random.rand(num_nodes_1)  # 确保o0是一维数组
abandon_rate = np.random.rand(num_nodes_1)  # 随机丢弃率
theta_coeff = 0.7
lambda_coeff = 0.7
gamma = 0.1  # 随机生成gamma
beta_ii = np.random.rand(num_nodes_1)  # 随机生成beta_ii
beta_iii = np.random.rand(num_nodes_1)  # 随机生成beta_iii


# 定义微分方程
def dynamics_1(y, t, adj_matrix, laplacian_matrix_1, abandon_rate, theta_coeff,
               lambda_coeff, gamma, beta_ii):
    x = y[:num_nodes_1]
    o = y[num_nodes_1:]

    x_dot = -abandon_rate * x * (1 - o) + (1 - x) * o * (adj_matrix_1.dot(x) + beta_ii)

    o_dot = - laplacian_matrix_1.dot(o) + gamma * x - o

    return np.concatenate((x_dot, o_dot))


def dynamics_2(y, t, adj_matrix, laplacian_matrices_1, laplacian_matrix_1, laplacian_matrices_2, abandon_rate,
               theta_coeff,
               lambda_coeff, gamma, beta_ii, beta_iii):
    x = y[:num_nodes_1]
    o = y[num_nodes_1:]
    x_dot = np.zeros_like(x)
    o_dot = np.zeros_like(o)
    for i, B_i in enumerate(laplacian_matrices_1):
        x_dot[i] = -abandon_rate[i] * x[i] * (1 - o[i]) + theta_coeff * (1 - x[i]) * o[i] * (
                    adj_matrix[i].dot(x) + beta_ii[i]) + (1 - theta_coeff) * (1 - x[i]) * o[i] * (
                               x.T.dot(B_i.dot(x)) + beta_iii[i])
    for i, B_i in enumerate(laplacian_matrices_2):
        o_dot[i] = -lambda_coeff * laplacian_matrix_1[i].dot(o) - (1 - lambda_coeff) * o.T.dot(B_i.dot(o)) + gamma * x[
            i] - o[i]

    return np.concatenate((x_dot, o_dot))


# 时间范围
t_max = 5000  # 最大时间范围
t_steps = 5000  # 时间步数
t = np.linspace(0, t_max, t_steps)
t_end_x1 = 0
t_end_o1 = 0
t_end_x2 = 0
t_end_o2 = 0
# 初始条件
y0 = np.concatenate((x0, o0))

# laplacian_matrices_1_data = np.array([mat.data for mat in laplacian_matrices_1])
# laplacian_matrices_1_indices = np.array([mat.indices for mat in laplacian_matrices_1])
# laplacian_matrices_1_indptr = np.array([mat.indptr for mat in laplacian_matrices_1])
#
# laplacian_matrices_2_data = np.array([mat.data for mat in laplacian_matrices_2])
# laplacian_matrices_2_indices = np.array([mat.indices for mat in laplacian_matrices_2])
# laplacian_matrices_2_indptr = np.array([mat.indptr for mat in laplacian_matrices_2])

# 求解ODE
with tqdm(total=2, desc="Solving ODEs") as pbar:
    y_sol_1 = odeint(dynamics_1, y0, t,
                     args=(adj_matrix, laplacian_matrix_1, abandon_rate, theta_coeff, lambda_coeff, gamma, beta_ii))

    pbar.update(1)
    y_sol_2 = odeint(dynamics_2, y0, t, args=(
    adj_matrix, laplacian_matrices_1, laplacian_matrix_1, laplacian_matrices_2, abandon_rate, theta_coeff, lambda_coeff,
    gamma, beta_ii, beta_iii))
    pbar.update(1)

x_sol_1 = y_sol_1[:, :num_nodes_1]
o_sol_1 = y_sol_1[:, num_nodes_1:]
x_sol_2 = y_sol_2[:, :num_nodes_1]
o_sol_2 = y_sol_2[:, num_nodes_1:]

tolerance = 0.00001
converged = False
# 计算收敛时间
for i in tqdm(range(1, len(t)), desc="Checking convergence for x_sol_1"):
    for j in range(num_nodes_1):
        if np.abs(x_sol_1[i][j] - x_sol_1[i - 1][j]) >= tolerance:
            break
    else:  # 这个 else 对应于 for 循环，表示 for 循环没有被 break 中断
        t_end_x1 = i
        converged = True
        break

if not converged:
    t_end_x1 = t_max

for i in tqdm(range(1, len(t)), desc="Checking convergence for o_sol_1"):
    for j in range(num_nodes_1):
        if np.abs(o_sol_1[i][j] - o_sol_1[i - 1][j]) >= tolerance:
            break
    else:  # 这个 else 对应于 for 循环，表示 for 循环没有被 break 中断
        t_end_o1 = i
        converged = True
        break

if not converged:
    t_end_o1 = t_max

# 计算收敛时间
for i in tqdm(range(1, len(t)), desc="Checking convergence for o_sol_1"):
    for j in range(num_nodes_1):
        if np.abs(x_sol_2[i][j] - x_sol_2[i - 1][j]) >= tolerance:
            break
    else:  # 这个 else 对应于 for 循环，表示 for 循环没有被 break 中断
        t_end_x2 = i
        converged = True
        break

if not converged:
    t_end_x2 = t_max

for i in tqdm(range(1, len(t)), desc="Checking convergence for o_sol_2"):
    for j in range(num_nodes_1):
        if np.abs(o_sol_2[i][j] - o_sol_2[i - 1][j]) >= tolerance:
            break
    else:  # 这个 else 对应于 for 循环，表示 for 循环没有被 break 中断
        t_end_o2 = i
        converged = True
        break

if not converged:
    t_end_o2 = t_max

if t_end_x1 < t_end_x2:
    t_final_x = t_end_x2
else:
    t_final_x = t_end_x1

if t_end_o1 < t_end_o2:
    t_final_o = t_end_o2
else:
    t_final_o = t_end_o1

# 绘制x的结果

plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0, t_final_x, len(x_sol_1)), x_sol_1[:, 0], label='higher-order coeff=0', color='blue')
plt.plot(np.linspace(0, t_final_x, len(x_sol_2)), x_sol_2[:, 0], label='higher-order coeff=0.6', color='orange')
for i in range(num_nodes_1):
    plt.plot(np.linspace(0, t_final_x, len(x_sol_1)), x_sol_1[:, i], color='blue')
    plt.plot(np.linspace(0, t_final_x, len(x_sol_2)), x_sol_2[:, i], color='orange')

plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Dynamics of x')
plt.legend()
plt.show()

# 绘制o的结果
plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0, t_final_o, len(o_sol_1)), o_sol_1[:, 0], label='higher-order coeff=0', color='blue')
plt.plot(np.linspace(0, t_final_o, len(o_sol_2)), o_sol_2[:, 0], label='higher-order coeff=0.6', color='orange')
for i in range(num_nodes_1):
    plt.plot(np.linspace(0, t_final_o, len(o_sol_1)), o_sol_1[:, i], color='blue')
    plt.plot(np.linspace(0, t_final_o, len(o_sol_2)), o_sol_2[:, i], color='orange')

plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Dynamics of o')
plt.legend()
plt.show()
print(f"adoption without higher-order 的收敛时间是: {t_end_x1}")
print(f"adoption with higher-order=0.6 的收敛时间是: {t_end_x2}")
print(f"opinion without higher-order 的收敛时间是: {t_end_o1}")
print(f"opinion with higher-order=0.6 的收敛时间是: {t_end_o2}")
print(num_nodes_1)