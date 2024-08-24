import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from scipy.sparse import lil_matrix, csr_matrix, diags
from tqdm import tqdm
from joblib import Parallel, delayed

# 读取CSV文件
df1 = pd.read_csv('size-2-unique-sorted.csv')

# 获取节点列表
nodes_1 = set(df1['node_1']).union(set(df1['node_2']))
node_index_1 = {node: idx for idx, node in enumerate(nodes_1)}

# 初始化稀疏邻接矩阵
num_nodes_1 = len(nodes_1)
adj_matrix = lil_matrix((num_nodes_1, num_nodes_1), dtype='float32')

# 填充稀疏邻接矩阵
def fill_adj_matrix(row):
    node1 = node_index_1[row['node_1']]
    node2 = node_index_1[row['node_2']]
    adj_matrix[node1, node2] = 1
    adj_matrix[node2, node1] = 1

_ = Parallel(n_jobs=-1)(delayed(fill_adj_matrix)(row) for _, row in tqdm(df1.iterrows(), total=df1.shape[0], desc="Filling adjacency matrix"))

# 转换为 CSR 格式
adj_matrix = adj_matrix.tocsr()

# 计算稀疏度矩阵
degree_matrix = diags(np.array(adj_matrix.sum(axis=1)).flatten(), 0)

# 计算稀疏拉普拉斯矩阵
laplacian_matrix = degree_matrix - adj_matrix

# 读取CSV文件
df2 = pd.read_csv('size-3-unique-sorted.csv')

# 获取节点列表
nodes_2 = set(df2['node_1']).union(set(df2['node_2'])).union(set(df2['node_3']))
node_index_2 = {node: idx for idx, node in enumerate(nodes_2)}

# 初始化稀疏拉普拉斯矩阵列表
num_nodes_2 = len(nodes_2)
laplacian_matrices = [lil_matrix((num_nodes_1, num_nodes_1), dtype='float32') for _ in range(num_nodes_1)]

# 填充稀疏拉普拉斯矩阵
def fill_laplacian_matrices(row):
    node1 = node_index_2[row['node_1']]
    node2 = node_index_2[row['node_2']]
    node3 = node_index_2[row['node_3']]

    # 更新稀疏拉普拉斯矩阵
    laplacian_matrices[node1][node2, node3] -= 1
    laplacian_matrices[node1][node1, node1] += 1  # 每个三元组对node1的影响

    laplacian_matrices[node2][node1, node3] -= 1
    laplacian_matrices[node2][node2, node2] += 1  # 每个三元组对node2的影响

    laplacian_matrices[node3][node1, node2] -= 1
    laplacian_matrices[node3][node3, node3] += 1  # 每个三元组对node3的影响

_ = Parallel(n_jobs=-1)(delayed(fill_laplacian_matrices)(row) for _, row in tqdm(df2.iterrows(), total=df2.shape[0], desc="Filling Laplacian matrices"))

# 转换为 CSR 格式
laplacian_matrices = [mat.tocsr() for mat in laplacian_matrices]

# 初始条件（列矩阵x）
x0 = np.random.rand(num_nodes_1)  # 确保x0是一维数组
lambda_coeff = 0.6

def dynamics1(x, t, laplacian_matrix):
    return -laplacian_matrix.dot(x)

# 定义动力学方程
def dynamics2(x, t, laplacian_matrix, laplacian_matrices, lambda_coeff):
    x_dot = np.zeros_like(x)
    for i, B_i in enumerate(laplacian_matrices):
        x_dot[i] = -lambda_coeff * laplacian_matrix[i].dot(x) - (1 - lambda_coeff) * x.T.dot(B_i.dot(x))
    return x_dot

# 时间范围
t_max = 10  # 最大时间范围
t_steps = 2000  # 时间步数
t = np.linspace(0, t_max, t_steps)

# 求解方程
with tqdm(total=2, desc="Solving ODEs") as pbar:
    x_sol_1 = odeint(dynamics1, x0, t, args=(laplacian_matrix,))

    pbar.update(1)
    x_sol_2 = odeint(dynamics2, x0, t, args=(laplacian_matrix, laplacian_matrices, lambda_coeff))
    pbar.update(1)

# 绘制结果
tolerance = 0.001
converged = False
for i in tqdm(range(1, len(t)), desc="Checking convergence for x_sol_1"):
    for j in range(num_nodes_1):
        if np.abs(x_sol_1[i][j] - x_sol_1[i-1][j]) >= tolerance:
            break
    else:  # 这个 else 对应于 for 循环，表示 for 循环没有被 break 中断
        t_end = i
        converged = True
        break

if not converged:
    t_end = t_max

for i in tqdm(range(1, len(t)), desc="Checking convergence for x_sol_2"):
    for j in range(num_nodes_1):
        if np.abs(x_sol_2[i][j] - x_sol_2[i-1][j]) >= tolerance:
            break
    else:  # 这个 else 对应于 for 循环，表示 for 循环没有被 break 中断
        t_end_2 = i
        converged = True
        break

if not converged:
    t_end_2 = t_max

# 计算收敛时间

# 输出结果
if t_end < t_end_2:
    t_final = t_end_2
else:
    t_final = t_end

plt.figure(figsize=(6, 8))
plt.plot(np.linspace(0, t_final, len(x_sol_1)), x_sol_1[:, 0], label=f'Higher-order coeff=0', color='blue')
plt.plot(np.linspace(0, t_final, len(x_sol_2)), x_sol_2[:, 0], label=f'Higher-order coeff={1-lambda_coeff}', color='orange')

for i in range(num_nodes_1):
    plt.plot(np.linspace(0, t_final, len(x_sol_1)), x_sol_1[:, i], color='blue')
    plt.plot(np.linspace(0, t_final, len(x_sol_2)), x_sol_2[:, i], color='orange')

plt.xlabel('Time')
plt.ylabel('x')
plt.title('bills house-three body dynamics')
plt.legend()
plt.show()
print("higher-order=0:\n", x_sol_1[t_end][0])
print(f"higher-order={lambda_coeff}:\n", x_sol_2[t_end_2][0])
print(f"x_sol_1 的收敛时间是: {t_end}")
print(f"x_sol_2 的收敛时间是: {t_end_2}")
