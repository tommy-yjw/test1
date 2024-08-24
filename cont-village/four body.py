import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from scipy.sparse import lil_matrix, csr_matrix, diags
from tqdm import tqdm

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
x0 = np.random.rand(num_nodes_1)  # 确保x0是一维数组
# 读取CSV文件
df2 = pd.read_csv('size-3-unique-sorted.csv')

# 获取节点列表
nodes_2 = set(df2['node_1']).union(set(df2['node_2'])).union(set(df2['node_3']))
node_index_2 = {node: idx for idx, node in enumerate(nodes_2)}

# 初始化稀疏拉普拉斯矩阵列表
num_nodes_2 = len(nodes_2)
result_1 = np.zeros(num_nodes_1, dtype='float32')
result_2 = np.zeros(num_nodes_1, dtype='float32')
laplacian_matrices = [lil_matrix((num_nodes_1, num_nodes_1), dtype='float32') for _ in range(num_nodes_1)]


# 填充稀疏拉普拉斯矩阵
def update_result_1(x0, result):
    result[:] = 0
    for _, row in df2.iterrows():
        node1 = node_index_2[row['node_1']]
        node2 = node_index_2[row['node_2']]
        node3 = node_index_2[row['node_3']]

        result[node1] += x0[node2] * x0[node3] - x0[node1] ** 2
        result[node2] += x0[node1] * x0[node3] - x0[node2] ** 2
        result[node3] += x0[node1] * x0[node2] - x0[node3] ** 2
    return result

# 转换为 CSR 格式
# laplacian_matrices = [mat.tocsr() for mat in laplacian_matrices]

# 读取CSV文件
df3 = pd.read_csv('size-4-unique-sorted.csv')

# 获取节点列表
nodes_3 = set(df3['node_1']).union(set(df3['node_2'])).union(set(df3['node_3'])).union(set(df3['node_4']))
node_index_3 = {node: idx for idx, node in enumerate(nodes_3)}

# 初始化稀疏拉普拉斯矩阵列表和连接矩阵列表
num_nodes_3 = len(nodes_3)
result = np.zeros(num_nodes_1, dtype='float32')

# 初始条件（列矩阵x）

result_3 = np.zeros(num_nodes_1, dtype='float32')


# 计算表达式 xj * xk * xl - xi^3
def update_result(x0, result):
    result[:] = 0
    for _, row in df3.iterrows():
        node1 = node_index_3[row['node_1']]
        node2 = node_index_3[row['node_2']]
        node3 = node_index_3[row['node_3']]
        node4 = node_index_3[row['node_4']]

        # 计算 xj * xk * xl - xi^3 对于每个节点
        result[node1] += x0[node2] * x0[node3] * x0[node4] - x0[node1] ** 3
        result[node2] += x0[node1] * x0[node3] * x0[node4] - x0[node2] ** 3
        result[node3] += x0[node1] * x0[node2] * x0[node4] - x0[node3] ** 3
        result[node4] += x0[node1] * x0[node2] * x0[node3] - x0[node4] ** 3
    return result

# 初始条件（列矩阵x）
lambda_coeff = 0.3

lambda_coeff_1 = 0.4

# 确保lambda_coeff_1和lambda_coeff_2系数之和为lambda_coeff

lambda_coeff_2 = 0.3


def dynamics1(x, t, laplacian_matrix):
    return -laplacian_matrix.dot(x)


# 定义动力学方程
def dynamics2(x, t, laplacian_matrix, result_1, lambda_coeff):
    return -lambda_coeff * laplacian_matrix.dot(x) + (1 - lambda_coeff) * update_result_1(x, result_1)


def dynamics3(x, t, laplacian_matrix, result_2, lambda_coeff, lambda_coeff_1, lambda_coeff_2, result_3):
    return (-lambda_coeff * laplacian_matrix.dot(x) + lambda_coeff_1 * update_result_1(x, result_2) +
            lambda_coeff_2 * update_result(x, result_3))


# 时间范围
t_max = 20  # 最大时间范围
t_steps = 3000  # 时间步数
t = np.linspace(0, t_max, t_steps)
t_end = 0
t_end_2 = 0
t_end_3 = 0
# 求解ODE
with tqdm(total=3, desc="Solving ODEs") as pbar:
    x_sol_1 = odeint(dynamics1, x0, t, args=(laplacian_matrix,))
    pbar.update(1)
    x_sol_2 = odeint(dynamics2, x0, t, args=(laplacian_matrix, result_1, lambda_coeff))
    pbar.update(1)
    x_sol_3 = odeint(dynamics3, x0, t, args=(laplacian_matrix, result_2, lambda_coeff, lambda_coeff_1, lambda_coeff_2, result_3))
    pbar.update(1)

# 绘制结果
tolerance = 0.001
converged = False
for i in tqdm(range(1, len(t)), desc="Checking convergence for x_sol_1"):
    for j in range(num_nodes_1):
        if np.abs(x_sol_1[i][j] - x_sol_1[i - 1][j]) >= tolerance:
            break
    else:  # 这个 else 对应于 for 循环，表示 for 循环没有被 break 中断
        t_end = i
        converged = True
        break

if not converged:
    t_end = t_max

for i in tqdm(range(1, len(t)), desc="Checking convergence for x_sol_2"):
    for j in range(num_nodes_1):
        if np.abs(x_sol_2[i][j] - x_sol_2[i - 1][j]) >= tolerance:
            break
    else:  # 这个 else 对应于 for 循环，表示 for 循环没有被 break 中断
        t_end_2 = i
        converged = True
        break

if not converged:
    t_end_2 = t_max

for i in tqdm(range(1, len(t)), desc="Checking convergence for x_sol_3"):
    for j in range(num_nodes_1):
        if np.abs(x_sol_3[i][j] - x_sol_3[i - 1][j]) >= tolerance:
            break
    else:  # 这个 else 对应于 for 循环，表示 for 循环没有被 break 中断
        t_end_3 = i
        converged = True
        break

if not converged:
    t_end_3 = t_max

# 计算收敛时间

# 比较t_end,t_end_2和t_end_3,哪个最大就作为t_final
t_final = max(t_end, t_end_2, t_end_3)

plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0, t_final, len(x_sol_1)), x_sol_1[:, 0], label=f'Higher-order coeff_2=0,Higher-order coeff_3=0',
         color='blue')
plt.plot(np.linspace(0, t_final, len(x_sol_2)), x_sol_2[:, 0],
         label=f'Higher-order coeff_2={1 - lambda_coeff},Higher-order coeff_3=0',
         color='orange')
plt.plot(np.linspace(0, t_final, len(x_sol_3)), x_sol_3[:, 0],
         label=f'Higher-order coeff={lambda_coeff_1},Higher-order coeff_3={lambda_coeff_2}',
         color='green')

for i in range(num_nodes_1):
    plt.plot(np.linspace(0, t_final, len(x_sol_1)), x_sol_1[:, i], color='blue')
    plt.plot(np.linspace(0, t_final, len(x_sol_2)), x_sol_2[:, i], color='orange')
    plt.plot(np.linspace(0, t_final, len(x_sol_3)), x_sol_3[:, i], color='green')

plt.xlabel('Time')
plt.ylabel('x')
plt.title('cont village-three body dynamics')
plt.legend()
plt.show()
print("higher-order=0:\n", x_sol_1[t_end][0])
print(f"higher-order_2={1 - lambda_coeff}:\n", x_sol_2[t_end_2][0])
print(f"higher-order_2={lambda_coeff_1}:\n", x_sol_3[t_end_3][0])
print(f"x_sol_1 的收敛时间是: {t_end}")
print(f"x_sol_2 的收敛时间是: {t_end_2}")
print(f"x_sol_3 的收敛时间是: {t_end_3}")
