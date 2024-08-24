import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint

# 读取第一个CSV文件
df1 = pd.read_csv('size-2-unique-sorted.csv')

# 获取节点列表
nodes_1 = set(df1['node_1']).union(set(df1['node_2']))
node_index_1 = {node: idx for idx, node in enumerate(nodes_1)}

# 初始化邻接矩阵
num_nodes_1 = len(nodes_1)
adj_matrix_1 = np.zeros((num_nodes_1, num_nodes_1))

# 填充第一个邻接矩阵（node_2指向node_1）
for _, row in df1.iterrows():
    node1 = node_index_1[row['node_1']]
    node2 = node_index_1[row['node_2']]
    adj_matrix_1[node2, node1] = 1  # node_2指向node_1
# 计算度矩阵
degree_matrix = np.diag(np.sum(adj_matrix_1, axis=1))

# 计算拉普拉斯矩阵
laplacian_matrix = degree_matrix - adj_matrix_1
# 读取CSV文件
df2 = pd.read_csv('size-3-unique-sorted.csv')

# 获取节点列表
nodes_2 = set(df2['node_1']).union(set(df2['node_2'])).union(set(df2['node_3']))
node_index_2 = {node: idx for idx, node in enumerate(nodes_2)}

# 初始化拉普拉斯矩阵列表
num_nodes_2 = len(nodes_2)
laplacian_matrices = [np.zeros((num_nodes_1, num_nodes_1)) for _ in range(num_nodes_1)]

# 填充第二个邻接矩阵（node_2和node_3指向node_1）
for _, row in df2.iterrows():
    node1 = node_index_2[row['node_1']]
    node2 = node_index_2[row['node_2']]
    node3 = node_index_2[row['node_3']]
    laplacian_matrices[node1][node2, node3] -= 1

    laplacian_matrices[node1][node1, node1] += 1  # 每个三元组对node1的影响

x0 = np.random.rand(num_nodes_1)  # 确保x0是一维数组
lambda_coeff = np.random.rand()

def dynamics1(x, t, laplacian_matrix):
    return -np.dot(laplacian_matrix, x)
# 定义动力学方程
def dynamics2(x, t, laplacian_matrix, laplacian_matrices, lambda_coeff):
    x_dot = np.zeros_like(x)
    for i, B_i in enumerate(laplacian_matrices):
            x_dot[i] = -lambda_coeff * np.dot(laplacian_matrix[i], x) - (1 - lambda_coeff) * np.dot(x.T, np.dot(B_i, x))
    return x_dot


# 时间范围
t_max = 10  # 最大时间范围
t_steps = 5000  # 时间步数
t = np.linspace(0, t_max, t_steps)


# 求解ODE
x_sol_1 = odeint(dynamics1, x0, t, args=(laplacian_matrix,))

x_sol_2 = odeint(dynamics2, x0, t, args=(laplacian_matrix, laplacian_matrices, lambda_coeff))
# 绘制结果
tolerance = 0.001
converged = False
for i in range(1, len(t)):
    for j in range(num_nodes_1):
        if np.abs(x_sol_1[i][j] - x_sol_1[i-1][j]) >= tolerance:
            break
    else:  # 这个 else 对应于 for 循环，表示 for 循环没有被 break 中断
        t_end = i
        converged = True
        break

if not converged:
    t_end = t_max


for i in range(1, len(t)):
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
if t_end < t_end_2:
    t_final = t_end_2
else:
    t_final = t_end
# 输出结果

plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0, t_final, len(x_sol_1)), x_sol_1[:, 0], label=f'Higher-order coeff=0', color='blue')
plt.plot(np.linspace(0, t_final, len(x_sol_2)), x_sol_2[:, 0], label=f'Higher-order coeff={lambda_coeff}', color='orange')

for i in range(num_nodes_1):
    plt.plot(np.linspace(0, t_final, len(x_sol_1)), x_sol_1[:, i], color='blue')
    plt.plot(np.linspace(0, t_final, len(x_sol_2)), x_sol_2[:, i], color='orange')

plt.xlabel('Time')
plt.ylabel('x')
plt.title('cont-village three body dynamics directed graph')
plt.legend()
plt.show()
print("higher-order=0:\n", x_sol_1[t_end][0])
print(f"higher-order={lambda_coeff}:\n", x_sol_2[t_end_2][0])
print(f"x_sol_1 的收敛时间是: {t_end}")
print(f"x_sol_2 的收敛时间是: {t_end_2}")

