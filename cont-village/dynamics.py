from fractions import Fraction

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint

# 读取CSV文件
df = pd.read_csv('size-2-unique-sorted.csv')

# 获取节点列表
nodes = set(df['node_1']).union(set(df['node_2']))
node_index = {node: idx for idx, node in enumerate(nodes)}

# 初始化邻接矩阵
num_nodes = len(nodes)
adj_matrix = np.zeros((num_nodes, num_nodes))


# 填充邻接矩阵
for _, row in df.iterrows():
    node1 = node_index[row['node_1']]
    node2 = node_index[row['node_2']]
    adj_matrix[node1, node2] = 1
    adj_matrix[node2, node1] = 1

# 计算度矩阵
degree_matrix = np.diag(np.sum(adj_matrix, axis=1))

# 计算拉普拉斯矩阵
laplacian_matrix = degree_matrix - adj_matrix


# 初始条件（列矩阵x）
x0 = np.random.rand(num_nodes, 1)
u0 = np.random.rand(num_nodes, 1)
lambda_1 = 0.5
# 定义动力学方程
def dynamics(x, t):
    return -np.dot(laplacian_matrix, x)
# def dynamics(x, t, laplacian_matrix, lambda_1, u0):
#     return -lambda_1 * np.dot(laplacian_matrix, x) + (1 - lambda_1) * (u0 - x)

# 时间范围
t_max = 30  # 最大时间范围
t_steps = 1000  # 时间步数
t = np.linspace(0, t_max, t_steps)

# 求解ODE
x_sol = odeint(dynamics, x0.flatten(), t)

# x_sol = odeint(dynamics, x0.flatten(), t, args=(laplacian_matrix, lambda_1, u0.flatten()))


# 检查每个时间步长下x的变化量
tolerance = 0.001
for i in range(1, len(t)):
    for j in range(num_nodes):
        if np.abs(x_sol[i][j] - x_sol[i-1][j]) < tolerance:
            t_end = i
            break
        else:
            t_end = t_max



# 绘制结果
plt.figure(figsize=(10, 6))
for i in range(num_nodes):
    plt.plot(np.linspace(0, t_end, len(x_sol)), x_sol[:, i], label=f'Node {i+1}')
plt.xlabel('Time')
plt.ylabel('x')
plt.title('Dynamics of the System')
plt.legend()
plt.show()
print("邻接矩阵:\n", adj_matrix)
print("度矩阵:\n", degree_matrix)
print("拉普拉斯矩阵:\n", laplacian_matrix)
print("节点列表:\n", num_nodes)
print("初始条件:\n", x_sol[t_end][0])
