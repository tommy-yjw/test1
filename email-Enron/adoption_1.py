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
    laplacian_matrices_1[node1][node2, node3] += 1
    laplacian_matrices_1[node2][node1, node3] += 1
    laplacian_matrices_1[node3][node1, node2] += 1

# 转换为 CSR 格式
laplacian_matrices_1 = [mat.tocsr() for mat in laplacian_matrices_1]

# 填充稀疏拉普拉斯矩阵
for _, row in tqdm(df2.iterrows(), total=df2.shape[0], desc="Filling Laplacian matrices_2"):
    node1 = node_index_2[row['node_1']]
    node2 = node_index_2[row['node_2']]
    node3 = node_index_2[row['node_3']]

    # 更新稀疏拉普拉斯矩阵
    laplacian_matrices_2[node1][node2, node3] -= 1
    laplacian_matrices_2[node1][node1, node1] += 1  # 每个三元组对node1的影响

    laplacian_matrices_2[node2][node1, node3] -= 1
    laplacian_matrices_2[node2][node2, node2] += 1  # 每个三元组对node2的影响

    laplacian_matrices_2[node3][node1, node2] -= 1
    laplacian_matrices_2[node3][node3, node3] += 1  # 每个三元组对node3的影响

# 转换为 CSR 格式
laplacian_matrices_2 = [mat.tocsr() for mat in laplacian_matrices_2]

# 初始条件（列矩阵x和o）
x0 = np.random.rand(num_nodes_1)  # 确保x0是一维数组
o0 = np.random.rand(num_nodes_1)  # 确保o0是一维数组
abandon_rate = np.random.rand(num_nodes_1)  # 随机丢弃率
theta_coeff = 0.7
lambda_coeff = 0.7
gamma = 1  # 随机生成gamma
beta_ii = np.random.rand(num_nodes_1)  # 随机生成beta_ii
beta_iii = np.random.rand(num_nodes_1)  # 随机生成beta_iii

def dynamics_1(y, t, adj_matrix,  laplacian_matrix,  abandon_rate, theta_coeff,
             lambda_coeff, gamma, beta_ii):
    x = y[:num_nodes_1]
    o = y[num_nodes_1:]
    x_dot = np.zeros_like(x)
    o_dot = np.zeros_like(o)
    for i, B_i in enumerate(laplacian_matrices_1):
        x_dot[i] = -abandon_rate[i] * x[i] * (1 - o[i]) + (1 - x[i]) * o[i] * (adj_matrix[i].dot(x) + beta_ii[i])
    for i, B_i in enumerate(laplacian_matrices_2):
        o_dot[i] = - laplacian_matrix[i].dot(o) + gamma * x[i] - o[i]

    return np.concatenate((x_dot, o_dot))
def dynamics_2(y, t, adj_matrix, laplacian_matrices_1, laplacian_matrix, laplacian_matrices_2, abandon_rate, theta_coeff,
             lambda_coeff, gamma, beta_ii, beta_iii):
    x = y[:num_nodes_1]
    o = y[num_nodes_1:]
    x_dot = np.zeros_like(x)
    o_dot = np.zeros_like(o)
    for i, B_i in enumerate(laplacian_matrices_1):
        x_dot[i] = -abandon_rate[i] * x[i] * (1 - o[i]) + theta_coeff * (1 - x[i]) * o[i] * (adj_matrix[i].dot(x) + beta_ii[i]) + (1 - theta_coeff) * (1 - x[i]) * o[i] * (x.T.dot(B_i.dot(x)) + beta_iii[i])
    for i, B_i in enumerate(laplacian_matrices_2):
        o_dot[i] = -lambda_coeff * laplacian_matrix[i].dot(o) - (1 - lambda_coeff) * o.T.dot(B_i.dot(o)) + gamma * x[i] - o[i]

    return np.concatenate((x_dot, o_dot))


# 时间范围
t_max = 10  # 最大时间范围
t_steps = 3000  # 时间步数
t = np.linspace(0, t_max, t_steps)

# 初始条件
y0 = np.concatenate((x0, o0))



# 求解ODE
with tqdm(total=2, desc="Solving ODEs") as pbar:
    y_sol_1 = odeint(dynamics_1, y0, t, args=(adj_matrix, laplacian_matrix, abandon_rate, theta_coeff, lambda_coeff, gamma, beta_ii))
    pbar.update(1)
    y_sol_2 = odeint(dynamics_2, y0, t, args=(adj_matrix, laplacian_matrices_1, laplacian_matrix, laplacian_matrices_2, abandon_rate, theta_coeff, lambda_coeff, gamma, beta_ii, beta_iii))
    pbar.update(1)

x_sol_1 = y_sol_1[:, :num_nodes_1]
o_sol_1 = y_sol_1[:, num_nodes_1:]
x_sol_2 = y_sol_2[:, :num_nodes_1]
o_sol_2 = y_sol_2[:, num_nodes_1:]

tolerance = 0.001
converged = False
for i in tqdm(range(1, len(t)), desc="Checking convergence for x_sol_1"):
    for j in range(num_nodes_1):
        if np.abs(x_sol_1[i][j] - x_sol_1[i-1][j]) >= tolerance:
            break
    else:  # 这个 else 对应于 for 循环，表示 for 循环没有被 break 中断
        t_end_x1 = i
        converged = True
        break

if not converged:
    t_end_x1 = t_max

for i in tqdm(range(1, len(t)), desc="Checking convergence for x_sol_2"):
    for j in range(num_nodes_1):
        if np.abs(o_sol_1[i][j] - o_sol_1[i-1][j]) >= tolerance:
            break
    else:  # 这个 else 对应于 for 循环，表示 for 循环没有被 break 中断
        t_end_o1 = i
        converged = True
        break

if not converged:
    t_end_o1 = t_max

# 计算收敛时间
for i in tqdm(range(1, len(t)), desc="Checking convergence for x_sol_1"):
    for j in range(num_nodes_1):
        if np.abs(x_sol_2[i][j] - x_sol_2[i-1][j]) >= tolerance:
            break
    else:  # 这个 else 对应于 for 循环，表示 for 循环没有被 break 中断
        t_end_x2 = i
        converged = True
        break

if not converged:
    t_end_x2 = t_max

for i in tqdm(range(1, len(t)), desc="Checking convergence for x_sol_2"):
    for j in range(num_nodes_1):
        if np.abs(o_sol_2[i][j] - o_sol_2[i-1][j]) >= tolerance:
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
plt.plot(np.linspace(0, t_final_x, len(x_sol_1)), x_sol_1[:, 0], label='higher-order coeff=0',color='blue')
plt.plot(np.linspace(0, t_final_x, len(x_sol_2)), x_sol_2[:, 0], label='higher-order coeff=0.6',color='orange')
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
plt.plot(np.linspace(0, t_final_o, len(o_sol_1)), o_sol_1[:, 0], label='higher-order coeff=0',color='blue')
plt.plot(np.linspace(0, t_final_o, len(o_sol_2)), o_sol_2[:, 0], label='higher-order coeff=0.6',color='orange')
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
