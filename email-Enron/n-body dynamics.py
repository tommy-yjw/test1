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



# 计算稀疏度矩阵
adj_matrix = adj_matrix.tocsr()

# 计算稀疏度矩阵
degree_matrix = diags(np.array(adj_matrix.sum(axis=1)).flatten(), 0)
laplacian_matrix = degree_matrix - adj_matrix


# 初始条件


# 读取CSV文件
df_list = [pd.read_csv(f'size-{i}-unique-sorted.csv') for i in range(3, 9)]

# 获取节点列表
nodes_list = [set(df.iloc[:, :df.shape[1]].values.flatten()) for df in df_list]
node_index_list = [{node: idx for idx, node in enumerate(nodes)} for nodes in nodes_list]




# 定义更新结果的函数
def update_result_3(x0, df, node_index, result):
    result[:] = 0
    for _, row in df.iterrows():
        nodes = [node_index[row[f'node_{i}']] for i in range(1, 4)]
        product = np.prod([x0[node] for node in nodes[1:]])
        for i in range(len(nodes)):
            result[nodes[i]] += product - x0[nodes[i]] ** 2
    return result


def update_result_4(x0, df, node_index, result):
    result[:] = 0
    for _, row in df.iterrows():
        nodes = [node_index[row[f'node_{i}']] for i in range(1, 5)]
        product = np.prod([x0[node] for node in nodes[1:]])
        for i in range(len(nodes)):
            result[nodes[i]] += product - x0[nodes[i]] ** 3
    return result


def update_result_5(x0, df, node_index, result):
    result[:] = 0
    for _, row in df.iterrows():
        nodes = [node_index[row[f'node_{i}']] for i in range(1, 6)]
        product = np.prod([x0[node] for node in nodes[1:]])
        for i in range(len(nodes)):
            result[nodes[i]] += product - x0[nodes[i]] ** 4
    return result


def update_result_6(x0, df, node_index, result):
    result[:] = 0
    for _, row in df.iterrows():
        nodes = [node_index[row[f'node_{i}']] for i in range(1, 7)]
        product = np.prod([x0[node] for node in nodes[1:]])
        for i in range(len(nodes)):
            result[nodes[i]] += product - x0[nodes[i]] ** 5
    return result


def update_result_7(x0, df, node_index, result):
    result[:] = 0
    for _, row in df.iterrows():
        nodes = [node_index[row[f'node_{i}']] for i in range(1, 8)]
        product = np.prod([x0[node] for node in nodes[1:]])
        for i in range(len(nodes)):
            result[nodes[i]] += product - x0[nodes[i]] ** 6
    return result


def update_result_8(x0, df, node_index, result):
    result[:] = 0
    for _, row in df.iterrows():
        nodes = [node_index[row[f'node_{i}']] for i in range(1, 9)]
        product = np.prod([x0[node] for node in nodes[1:]])
        for i in range(len(nodes)):
            result[nodes[i]] += product - x0[nodes[i]] ** 7
    return result

def dynamics2(x, t, laplacian_matrix):
    return -laplacian_matrix.dot(x)

def dynamics3(x, t, laplacian_matrix, df_list, node_index_list, lambda_coeffs):
    final_result = np.zeros(num_nodes_1, dtype='float32')
    result_1 = {}
    for i in range(1):
        result_1[i] = np.zeros(num_nodes_1, dtype='float32')
    for df, node_index, lambda_coeff, i in zip(df_list, node_index_list, lambda_coeffs, range(1)):
        if i == 0:
            final_result += 0.9 * update_result_3(x, df, node_index, result_1[i])
    return -0.1 * laplacian_matrix.dot(x) + final_result

def dynamics4(x, t, laplacian_matrix, df_list, node_index_list, lambda_coeffs):
    final_result = np.zeros(num_nodes_1, dtype='float32')
    result_2 = {}
    for i in range(2):
        result_2[i] = np.zeros(num_nodes_1, dtype='float32')
    for df, node_index, lambda_coeff, i in zip(df_list, node_index_list, lambda_coeffs, range(2)):
        if i == 0:
            final_result += lambda_coeffs[i] * update_result_3(x, df, node_index, result_2[i])
        elif i == 1:
            final_result += 0.8 * update_result_4(x, df, node_index, result_2[i])
    return -0.1 * laplacian_matrix.dot(x) + final_result

def dynamics5(x, t, laplacian_matrix, df_list, node_index_list, lambda_coeffs):
    final_result = np.zeros(num_nodes_1, dtype='float32')
    result_3 = {}
    for i in range(3):
        result_3[i] = np.zeros(num_nodes_1, dtype='float32')
    for df, node_index, lambda_coeff, i in zip(df_list, node_index_list, lambda_coeffs, range(3)):
        if i == 0:
            final_result += lambda_coeffs[i] * update_result_3(x, df, node_index, result_3[i])
        elif i == 1:
            final_result += lambda_coeffs[i] * update_result_4(x, df, node_index, result_3[i])
        elif i == 2:
            final_result += 0.6 * update_result_5(x, df, node_index, result_3[i])
    return -0.1 * laplacian_matrix.dot(x) + final_result
def dynamics6(x, t, laplacian_matrix, df_list, node_index_list, lambda_coeffs):
    final_result = np.zeros(num_nodes_1, dtype='float32')
    result_4 = {}
    for i in range(4):
        result_4[i] = np.zeros(num_nodes_1, dtype='float32')
    for df, node_index, lambda_coeff, i in zip(df_list, node_index_list, lambda_coeffs, range(4)):
        if i == 0:
            final_result += lambda_coeffs[i] * update_result_3(x, df, node_index, result_4[i])
        elif i == 1:
            final_result += lambda_coeffs[i] * update_result_4(x, df, node_index, result_4[i])
        elif i == 2:
            final_result += lambda_coeffs[i] * update_result_5(x, df, node_index, result_4[i])
        elif i == 3:
            final_result += 0.45 * update_result_6(x, df, node_index, result_4[i])
    return -0.1 * laplacian_matrix.dot(x) + final_result
def dynamics7(x, t, laplacian_matrix, df_list, node_index_list, lambda_coeffs):
    final_result = np.zeros(num_nodes_1, dtype='float32')
    result_5 = {}
    for i in range(5):
        result_5[i] = np.zeros(num_nodes_1, dtype='float32')
    for df, node_index, lambda_coeff, i in zip(df_list, node_index_list, lambda_coeffs, range(5)):
        if i == 0:
            final_result += lambda_coeffs[i] * update_result_3(x, df, node_index, result_5[i])
        elif i == 1:
            final_result += lambda_coeffs[i] * update_result_4(x, df, node_index, result_5[i])
        elif i == 2:
            final_result += lambda_coeffs[i] * update_result_5(x, df, node_index, result_5[i])
        elif i == 3:
            final_result += lambda_coeffs[i] * update_result_6(x, df, node_index, result_5[i])
        elif i == 4:
            final_result += 0.3 * update_result_7(x, df, node_index, result_5[i])
    return -0.1 * laplacian_matrix.dot(x) + final_result


# 定义动力学方程
def dynamics8(x, t, laplacian_matrix, df_list, node_index_list, lambda_coeffs):
    final_result = np.zeros(num_nodes_1, dtype='float32')
    result_6 = {}
    for i in range(6):
        result_6[i] = np.zeros(num_nodes_1, dtype='float32')
    for df, node_index, lambda_coeff, i in zip(df_list, node_index_list, lambda_coeffs, range(6)):
        if i == 0:
            final_result += lambda_coeffs[i] * update_result_3(x, df, node_index, result_6[i])
        elif i == 1:
            final_result += lambda_coeffs[i] * update_result_4(x, df, node_index, result_6[i])
        elif i == 2:
            final_result += lambda_coeffs[i] * update_result_5(x, df, node_index, result_6[i])
        elif i == 3:
            final_result += lambda_coeffs[i] * update_result_6(x, df, node_index, result_6[i])
        elif i == 4:
            final_result += lambda_coeffs[i] * update_result_7(x, df, node_index, result_6[i])
        elif i == 5:
            final_result += lambda_coeffs[i] * update_result_8(x, df, node_index, result_6[i])
    return -0.1 * laplacian_matrix.dot(x) + final_result


# 时间范围
t_max = 20
t_steps = 3000
t = np.linspace(0, t_max, t_steps)
x0 = np.random.rand(num_nodes_1)
# 系数之和为1
lambda_coeffs = np.array([0.1, 0.2, 0.15, 0.15, 0.2, 0.1])

# 求解ODE
with tqdm(total=7, desc="Solving ODE") as pbar:
    x_sol_2 = odeint(dynamics2, x0, t, args=(laplacian_matrix,))
    pbar.update(1)
    x_sol_3 = odeint(dynamics3, x0, t, args=(laplacian_matrix, df_list, node_index_list, lambda_coeffs))
    pbar.update(1)
    x_sol_4 = odeint(dynamics4, x0, t, args=(laplacian_matrix, df_list, node_index_list, lambda_coeffs))
    pbar.update(1)
    x_sol_5 = odeint(dynamics5, x0, t, args=(laplacian_matrix, df_list, node_index_list, lambda_coeffs))
    pbar.update(1)
    x_sol_6 = odeint(dynamics6, x0, t, args=(laplacian_matrix, df_list, node_index_list, lambda_coeffs))
    pbar.update(1)
    x_sol_7 = odeint(dynamics7, x0, t, args=(laplacian_matrix, df_list, node_index_list, lambda_coeffs))
    pbar.update(1)
    x_sol_8 = odeint(dynamics8, x0, t, args=(laplacian_matrix, df_list, node_index_list, lambda_coeffs))
    pbar.update(1)

tolerance = 0.001
converged = False
t_ends = []

# 检查每个dynamics的收敛时间
for idx, x_sol in enumerate([x_sol_2, x_sol_3, x_sol_4, x_sol_5, x_sol_6, x_sol_7, x_sol_8]):
    for i in tqdm(range(1, len(t)), desc=f"Checking convergence for x_sol_{idx+2}"):
        for j in range(num_nodes_1):
            if np.abs(x_sol[i][j] - x_sol[i - 1][j]) >= tolerance:
                break
        else:  # 这个 else 对应于 for 循环，表示 for 循环没有被 break 中断
            t_ends.append(i)
            converged = True
            break
    if not converged:
        t_ends.append(t_max)
    converged = False

# 计算最大收敛时间
t_final = max(t_ends)
t = np.linspace(0, t_final, t_steps)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(t, x_sol_2[:, 0], color='blue', label='two-body dynamics')
plt.plot(t, x_sol_3[:, 0], color='orange', label='three-body dynamics')
plt.plot(t, x_sol_4[:, 0], color='green', label='four-body dynamics')
plt.plot(t, x_sol_5[:, 0], color='red', label='five-body dynamics')
plt.plot(t, x_sol_6[:, 0], color='brown', label='six-body dynamics')
plt.plot(t, x_sol_7[:, 0], color='cyan', label='seven-body dynamics')
plt.plot(t, x_sol_8[:, 0], color='olive', label='eight-body dynamics')
for j in range(num_nodes_1):
    plt.plot(t, x_sol_2[:, j], color='blue')
    plt.plot(t, x_sol_3[:, j], color='orange')
    plt.plot(t, x_sol_4[:, j], color='green')
    plt.plot(t, x_sol_5[:, j], color='red')
    plt.plot(t, x_sol_6[:, j], color='brown')
    plt.plot(t, x_sol_7[:, j], color='cyan')
    plt.plot(t, x_sol_8[:, j], color='olive')
plt.xlabel('Time')
plt.ylabel('x')
plt.title('email Enron Multi-body dynamics')
plt.legend()
plt.show()
print(f"two-body dynamics:\n", x_sol_2[t_final][0])
print(f"Convergence time: {t_ends[0]}")
print(f"three-body dynamics:\n", x_sol_3[t_final][0])
print(f"Convergence time: {t_ends[1]}")
print(f"four-body dynamics:\n", x_sol_4[t_final][0])
print(f"Convergence time: {t_ends[2]}")
print(f"five-body dynamics:\n", x_sol_5[t_final][0])
print(f"Convergence time: {t_ends[3]}")
print(f"six-body dynamics:\n", x_sol_6[t_final][0])
print(f"Convergence time: {t_ends[4]}")
print(f"seven-body dynamics:\n", x_sol_7[t_final][0])
print(f"Convergence time: {t_ends[5]}")
print(f"eight-body dynamics:\n", x_sol_8[t_final][0])
print(f"Convergence time: {t_ends[6]}")
