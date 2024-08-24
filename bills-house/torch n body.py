import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix, diags
from tqdm import tqdm
import torch
import torch.nn as nn
from torchdiffeq import odeint

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
degree_matrix = diags(np.array(adj_matrix.sum(axis=1)).flatten(), 0)
laplacian_matrix = degree_matrix - adj_matrix

# 将数据转换为PyTorch张量
adj_matrix_torch = torch.tensor(adj_matrix.toarray(), dtype=torch.float32)
degree_matrix_torch = torch.tensor(degree_matrix.toarray(), dtype=torch.float32)
laplacian_matrix_torch = torch.tensor(laplacian_matrix.toarray(), dtype=torch.float32)

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

# 定义PyTorch模型
class DynamicsModel(nn.Module):
    def __init__(self, laplacian_matrix, df_list, node_index_list, lambda_coeffs):
        super(DynamicsModel, self).__init__()
        self.laplacian_matrix = laplacian_matrix
        self.df_list = df_list
        self.node_index_list = node_index_list
        self.lambda_coeffs = lambda_coeffs

    def forward(self, t, x):
        final_result = torch.zeros(num_nodes_1, dtype=torch.float32)
        result_1 = {}
        for i in range(1):
            result_1[i] = torch.zeros(num_nodes_1, dtype=torch.float32)
        for df, node_index, lambda_coeff, i in zip(self.df_list, self.node_index_list, self.lambda_coeffs, range(1)):
            if i == 0:
                final_result += 0.9 * torch.tensor(update_result_3(x.detach().numpy(), df, node_index, result_1[i].detach().numpy()))
        return -0.1 * self.laplacian_matrix.matmul(x) + final_result

# 初始条件
x0 = np.random.rand(num_nodes_1)
x0_torch = torch.tensor(x0, dtype=torch.float32)

# 系数之和为1
lambda_coeffs = np.array([0.1, 0.2, 0.15, 0.15, 0.2, 0.1])

# 初始化模型
model = DynamicsModel(laplacian_matrix_torch, df_list, node_index_list, lambda_coeffs)

# 时间范围
t_max = 20
t_steps = 3000
t = np.linspace(0, t_max, t_steps)

# 求解ODE
with tqdm(total=1, desc="Solving ODE") as pbar:
    x_sol_2_torch = odeint(model, x0_torch, torch.tensor(t), method='dopri5')
    pbar.update(1)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(t, x_sol_2_torch.detach().numpy()[:, 0], color='blue', label='two-body dynamics')
plt.xlabel('Time')
plt.ylabel('x')
plt.title('email Enron Multi-body dynamics')
plt.legend()
plt.show()
