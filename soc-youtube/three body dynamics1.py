import pandas as pd
import torch
from scipy.sparse import lil_matrix
from tqdm import tqdm
from torchdiffeq import odeint
import matplotlib.pyplot as plt

# 读取CSV文件
df1 = pd.read_csv('size-2-unique-sorted.csv')

# 获取节点列表并建立索引
nodes_1 = set(df1['node_1']).union(set(df1['node_2']))
node_index_1 = {node: idx for idx, node in enumerate(nodes_1)}
num_nodes_1 = len(nodes_1)

# 初始化稀疏邻接矩阵
adj_matrix = lil_matrix((num_nodes_1, num_nodes_1), dtype='float32')

# 填充稀疏邻接矩阵
for _, row in tqdm(df1.iterrows(), total=df1.shape[0], desc="Filling adjacency matrix"):
    node1 = node_index_1[row['node_1']]
    node2 = node_index_1[row['node_2']]
    adj_matrix[node1, node2] = 1
    adj_matrix[node2, node1] = 1

# 转换为 CSR 稀疏格式，并转换为 PyTorch 稀疏张量
adj_matrix = torch.sparse_coo_tensor(
    torch.tensor(adj_matrix.nonzero(), dtype=torch.long),
    torch.tensor(adj_matrix[adj_matrix.nonzero()].toarray().flatten(), dtype=torch.float32),
    size=adj_matrix.shape
)

# 计算拉普拉斯矩阵
degree_values = torch.sparse.sum(adj_matrix, dim=1).to_dense()
degree_indices = torch.arange(num_nodes_1)
degree_matrix = torch.sparse_coo_tensor(
    torch.stack([degree_indices, degree_indices]),
    degree_values,
    (num_nodes_1, num_nodes_1)
)
laplacian_matrix = degree_matrix - adj_matrix

# 清理无用变量，释放内存
del df1, adj_matrix

# 读取CSV文件
df2 = pd.read_csv('size-3-unique-sorted.csv')

# 获取节点列表并建立索引
nodes_2 = set(df2['node_1']).union(set(df2['node_2'])).union(set(df2['node_3']))
node_index_2 = {node: idx for idx, node in enumerate(nodes_2)}
num_nodes_2 = len(nodes_2)

# 初始化稀疏拉普拉斯矩阵列表
laplacian_matrices = []

for i in range(num_nodes_1):
    laplacian_matrices.append(torch.sparse_coo_tensor(torch.zeros((2, 0), dtype=torch.long), torch.tensor([]), (num_nodes_1, num_nodes_1)))

# 填充稀疏拉普拉斯矩阵
for _, row in tqdm(df2.iterrows(), total=df2.shape[0], desc="Filling Laplacian matrices"):
    node1 = node_index_2[row['node_1']]
    node2 = node_index_2[row['node_2']]
    node3 = node_index_2[row['node_3']]

    indices = torch.tensor([[node1, node2, node3, node1, node2, node3],
                            [node2, node3, node1, node3, node1, node2]], dtype=torch.long)
    values = torch.tensor([-1, -1, -1, 1, 1, 1], dtype=torch.float32)
    laplacian_matrices[node1] = laplacian_matrices[node1] + torch.sparse_coo_tensor(indices, values, (num_nodes_1, num_nodes_1))

# 清理无用变量，释放内存
del df2, node_index_2

# 初始条件
x0 = torch.rand(num_nodes_1, dtype=torch.float32)
lambda_coeff = 0.4

# 定义动力学方程
def dynamics1(t, x, laplacian_matrix):
    return -torch.sparse.mm(laplacian_matrix, x.unsqueeze(-1)).squeeze()

def dynamics2(t, x, laplacian_matrix, laplacian_matrices, lambda_coeff):
    x_dot = torch.zeros_like(x)
    for i, B_i in enumerate(laplacian_matrices):
        laplacian_term = torch.sparse.mm(laplacian_matrix, x.unsqueeze(-1)).squeeze()
        B_i_term = torch.sparse.mm(B_i, x.unsqueeze(-1)).squeeze()
        x_dot[i] = -lambda_coeff * laplacian_term[i] - (1 - lambda_coeff) * (x.T @ B_i_term)
    return x_dot

# 使用 PyTorch 的求解器求解 ODE
t_max = 10  # 最大时间范围
t_steps = 2000  # 时间步数
t = torch.linspace(0, t_max, t_steps)

# 求解 ODE
with tqdm(total=2, desc="Solving ODEs") as pbar:
    x_sol_1 = odeint(lambda t, x: dynamics1(t, x, laplacian_matrix), x0, t, method='dopri5', atol=1e-5, rtol=1e-5)
    pbar.update(1)
    x_sol_2 = odeint(lambda t, x: dynamics2(t, x, laplacian_matrix, laplacian_matrices, lambda_coeff), x0, t, method='dopri5', atol=1e-5, rtol=1e-5)
    pbar.update(1)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(t.cpu().numpy(), x_sol_1[:, 0].cpu().numpy(), label=f'Higher-order coeff=0', color='blue')
plt.plot(t.cpu().numpy(), x_sol_2[:, 0].cpu().numpy(), label=f'Higher-order coeff={1-lambda_coeff}', color='orange')

for i in range(num_nodes_1):
    plt.plot(t.cpu().numpy(), x_sol_1[:, i].cpu().numpy(), color='blue')
    plt.plot(t.cpu().numpy(), x_sol_2[:, i].cpu().numpy(), color='orange')

plt.xlabel('Time')
plt.ylabel('x')
plt.title('Soc YouTube Three-Body Dynamics')
plt.legend()
plt.show()
