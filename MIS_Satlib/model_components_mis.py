# model_components_mis.py (Corrected)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import MessagePassing
import tqdm
# model_components_mis.py (最终完美复现版)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.utils.rnn as rnn_utils
from torch_geometric.nn import MessagePassing

# SinusoidalTimestepEmbedding 和 PrefixEncoder 保持不变，它们是正确的
class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t.float().unsqueeze(1) * embeddings.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
        return embeddings

class PrefixEncoder(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=node_feat_dim, hidden_size=hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, prefix_node_features, prefix_lengths):
        packed_features = rnn_utils.pack_padded_sequence(
            prefix_node_features,
            prefix_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        self.lstm.flatten_parameters()
        _, (hn, _) = self.lstm(packed_features)
        return self.linear(hn.squeeze(0))

class DIFUSCOGNNLayer(MessagePassing):
    """
    对DIFUSCO GNNLayer的忠实复现，适配了torch_geometric。
    实现了 U*h_i + Aggr(sigmoid(A*h_i + B*h_j) * V*h_j) 的核心逻辑。
    """
    def __init__(self, hidden_dim, aggregation="add"):
        super().__init__(aggr=aggregation)
        
        # 对应DIFUSCO中的U, V, A, B 线性层
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.A = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.B = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        # DIFUSCO在每层都使用了规范化
        self.norm_h = nn.LayerNorm(hidden_dim)
        self.norm_e = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_index):
        # h: [num_nodes, hidden_dim]
        h_in = h
        
        # 对应DIFUSCO中的 Uh
        Uh = self.U(h)
        
        # self.propagate 会调用 message, aggregate, 和 update 函数
        # 它负责执行消息传递的核心步骤
        aggregated_messages = self.propagate(edge_index, h=h)
        
        # 节点特征更新
        h_new = Uh + aggregated_messages
        h_new = self.norm_h(h_new)
        h_new = F.relu(h_new)
        
        # 最终的残差连接
        return h_in + h_new

    def message(self, h_i, h_j):
        """
        构建从节点j到节点i的消息。
        h_i: 目标节点特征 [num_edges, hidden_dim]
        h_j: 源节点特征 [num_edges, hidden_dim]
        """
        
        # 对应DIFUSCO中的 e = A*h_i + B*h_j
        # 这里模拟了边特征的计算过程
        e = self.A(h_i) + self.B(h_j)
        e = self.norm_e(e)
        e = F.relu(e)
        
        # 对应DIFUSCO中的门控机制: gates = sigmoid(e)
        gates = torch.sigmoid(e)
        
        # 对应DIFUSCO中的 V*h_j
        Vh_j = self.V(h_j)
        
        # 返回门控后的消息：gates * Vh_j
        return gates * Vh_j
# # Simplified GNN Layer for clarity, inspired by DIFUSCO but adapted for torch_geometric
# class MISGNNLayer(MessagePassing):
#     def __init__(self, hidden_dim):
#         super().__init__(aggr='add') # 'add' == 'sum'
#         self.msg_mlp = nn.Sequential(
#             nn.Linear(2 * hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim)
#         )
#         self.update_mlp = nn.Sequential(
#             nn.Linear(2 * hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim)
#         )
#         self.norm = nn.LayerNorm(hidden_dim)

#     def forward(self, x, edge_index):
#         # x has shape [N, hidden_dim]
#         # edge_index has shape [2, E]
#         out = self.propagate(edge_index, x=x)
#         out = self.update_mlp(torch.cat([x, out], dim=-1))
#         return self.norm(x + out)

#     def message(self, x_i, x_j):
#         # x_i has shape [E, hidden_dim]; x_j has shape [E, hidden_dim]
#         tmp = torch.cat([x_i, x_j], dim=-1)
#         return self.msg_mlp(tmp)

# # Other components like PrefixEncoder and SinusoidalTimestepEmbedding remain the same


# class SinusoidalTimestepEmbedding(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, t):
#         device = t.device
#         half_dim = self.dim // 2
#         embeddings = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
#         embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
#         embeddings = t.float().unsqueeze(1) * embeddings.unsqueeze(0)
#         embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
#         if self.dim % 2 == 1:
#             embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
#         return embeddings

# class PrefixEncoder(nn.Module):
#     def __init__(self, node_feat_dim, hidden_dim, output_dim):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size=node_feat_dim, hidden_size=hidden_dim, batch_first=True)
#         self.linear = nn.Linear(hidden_dim, output_dim)

#     def forward(self, prefix_node_features, prefix_lengths):
#         packed_features = torch.nn.utils.rnn.pack_padded_sequence(
#             prefix_node_features,
#             prefix_lengths.cpu(),
#             batch_first=True,
#             enforce_sorted=False
#         )
#         self.lstm.flatten_parameters()
#         _, (hn, _) = self.lstm(packed_features)
#         return self.linear(hn.squeeze(0))
        