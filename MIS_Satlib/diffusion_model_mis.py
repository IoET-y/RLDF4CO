# # diffusion_model_mis.py (Corrected and Final Version)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.nn.utils.rnn as rnn_utils
# import tqdm

# # We assume model_components_mis.py is in the same directory and contains these classes
# from model_components_mis import PrefixEncoder, MISGNNLayer, SinusoidalTimestepEmbedding

# class ConditionalMISSuffixDiffusionModel(nn.Module):
#     """
#     A diffusion model for solving the MIS problem with prefix conditions.
#     This version correctly handles dynamic graph sizes using torch_geometric
#     and has the corrected layer dimensions.
#     """
#     def __init__(self, node_feature_dim, node_embed_dim, gnn_n_layers, gnn_hidden_dim,
#                  prefix_enc_hidden_dim, prefix_cond_dim, time_embed_dim):
#         super().__init__()
        
#         # --- FIX: Hardcode the input dimension to 2 ---
#         # We are explicitly concatenating two 1-dimensional features:
#         # 1. The noisy label feature
#         # 2. The prefix state feature (0 for suffix, 1 for prefix)
#         # Therefore, the input dimension is always 2. This removes ambiguity.
#         self.initial_node_feature_proj = nn.Linear(2, node_embed_dim)

#         self.time_embed_mlp = nn.Sequential(
#             SinusoidalTimestepEmbedding(time_embed_dim),
#             nn.Linear(time_embed_dim, time_embed_dim),
#             nn.ReLU(),
#             nn.Linear(time_embed_dim, node_embed_dim) # Project to node_embed_dim
#         )
        
#         self.prefix_encoder = PrefixEncoder(
#             node_feat_dim=node_embed_dim,
#             hidden_dim=prefix_enc_hidden_dim,
#             output_dim=prefix_cond_dim
#         )
#         # This layer projects the prefix condition to the node_embed_dim
#         self.prefix_cond_proj = nn.Linear(prefix_cond_dim, node_embed_dim)
        
#         # GNN Layers
#         self.gnn_input_proj = nn.Linear(node_embed_dim, gnn_hidden_dim)
#         self.gnn_layers = nn.ModuleList([
#             MISGNNLayer(gnn_hidden_dim) for _ in range(gnn_n_layers)
#         ])
        
#         self.output_head = nn.Sequential(
#             nn.LayerNorm(gnn_hidden_dim),
#             nn.ReLU(),
#             nn.Linear(gnn_hidden_dim, 2) # 2 output classes (in or out of MIS)
#         )

#     def forward(self, graph_batch, t):
#         x, edge_index, batch_vector = graph_batch.x, graph_batch.edge_index, graph_batch.batch
#         node_prefix_state = graph_batch.node_prefix_state
#         device = x.device

#         # 1. Initial Node Embedding
#         initial_node_features_raw = torch.cat([x, node_prefix_state], dim=-1)
#         node_emb = self.initial_node_feature_proj(initial_node_features_raw)

#         # 2. Time and Prefix Conditioning
#         time_emb = self.time_embed_mlp(t)
#         time_emb_expanded = time_emb[batch_vector]

#         prefix_cond_vector = torch.zeros(graph_batch.num_graphs, self.prefix_encoder.linear.out_features, 
#                                          device=device, dtype=x.dtype)
#         prefix_lengths = graph_batch.prefix_len
#         has_prefix_mask = prefix_lengths > 0

#         if has_prefix_mask.any():
#             graphs_with_prefix = graph_batch.to_data_list()
#             features_to_encode = []
#             lengths_to_encode = []
            
#             node_counter = 0
#             for i in range(graph_batch.num_graphs):
#                 num_nodes_in_graph = graphs_with_prefix[i].num_nodes
#                 if prefix_lengths[i] > 0:
#                     prefix_indices_local = graphs_with_prefix[i].prefix_nodes
#                     prefix_features = node_emb[node_counter:node_counter + num_nodes_in_graph][prefix_indices_local]
#                     features_to_encode.append(prefix_features)
#                     lengths_to_encode.append(prefix_lengths[i])
#                 node_counter += num_nodes_in_graph

#             padded_features = rnn_utils.pad_sequence(features_to_encode, batch_first=True)
#             computed_cond = self.prefix_encoder(padded_features, torch.tensor(lengths_to_encode))
#             prefix_cond_vector[has_prefix_mask] = computed_cond

#         prefix_cond_emb = self.prefix_cond_proj(prefix_cond_vector)
#         prefix_cond_expanded = prefix_cond_emb[batch_vector]

#         # 3. Combine Embeddings and Project for GNN
#         # Add conditioning vectors to the node embeddings, similar to DIFUSCO
#         combined_emb = node_emb + time_emb_expanded + prefix_cond_expanded
#         h = self.gnn_input_proj(combined_emb)
        
#         # 4. GNN Propagation
#         for layer in self.gnn_layers:
#             h = layer(h, edge_index)
            
#         # 5. Output Prediction
#         node_logits = self.output_head(h)
        
#         return node_logits


# diffusion_model_mis.py (最终完美复现版)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from model_components_mis import PrefixEncoder, DIFUSCOGNNLayer, SinusoidalTimestepEmbedding

class ConditionalMISSuffixDiffusionModel(nn.Module):
    """
    一个用于求解带前缀条件的MIS问题的扩散模型。
    该模型架构忠实地复现了DIFUSCO GNNEncoder，并适配了torch_geometric。
    """
    def __init__(self, node_feature_dim, node_embed_dim, gnn_n_layers, gnn_hidden_dim,
                 prefix_enc_hidden_dim, prefix_cond_dim, time_embed_dim):
        super().__init__()
        
        # 1. 初始节点特征嵌入层
        # 输入: [noisy_label, prefix_state]，维度为2
        # 输出: 统一的节点嵌入维度 node_embed_dim
        self.initial_node_feature_proj = nn.Linear(2, node_embed_dim)

        # 2. 时间步嵌入模块 (Time Embedding)
        # 将标量时间步 t 转换为一个高维向量
        self.time_embed_mlp = nn.Sequential(
            SinusoidalTimestepEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, gnn_hidden_dim) # 输出维度匹配GNN隐层
        )
        
        # 3. 前缀编码器 (Prefix Encoder)
        self.prefix_encoder = PrefixEncoder(
            node_feat_dim=node_embed_dim,
            hidden_dim=prefix_enc_hidden_dim,
            output_dim=prefix_cond_dim
        )
        # 将前缀条件向量投影到GNN隐层维度
        self.prefix_cond_proj = nn.Linear(prefix_cond_dim, gnn_hidden_dim)
        
        # 4. GNN网络
        # 将初始节点嵌入投影到GNN隐层维度
        self.gnn_input_proj = nn.Linear(node_embed_dim, gnn_hidden_dim)
        
        # 创建GNN层列表
        self.gnn_layers = nn.ModuleList([
            DIFUSCOGNNLayer(gnn_hidden_dim) for _ in range(gnn_n_layers)
        ])
        
        # --- 关键复现：为每一层创建独立的条件投影层 ---
        # 这对应DIFUSCO源码中每层都注入时间信息的逻辑
        self.time_proj_layers = nn.ModuleList([
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim) for _ in range(gnn_n_layers)
        ])
        self.prefix_proj_layers = nn.ModuleList([
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim) for _ in range(gnn_n_layers)
        ])
            
        # 5. 输出头 (Output Head)
        # 将最终的节点表示映射为两类（在或不在MIS中）的logits
        self.output_head = nn.Sequential(
            nn.LayerNorm(gnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(gnn_hidden_dim, 2)
        )

    def forward(self, graph_batch, t):
        x, edge_index, batch_vector = graph_batch.x, graph_batch.edge_index, graph_batch.batch
        node_prefix_state = graph_batch.node_prefix_state
        device = x.device

        # --- 步骤 1: 初始嵌入 ---
        initial_node_features_raw = torch.cat([x, node_prefix_state], dim=-1)
        node_emb = self.initial_node_feature_proj(initial_node_features_raw) # -> [total_nodes, node_embed_dim]
        
        # --- 步骤 2: 计算条件向量 ---
        time_emb = self.time_embed_mlp(t) # -> [batch_size, gnn_hidden_dim]
        
        prefix_cond_vector = torch.zeros(graph_batch.num_graphs, self.prefix_encoder.linear.out_features, 
                                         device=device, dtype=x.dtype)
        prefix_lengths = graph_batch.prefix_len
        has_prefix_mask = prefix_lengths > 0

        if has_prefix_mask.any():
            graphs_with_prefix = graph_batch.to_data_list()
            features_to_encode, lengths_to_encode = [], []
            node_counter = 0
            for i in range(graph_batch.num_graphs):
                num_nodes_in_graph = graphs_with_prefix[i].num_nodes
                if prefix_lengths[i] > 0:
                    prefix_indices_local = graphs_with_prefix[i].prefix_nodes
                    prefix_features = node_emb[node_counter:node_counter + num_nodes_in_graph][prefix_indices_local]
                    features_to_encode.append(prefix_features)
                    lengths_to_encode.append(prefix_lengths[i])
                node_counter += num_nodes_in_graph
            padded_features = rnn_utils.pad_sequence(features_to_encode, batch_first=True)
            computed_cond = self.prefix_encoder(padded_features, torch.tensor(lengths_to_encode, device=device))
            prefix_cond_vector[has_prefix_mask] = computed_cond

        prefix_cond_emb = self.prefix_cond_proj(prefix_cond_vector) # -> [batch_size, gnn_hidden_dim]
        
        # --- 步骤 3: GNN传播 ---
        # 将初始节点嵌入投影到GNN隐层维度
        h = self.gnn_input_proj(node_emb)
        
        # 核心循环：逐层进行GNN计算和条件注入
        for i, layer in enumerate(self.gnn_layers):
            # 广播并投影当前层的条件向量
            time_cond_per_node = self.time_proj_layers[i](time_emb)[batch_vector]
            prefix_cond_per_node = self.prefix_proj_layers[i](prefix_cond_emb)[batch_vector]
            
            # 注入条件：将条件信息加到节点特征上
            h = h + time_cond_per_node + prefix_cond_per_node
            
            # 执行GNN层的消息传递和更新
            h = layer(h, edge_index)
            
        # --- 步骤 4: 输出最终预测 ---
        node_logits = self.output_head(h) # -> [total_nodes, 2]
        
        return node_logits