# inference_mis.py
import os
import tqdm
import torch
import numpy as np
import networkx as nx
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader as GraphDataLoader
from tqdm.auto import tqdm

# 确保从您的项目中导入正确的模块
from data_loader_mis import MISConditionalPrefixDataset
from diffusion_model_mis import ConditionalMISSuffixDiffusionModel
from discrete_diffusion_mis import NodeLabelDiffusion

def decode_mis_solution(adj_matrix, predicted_labels):
    """
    对模型输出的节点标签进行解码，确保它是一个有效的独立集，并计算其大小。
    
    Args:
        adj_matrix (torch.Tensor): 图的邻接矩阵 (N, N)。
        predicted_labels (torch.Tensor): 模型预测的每个节点是否在MIS中的标签 (N,)。

    Returns:
        int: 解码后有效的独立集的大小。
    """
    num_nodes = adj_matrix.shape[0]
    
    # 确保预测标签是二进制的
    solution_nodes = (predicted_labels > 0.5).nonzero().squeeze(-1)
    
    # 检查独立集的有效性：集合中任意两个节点之间都不应该有边
    is_valid = True
    for i in range(len(solution_nodes)):
        for j in range(i + 1, len(solution_nodes)):
            u, v = solution_nodes[i], solution_nodes[j]
            if adj_matrix[u, v] > 0:
                is_valid = False
                break
        if not is_valid:
            break
            
    # 如果无效，可以采取修正措施（例如，贪心移除冲突节点），但这里为了简单起见，我们只报告大小
    # 一个简单的贪心解码策略是从高概率节点开始构建集合
    
    sorted_indices = torch.argsort(predicted_labels, descending=True)
    
    final_mis_set = []
    is_in_mis = torch.zeros(num_nodes, dtype=torch.bool)
    
    for node_idx in sorted_indices:
        # 检查当前节点是否可以加入MIS
        # 即它不与任何已在MIS中的节点相邻
        can_add = True
        neighbors = adj_matrix[node_idx].nonzero().squeeze(-1)
        for neighbor in neighbors:
            if is_in_mis[neighbor]:
                can_add = False
                break
        
        if can_add:
            final_mis_set.append(node_idx.item())
            is_in_mis[node_idx] = True
            
    return len(final_mis_set)


def run_inference():
    """
    主函数，用于加载模型并执行推理。
    """
    print("--- MIS Diffusion Model Inference ---")

    # 1. 加载配置文件和设置
    try:
        config = OmegaConf.load("mis_config.yaml")
    except FileNotFoundError:
        print("错误: 找不到 mis_config.yaml 文件。请确保该文件在当前目录下。")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    # 2. 加载训练好的模型
    model_path = os.path.join(config.train.ckpt_dir, "best_model_stage2_medium_prefix.pth")
    if not os.path.exists(model_path):
        print(f"错误: 找不到训练好的模型文件: {model_path}")
        print("请先完成训练，或将模型文件路径修改为正确的路径。")
        return

    model = ConditionalMISSuffixDiffusionModel(
        node_feature_dim=config.model.node_feature_dim,
        node_embed_dim=config.model.node_embed_dim,
        gnn_n_layers=config.model.gnn_n_layers,
        gnn_hidden_dim=config.model.gnn_hidden_dim,
        prefix_enc_hidden_dim=config.model.prefix_enc_hidden_dim,
        prefix_cond_dim=config.model.prefix_cond_dim,
        time_embed_dim=config.model.time_embed_dim
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"成功加载模型: {model_path}")

    # 3. 准备测试数据集和DataLoader
    # 使用较小的前缀来测试模型的泛化和补全能力
    inference_prefix_options = [0.2] 
    test_dataset = MISConditionalPrefixDataset(
        data_dir=config.data.test_path, # 使用验证集进行测试
        prefix_k_options=inference_prefix_options,
        prefix_sampling_strategy='scattered'
    )
    # 使用较小的batch_size进行推理
    test_dataloader = GraphDataLoader(test_dataset, batch_size=64, shuffle=False)

    # 4. 初始化扩散处理器 (Diffusion Handler)
    diffusion_handler = NodeLabelDiffusion(
        num_timesteps=config.diffusion.num_timesteps,
        schedule_type=config.diffusion.schedule_type,
        device=device
    )

    # 5. 执行推理循环
    print("\n开始推理...")
    total_gt_size = 0
    total_pred_size = 0
    num_graphs_processed = 0

    # # 我们只测试一个批次作为示例
    # try:
    #     graph_batch = next(iter(test_dataloader)).to(device)
    # except StopIteration:
    #     print("错误: 数据加载器为空，无法获取测试数据。")
    #     return

    # # 运行DDIM采样来生成解
    # final_labels, final_probs = diffusion_handler.p_sample_loop_ddim(
    #     denoiser_model=model,
    #     graph_batch=graph_batch,
    #     num_inference_steps=100  # 使用较少的步数以加快推理速度
    # )

    # === 批次处理所有测试图 ===
    for graph_batch in tqdm(test_dataloader, desc="Processing batches"):
        graph_batch = graph_batch.to(device)
    
        # 运行 DDIM 采样生成解
        final_labels, final_probs = diffusion_handler.p_sample_loop_ddim(
            denoiser_model=model,
            graph_batch=graph_batch,
            num_inference_steps=50  # 可根据需要调整步数
        )
    
        # 解码与评估结果
        graphs = graph_batch.to_data_list()
        node_counter = 0
        for i, graph in enumerate(graphs):
            num_nodes_in_graph = graph.num_nodes
            
            # 获取预测结果对应图的那一段
            pred_labels_for_graph = final_labels[node_counter : node_counter + num_nodes_in_graph]
            pred_probs_for_graph = final_probs[node_counter : node_counter + num_nodes_in_graph]
    
            # 构造邻接矩阵
            with open(test_dataset.file_paths[num_graphs_processed], 'rb') as f:
                G = pickle.load(f)
            node_list = sorted(G.nodes())
            adj_matrix = torch.from_numpy(nx.to_scipy_sparse_array(G, nodelist=node_list).toarray()).to(device)
    
            # 解码预测结果
            predicted_mis_size = decode_mis_solution(adj_matrix, pred_probs_for_graph)
    
            # 获取真实MIS大小
            ground_truth_mis_size = graph.x_true.sum().item()
    
            print(f"图 {num_graphs_processed+1} (节点数: {num_nodes_in_graph}):")
            print(f"  真实MIS大小: {ground_truth_mis_size}")
            print(f"  预测MIS大小: {predicted_mis_size}")
            print(f"  性能差距 (Gap): {(ground_truth_mis_size - predicted_mis_size) / ground_truth_mis_size:.2%}\n")
    
            total_gt_size += ground_truth_mis_size
            total_pred_size += predicted_mis_size
            node_counter += num_nodes_in_graph
            num_graphs_processed += 1


        
    print("--- 批次总结 ---")
    avg_gap = (total_gt_size - total_pred_size) / total_gt_size if total_gt_size > 0 else 0
    print(f"处理图数量: {num_graphs_processed}")
    print(f"平均真实MIS大小: {total_gt_size / num_graphs_processed:.2f}")
    print(f"平均预测MIS大小: {total_pred_size / num_graphs_processed:.2f}")
    print(f"平均性能差距: {avg_gap:.2%}")


if __name__ == "__main__":
    # 确保我们有pickle5来加载数据
    try:
        import pickle
    except ImportError:
        print("警告: 未找到 `pickle5`。如果您的gpickle文件是用Python 3.8+保存的，则不需要。否则请 `pip install pickle5`")
        import pickle

    run_inference()