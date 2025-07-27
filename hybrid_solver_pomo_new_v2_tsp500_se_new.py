# python hybrid_solver_am.py --config hybrid_eval_config.yaml

import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import importlib
import argparse
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
from tensordict import TensorDict
import inspect
from collections import defaultdict

# --- RL4CO Imports ---
from rl4co.envs import get_env
# from rl4co.models.zoo.am.policy import AttentionModelPolicy #不再直接需要
from rl4co.utils.ops import unbatchify

# --- Diffusion Model Imports ---
from diffusion_model_new import ConditionalTSPSuffixDiffusionModel
from discrete_diffusion_new_new_new import AdjacencyMatrixDiffusion

# --- Helper Function Imports ---
from evalutaion_GPU_v2 import calculate_tsp_cost_batch, visualize_tsp_tour, apply_2opt_batch

class HybridSolver:
    """
    Implements a theoretically-driven hybrid solving approach inspired by
    the concepts of semantic entropy and cognitive divergence.
    [修正版]
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Solver using device: {self.device}")

        self.rl_policy = self._load_rl_policy()
        self.dm_model = self._load_dm_model()
        
        self.diffusion_handler = AdjacencyMatrixDiffusion(
            num_nodes=cfg.model.num_nodes,
            num_timesteps=cfg.diffusion.num_timesteps,
            schedule_type=cfg.diffusion.schedule_type,
            device=self.device
        )

    def _load_rl_policy(self):
        print(f"Loading RL model from: {self.cfg.rl_model.ckpt_path}")
        try:
            # Load the checkpoint onto the CPU to bypass device-specific errors
            ckpt = torch.load(self.cfg.rl_model.ckpt_path, map_location='cpu')
            
            # Extract the hyperparameters
            hparams = ckpt.get('hyper_parameters', ckpt.get('hparams')) 
            if hparams is None:
                raise ValueError("Could not find hyperparameters in checkpoint.")

            # Get the class for our RL model, e.g., AttentionModel
            rl_model_cls = getattr(importlib.import_module("rl4co.models.zoo"), self.cfg.rl_model.name)

            # Programmatically find the valid arguments for the model's constructor
            valid_args = inspect.signature(rl_model_cls.__init__).parameters
            
            # Create a new, clean `hparams` dictionary containing only the valid arguments
            cleaned_hparams = {}
            for arg_name in valid_args:
                if arg_name in hparams:
                    cleaned_hparams[arg_name] = hparams[arg_name]
            
            # --- THE FINAL FIX ---
            # Explicitly remove the 'env' key since we are passing it separately.
            cleaned_hparams.pop('env', None)
            
            # Create a fresh environment and instantiate the model with the clean hparams
            env = get_env(self.cfg.rl_model.problem, generator_params={"num_loc": self.cfg.model.num_nodes})
            model = rl_model_cls(env=env, **cleaned_hparams)
            
            # Load the model's learned weights
            model.load_state_dict(ckpt['state_dict'],strict=False)

            # Move the policy to the correct device and return it
            policy = model.policy.to(self.device)
            policy.eval()
            return policy
        except Exception as e:
            print(f"Error loading RL model: {e}")
            print("This might be due to a version mismatch in the checkpoint file or rl4co library.")
            exit()
            
    def _load_dm_model(self):
        # ... (此函数无需修改，保持原样)
        print(f"Loading Diffusion model from: {self.cfg.dm_model.ckpt_path}")
        model = ConditionalTSPSuffixDiffusionModel(
            num_nodes=self.cfg.model.num_nodes, node_coord_dim=self.cfg.model.node_coord_dim,
            pos_embed_num_feats=self.cfg.model.pos_embed_num_feats, node_embed_dim=self.cfg.model.node_embed_dim,
            prefix_node_embed_dim=self.cfg.model.node_embed_dim,
            prefix_enc_hidden_dim=self.cfg.model.prefix_enc_hidden_dim, prefix_cond_dim=self.cfg.model.prefix_cond_dim,
            gnn_n_layers=self.cfg.model.gnn_n_layers, gnn_hidden_dim=self.cfg.model.gnn_hidden_dim,
            gnn_aggregation=self.cfg.model.gnn_aggregation, gnn_norm=self.cfg.model.gnn_norm,
            gnn_learn_norm=self.cfg.model.gnn_learn_norm, gnn_gated=self.cfg.model.gnn_gated,
            time_embed_dim=self.cfg.model.time_embed_dim
        ).to(self.device)
        model.load_state_dict(torch.load(self.cfg.dm_model.ckpt_path, map_location=self.device))
        model.eval()
        return model

    # ==========================================================================================
    #  [修正后的核心函数]
    # ==========================================================================================
    def _compute_dm_prior_scores(self, instance_locs, candidate_prefixes, prefix_lengths):
        """
        [修正版]
        Computes a cheap, single-step denoising score ("energy") for a batch of candidate prefixes.
        A lower score (energy) indicates the DM finds the prefix more "plausible" or "self-consistent".
        """
        total_candidates, N = candidate_prefixes.shape[0], self.cfg.model.num_nodes
        device = self.device

        if total_candidates == 0:
            return torch.empty(0, device=device)

        t_probe = torch.full((total_candidates,), self.cfg.solver.dm_probe_timestep, device=device, dtype=torch.long)

        prefix_adj_target = torch.zeros(total_candidates, N, N, device=device, dtype=torch.float)
        valid_prefix_mask = prefix_lengths > 1
        
        if valid_prefix_mask.any():
            # 使用一个更安全的方式来构建邻接矩阵
            for i in range(total_candidates):
                if prefix_lengths[i] > 1:
                    p_nodes = candidate_prefixes[i, :prefix_lengths[i]]
                    prefix_adj_target[i, p_nodes[:-1], p_nodes[1:]] = 1.0
                    prefix_adj_target[i, p_nodes[1:], p_nodes[:-1]] = 1.0

        x_t, _ = self.diffusion_handler.q_sample(prefix_adj_target, t_probe)
        x_t_transformed = x_t.float() * 2.0 - 1.0

        # [修正BUG 2]: 使用更鲁棒的 scatter_ 方法创建 node_prefix_state，避免索引错误
        node_prefix_state_probe = torch.zeros(total_candidates, N, 1, device=device)
        max_len = prefix_lengths.max().item()
        if max_len > 0:
            # .clone() 避免inplace操作问题
            prefixes_for_scatter = candidate_prefixes[:, :max_len].long().clone().unsqueeze(-1)
            # 准备与 index 形状匹配的 src
            src = torch.ones_like(prefixes_for_scatter, dtype=torch.float)
            # 创建mask以忽略padding值 (假设padding值为0)
            len_mask = torch.arange(max_len, device=device).unsqueeze(0) < prefix_lengths.unsqueeze(1)
            src[~len_mask.unsqueeze(-1)] = 0
            node_prefix_state_probe.scatter_(dim=1, index=prefixes_for_scatter, src=src)


        predicted_x_0_logits = self.dm_model(
            x_t_transformed, t_probe.float(), instance_locs,
            candidate_prefixes, prefix_lengths, node_prefix_state_probe
        )

        loss_mask = prefix_adj_target > 0
        if not loss_mask.any():
            return torch.full((total_candidates,), 999.0, device=device)

        reconstruction_loss = F.binary_cross_entropy_with_logits(
            predicted_x_0_logits[loss_mask],
            prefix_adj_target[loss_mask],
            reduction='none'
        )
        
        num_edges_per_prefix = (prefix_lengths - 1).clamp(min=0)
        total_loss_per_candidate = torch.zeros(total_candidates, device=device)
        
        loss_indices_map = torch.where(loss_mask)
        loss_idx_for_scatter = loss_indices_map[0]
        # 使用 scatter_add_ 高效求和. 注意，这里可能需要根据 PyTorch 版本微调
        # 一个更简单、清晰的方法是循环
        for i in range(total_candidates):
            total_loss_per_candidate[i] = reconstruction_loss[loss_indices_map[0] == i].sum()
        
        avg_loss_per_candidate = total_loss_per_candidate / (2 * num_edges_per_prefix.clamp(min=1).float())
        avg_loss_per_candidate[num_edges_per_prefix == 0] = 999.0
        
        return avg_loss_per_candidate
    
    # ==========================================================================================
    #  [修正后的主求解函数]
    # ==========================================================================================
    @torch.no_grad()
    def solve_batch_hybrid_vs_proposals(self, td, env):
        print("\n--- Running with SEAT-inspired, theory-driven trigger [FIXED] ---")
        B, N, _ = td['locs'].shape
        device = self.device
        
        # --- [你需要从你原来的代码中复制这些辅助函数过来] ---
        def decode_dm_heatmap_simple_greedy_batch(adj_matrices_probs, batch_prefix_nodes):
            B_decode, N_decode, _ = adj_matrices_probs.shape
            final_tours = torch.full((B_decode, N_decode), -1, dtype=torch.long, device=device)
            visited_mask = torch.zeros((B_decode, N_decode), dtype=torch.bool, device=device)
            
            # 使用前缀的第一个节点作为起点
            # 注意：batch_prefix_nodes应包含至少一个节点
            current_nodes = batch_prefix_nodes[:, 0]
            final_tours[:, 0] = current_nodes
            visited_mask.scatter_(1, current_nodes.unsqueeze(1), True)
        
            for i in range(1, N_decode):
                step_probs = adj_matrices_probs.clone()
                # 使用已经访问过的节点的mask来屏蔽选择
                current_node_mask = visited_mask.unsqueeze(1).expand(-1, N_decode, -1)
                step_probs.masked_fill_(current_node_mask, -1e9)
                
                next_node_probs = step_probs.gather(1, current_nodes.view(-1, 1, 1).expand(-1, -1, N_decode)).squeeze(1)
                next_nodes = torch.argmax(next_node_probs, dim=1)
                
                final_tours[:, i] = next_nodes
                visited_mask.scatter_(1, next_nodes.unsqueeze(1), True)
                current_nodes = next_nodes
                
            decoding_ok_mask = (final_tours != -1).all(dim=1)
            return final_tours, decoding_ok_mask

        def construct_tour_from_edges(edge_list, num_nodes, start_node=0):
            if not edge_list or len(edge_list) < num_nodes: return []
            adj = defaultdict(list)
            for u, v in edge_list:
                adj[u].append(v)
                adj[v].append(u)
            if start_node not in adj:
                start_node = next(iter(adj)) if adj else 0
            tour = [start_node]
            visited_nodes = {start_node}
            prev_node = -1
            curr_node = start_node
            while len(tour) < num_nodes:
                neighbors = adj.get(curr_node, [])
                next_node_found = False
                for neighbor in neighbors:
                    if neighbor != prev_node:
                        next_node = neighbor
                        next_node_found = True
                        break
                if not next_node_found or next_node in visited_nodes:
                    return []
                tour.append(next_node)
                visited_nodes.add(next_node)
                prev_node = curr_node
                curr_node = next_node
            return tour
        
        # --- 初始化 ---
        td_step = env.reset(td.clone())
        hybrid_solutions = torch.zeros(B, N, dtype=torch.long, device=device)
        dm_triggered_flags = torch.zeros(B, dtype=torch.bool, device=device)
        
        dm_proposal_stats = [{
            "cost": torch.tensor(float('inf'), device=device),
            "tour": torch.zeros(N, dtype=torch.long, device=device),
            "generation_step": -1, "prefix_node": -1, "rl_greedy_node_at_step": -1, "candidates_for_the_step": None
        } for _ in range(B)]

        node_embeds, _ = self.rl_policy.encoder(td_step)
        cached_embeds = self.rl_policy.decoder._precompute_cache(node_embeds)

        # --- 主循环 ---
        while td_step['i'][0] < N:
            step_idx = td_step['i'].squeeze(-1)
            current_step_scalar = step_idx[0].item()

            logits, _ = self.rl_policy.decoder(td_step, cached_embeds)
            mask = td_step["action_mask"]
            probs = F.softmax(logits + mask.log(), dim=-1)
            
            rl_greedy_choice = probs.argmax(-1)
            best_next_nodes_for_hybrid_path = rl_greedy_choice.clone()

            active_mask = ~td_step["done"].squeeze(-1)
            if not active_mask.any(): break
            
            probe_mask = active_mask &  ~dm_triggered_flags
            print(f"[DEBUG] entering loop, i={current_step_scalar}")

         
            if probe_mask.any() and self.cfg.solver.use_theory_trigger:
 
                indices_to_probe = probe_mask.nonzero().squeeze(-1)
                num_to_probe = len(indices_to_probe)
                
                M = self.cfg.solver.probe_rl_top_m
                probs_rl_probe = probs[probe_mask]
                num_available_actions = int(mask[probe_mask].sum(dim=1).min().item())
                M = min(M, num_available_actions)

                if M > 0:
                    top_m_probs, top_m_indices = torch.topk(probs_rl_probe, k=M, dim=1)
                    
                    # 计算策略熵
                    entropy_rl = -torch.sum(top_m_probs * torch.log(top_m_probs + 1e-9), dim=-1)
                    is_high_entropy = entropy_rl > self.cfg.solver.entropy_threshold
                    
                    # [修正BUG 2]: 区分第0步和后续步骤的触发逻辑
                    if current_step_scalar == 0:
                        # 在第0步，KL散度无意义，我们只依赖策略熵
                        print(f"[DEBUG] step 0, Entropy={entropy_rl.mean().item():.3f}. TOP-M probs = {top_m_probs} ; KL divergence is skipped.")
                        trigger_now_mask_relative = is_high_entropy
                    else:
                        # 在后续步骤，我们同时使用熵和KL散度
                        path_so_far = hybrid_solutions[probe_mask, :current_step_scalar]
                        expanded_paths = path_so_far.repeat_interleave(M, dim=0)
                        candidate_nodes = top_m_indices.reshape(-1, 1)
    
                        prefix_part = torch.cat([expanded_paths, candidate_nodes], dim=1)
                        padding = torch.zeros(prefix_part.shape[0], N - prefix_part.shape[1], dtype=torch.long, device=device)
                        candidate_prefixes = torch.cat([prefix_part, padding], dim=1)
                        prefix_lengths = torch.full((num_to_probe * M,), current_step_scalar + 1, device=device)
    
                        dm_to_instance_idx = torch.arange(num_to_probe, device=device).repeat_interleave(M)
                        expanded_locs = td['locs'][indices_to_probe][dm_to_instance_idx]
                        
                        dm_scores = self._compute_dm_prior_scores(
                            expanded_locs, candidate_prefixes, prefix_lengths
                        ).view(num_to_probe, M)
    
                        # [修正BUG 1]: 使用数值稳定的`F.kl_div`代替手动计算
                        # PyTorch的kl_div期望输入是log-probabilities
                        log_p_dm = F.log_softmax(-dm_scores / self.cfg.solver.dm_prior_temp, dim=-1)
                        p_rl = top_m_probs
                        
                        # F.kl_div(q.log(), p) 计算 D_KL(p || q), p和q分别是概率
                        # reduction='none' 使得我们可以手动对每个实例求和
                        kl_divergence = F.kl_div(log_p_dm, p_rl, reduction='none').sum(dim=-1)
                        
                        is_high_divergence = kl_divergence > self.cfg.solver.kl_div_threshold
                        trigger_now_mask_relative = is_high_entropy | is_high_divergence
                
                        print(f"[DEBUG] step {current_step_scalar}, TOP-M probs = {top_m_probs} entropy={entropy_rl.mean():.3f}， KL={kl_divergence.mean():.3f}")   



                    
                    if trigger_now_mask_relative.any():
                        absolute_trigger_indices = indices_to_probe[trigger_now_mask_relative]
                        print(f"--- Step {current_step_scalar}: Theory-trigger fired for {len(absolute_trigger_indices)} instances. ---")
                        dm_triggered_flags[absolute_trigger_indices] = True
                        
                        # --- [开始复用你原来的DM调用逻辑] ---
                        using_diffusion_mask = absolute_trigger_indices
                        num_uncertain = using_diffusion_mask.numel()
                        
                        probs_to_trigger = probs[using_diffusion_mask]
                        sorted_probs_trigger, sorted_indices_trigger = torch.sort(probs_to_trigger, dim=-1, descending=True)
                        cum_probs_trigger = torch.cumsum(sorted_probs_trigger, dim=-1)
                        cum_thresh = self.cfg.solver.dynamic_n_cumulative_threshold
                        dynamic_n_indices = torch.argmax((cum_probs_trigger >= cum_thresh).int(), dim=-1)
                        dynamic_n_candidates = dynamic_n_indices + 1
                        max_n_in_batch = int(dynamic_n_candidates.max().item())
                        
                        proposals = sorted_indices_trigger[:, :max_n_in_batch]
                        
                        path_so_far_triggered = hybrid_solutions[using_diffusion_mask, :current_step_scalar]
                        expanded_paths_triggered = path_so_far_triggered.repeat_interleave(dynamic_n_candidates, dim=0)
                        arange_mask = torch.arange(max_n_in_batch, device=device).unsqueeze(0)
                        selection_mask = arange_mask < dynamic_n_candidates.unsqueeze(1)
                        candidate_nodes_triggered = proposals[selection_mask]
                        
                        final_prefix_part = torch.cat([expanded_paths_triggered, candidate_nodes_triggered.unsqueeze(1)], dim=1)
                        final_padding = torch.zeros(final_prefix_part.shape[0], N - final_prefix_part.shape[1], dtype=torch.long, device=device)
                        final_prefixes = torch.cat([final_prefix_part, final_padding], dim=1)
                        
                        prefix_lengths_dm = torch.full((final_prefixes.shape[0],), current_step_scalar + 1, device=device)
                        dm_to_instance_idx_final = torch.arange(num_uncertain, device=device).repeat_interleave(dynamic_n_candidates)
                        expanded_locs_dm = td['locs'][using_diffusion_mask][dm_to_instance_idx_final]
                        
                        node_prefix_state_dm = torch.zeros(final_prefixes.shape[0], N, 1, device=device)
                        max_len_dm = prefix_lengths_dm.max().item()
                        prefixes_for_scatter_dm = final_prefixes[:, :max_len_dm].long().clone().unsqueeze(-1)
                        src_dm = torch.ones_like(prefixes_for_scatter_dm, dtype=torch.float)
                        len_mask_dm = torch.arange(max_len_dm, device=device).unsqueeze(0) < prefix_lengths_dm.unsqueeze(1)
                        src_dm[~len_mask_dm.unsqueeze(-1)] = 0
                        node_prefix_state_dm.scatter_(dim=1, index=prefixes_for_scatter_dm, src=src_dm)

                        _, generated_adj_matrices_probs = self.diffusion_handler.p_sample_loop_ddim(
                            denoiser_model=self.dm_model, instance_locs=expanded_locs_dm,
                            prefix_nodes=final_prefixes, prefix_lengths=prefix_lengths_dm,
                            node_prefix_state=node_prefix_state_dm,
                            num_inference_steps=self.cfg.solver.dm_inference_steps,
                            schedule=self.cfg.eval.inference_schedule_type
                        )

                        # 这里调用你自己的解码函数
                        decoded_tours, decoding_ok_mask = decode_dm_heatmap_simple_greedy_batch(generated_adj_matrices_probs, final_prefixes)
                        costs = torch.full((final_prefixes.shape[0],), float('inf'), device=device)
                        if decoding_ok_mask.any():
                             costs[decoding_ok_mask] = calculate_tsp_cost_batch(expanded_locs_dm[decoding_ok_mask], decoded_tours[decoding_ok_mask])
                        
                        costs_split = torch.split(costs, dynamic_n_candidates.cpu().tolist())
                        tours_split = torch.split(decoded_tours, dynamic_n_candidates.cpu().tolist())
                        
                        dm_chosen_nodes = torch.zeros(num_uncertain, dtype=torch.long, device=device)
                        for i in range(num_uncertain):
                            if len(costs_split[i]) == 0: continue
                            best_local_idx = torch.argmin(costs_split[i])
                            dm_chosen_nodes[i] = proposals[i, best_local_idx]
                            
                            best_dm_cost = costs_split[i][best_local_idx]
                            original_batch_idx = using_diffusion_mask[i].item()
                            if not torch.isinf(best_dm_cost) and best_dm_cost < dm_proposal_stats[original_batch_idx]["cost"]:
                                dm_proposal_stats[original_batch_idx].update({
                                    "cost": best_dm_cost, "tour": tours_split[i][best_local_idx],
                                    "generation_step": current_step_scalar, "prefix_node": proposals[i, best_local_idx].item(),
                                    "rl_greedy_node_at_step": rl_greedy_choice[original_batch_idx].item(),
                                    "candidates_for_the_step": proposals[i].cpu().numpy()
                                })
                        
                        best_next_nodes_for_hybrid_path[using_diffusion_mask] = dm_chosen_nodes
            
            hybrid_solutions[torch.arange(B), step_idx] = best_next_nodes_for_hybrid_path
            td_step.set("action", best_next_nodes_for_hybrid_path)
            td_step = env.step(td_step)["next"]

        # --- Final Selection and Statistics Logging (此部分无需修改) ---
        final_hybrid_costs = calculate_tsp_cost_batch(td['locs'], hybrid_solutions)
        final_solutions = torch.zeros(B, N, device=device, dtype=torch.long)
        run_statistics = [{} for _ in range(B)]

        for i in range(B):
            hybrid_cost = final_hybrid_costs[i]
            proposal_cost = dm_proposal_stats[i]["cost"]
            if proposal_cost < hybrid_cost:
                final_solutions[i] = dm_proposal_stats[i]["tour"]
                run_statistics[i] = {"best_cost": proposal_cost, "best_tour": dm_proposal_stats[i]["tour"], "source": "DM Proposal", **dm_proposal_stats[i]}
            else:
                final_solutions[i] = hybrid_solutions[i]
                run_statistics[i] = {"best_cost": hybrid_cost, "best_tour": hybrid_solutions[i], "source": "Hybrid Path", **dm_proposal_stats[i]}

        print("--- Hybrid-vs-Proposals run finished. Final selection complete. ---")
        return final_solutions, run_statistics

def run(cfg: DictConfig):
    solver = HybridSolver(cfg)
    device = solver.device
    env = get_env(cfg.rl_model.problem, generator_params={"num_loc": cfg.model.num_nodes})
    # 修改为你自己的测试数据路径
    dataset = env.dataset(filename=cfg.data.test_path)
    num_samples_to_evaluate = 100
    eval_dataset = torch.utils.data.Subset(dataset, range(num_samples_to_evaluate))
    dataloader = DataLoader(eval_dataset, batch_size=cfg.eval.batch_size, shuffle=False)

    all_stats = []
    all_gt_costs = [] # [修正] 新增列表，用于存储每个批次的GT成本
    start_time = time.time()
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Solving Batches")):
        td = TensorDict(batch, batch_size=batch['locs'].shape[0]).to(device)
        td['locs'] = td['locs'].float()
        
        solved_tours, batch_stats = solver.solve_batch_hybrid_vs_proposals(td, env)
        all_stats.extend(batch_stats)

        if cfg.solver.get("apply_two_opt", True):
            print("Applying 2-opt post-processing...")
            solved_tours = apply_2opt_batch(solved_tours, td['locs'])
            
        final_costs = calculate_tsp_cost_batch(td['locs'], solved_tours)
        # [修正] 计算并存储Ground Truth (Ordered) Cost
        gt_tour_indices = torch.arange(cfg.model.num_nodes, device=device).unsqueeze(0).repeat(td.shape[0], 1)
        gt_costs = calculate_tsp_cost_batch(td['locs'], gt_tour_indices)
        all_gt_costs.append(gt_costs.cpu())        
        # 更新batch_stats中的最终成本
        for i, stat in enumerate(batch_stats):
            stat['final_cost_after_2opt'] = final_costs[i].item()
    
    total_time = time.time() - start_time
    final_costs_all = [s['final_cost_after_2opt'] for s in all_stats]
    avg_final_cost = np.mean(final_costs_all)
    # [修正] 计算平均GT成本和差距
    gt_costs_tensor = torch.cat(all_gt_costs)
    avg_gt_cost = gt_costs_tensor.mean().item()
    optimality_gap = ((avg_final_cost / avg_gt_cost) - 1) * 100 if avg_gt_cost > 0 else float('inf')
    
    print("\n" + "=" * 60)
    print("--- Hybrid Solver Evaluation Summary ---")
    print(f"Total time: {total_time:.2f}s")
    print(f"Evaluated {len(all_stats)} instances.")
    print(f"Average Final Cost: {avg_final_cost:.4f}")
    # [修正] 打印您需要的统计信息
    print(f"Average Ground Truth (Ordered) Cost: {avg_gt_cost:.4f}")
    print(f"Optimality Gap vs Ordered GT: {optimality_gap:.2f}%")
    
    print("=" * 60)
    
    # ... (你原来的结果统计和可视化代码) ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid RL-DM Solver with Theory-Driven Trigger")
    parser.add_argument(
        "--config", type=str, default="hybrid_eval_config.yaml",
        help="Path to the unified YAML configuration file."
    )
    args = parser.parse_args()
    
    # 加载基础配置
    cfg = OmegaConf.load(args.config)
    
    # 你可以把默认的solver配置硬编码在这里，或者完全依赖yaml文件
    default_solver_cfg = OmegaConf.create({
        'solver': {
            'use_theory_trigger': True,
            'probe_rl_top_m': 6,
            'dm_probe_timestep': 500,
            'dm_prior_temp': 0.1,
            'entropy_threshold': 11.2,
            'kl_div_threshold': 10.14,
            'dm_inference_steps': 5,
            'dynamic_n_cumulative_threshold': 0.8,
            'apply_two_opt': False
            
        }
    })
    
    # 合并配置，yaml文件中的配置会覆盖这里的默认值
    cfg = OmegaConf.merge(default_solver_cfg, cfg)

    print("--- Running Hybrid Solver with Final Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("----------------------------------------------------")
    run(cfg)