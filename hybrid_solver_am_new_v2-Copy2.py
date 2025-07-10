#python hybrid_solver_am.py --config hybrid_eval_config.yaml

import torch
import torch.nn.functional as F
import numpy as np
import yaml
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
from rl4co.models.zoo.am.policy import AttentionModelPolicy
from rl4co.utils.ops import unbatchify

# --- Diffusion Model Imports ---
from diffusion_model_new import ConditionalTSPSuffixDiffusionModel
from discrete_diffusion_new_new_new import AdjacencyMatrixDiffusion

# --- Helper Function Imports ---
from evalutaion_GPU_v2 import calculate_tsp_cost_batch, visualize_tsp_tour,apply_2opt_batch
import inspect # Make sure to add this import at the top of your file

class HybridSolver:
    """
    Implements the intelligent STEP-BY-STEP hybrid solving approach.
    This version correctly interfaces with the rl4co environment and decoder
    to get accurate step-by-step probabilities for uncertainty-driven DM calls.
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
            model.load_state_dict(ckpt['state_dict'])

            # Move the policy to the correct device and return it
            policy = model.policy.to(self.device)
            policy.eval()
            return policy

        except Exception as e:
            print(f"Error loading RL model: {e}")
            print("This might be due to a version mismatch in the checkpoint file or rl4co library.")
            exit()


    def _load_dm_model(self):
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
        model.load_state_dict(torch.load(self.cfg.dm_model.ckpt_path, map_location=self.device)) # removed weights_only=True for compatibility
        model.eval()
        return model


    @torch.no_grad()
    def analyze_rl_certainty(self, td, env):
        """
        Runs the RL policy without DM calls to collect data on decision certainty.
        This helps in calibrating the uncertainty thresholds.
        """
        print("--- Starting RL Certainty Analysis Run ---")
        B, N, _ = td['locs'].shape
        
        # 存储每一步的分析数据
        all_confidence_ratios = []
        all_p1_values = []
        all_top_5_gaps = []
        all_cum_probs_n5 = [] # 累积概率 for Top-5

        td_step = env.reset(td.clone())
        node_embeds, _ = self.rl_policy.encoder(td_step)
        cached_embeds = self.rl_policy.decoder._precompute_cache(node_embeds)
        self.rl_policy.decoder.eval()

        while not td_step["done"].all():
            step_idx = td_step['i'].squeeze(-1)
            active_mask = ~td_step["done"].squeeze(-1)
            if not active_mask.any(): break

            logits, _ = self.rl_policy.decoder(td_step, cached_embeds)
            mask = td_step["action_mask"]
            probs = F.softmax(logits + mask.log(), dim=-1)
            
            # --- 收集需要分析的数据 ---
            if active_mask.any():
                active_probs = probs[active_mask]
                sorted_probs, _ = torch.sort(active_probs, dim=-1, descending=True)
                
                p1 = sorted_probs[:, 0]
                p2 = sorted_probs[:, 1]
                confidence_ratio = p1 / (p2 + 1e-9)
                
                # 计算前5个概率之间的gap
                gaps = sorted_probs[:, :5] - sorted_probs[:, 1:6] if N > 5 else sorted_probs[:, :-1] - sorted_probs[:, 1:]

                # 计算Top-5的累积概率
                cum_prob_5 = torch.sum(sorted_probs[:, :5], dim=-1)

                all_confidence_ratios.append(confidence_ratio.cpu())
                all_p1_values.append(p1.cpu())
                all_top_5_gaps.append(gaps.cpu())
                all_cum_probs_n5.append(cum_prob_5.cpu())

            # 仅使用RL策略进行下一步
            best_next_nodes = probs.argmax(-1)
            td_step.set("action", best_next_nodes)
            td_step = env.step(td_step)["next"]
        
        # --- 绘图和分析 ---
        all_confidence_ratios = torch.cat(all_confidence_ratios)
        all_p1_values = torch.cat(all_p1_values)
        all_top_5_gaps = torch.cat(all_top_5_gaps).numpy()

        plt.figure(figsize=(18, 5))
        
        plt.subplot(1, 3, 1)
        plt.hist(all_confidence_ratios.log10().numpy(), bins=50, alpha=0.7)
        plt.axvline(np.log10(2.0), color='r', linestyle='--', label='Ratio=2.0')
        plt.axvline(np.log10(4.0), color='g', linestyle='--', label='Ratio=4.0')
        plt.xlabel("Confidence Ratio (log10 scale)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Confidence Ratios")
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.hist(all_p1_values.numpy(), bins=50, alpha=0.7)
        plt.xlabel("Probability of Best Action (p1)")
        plt.ylabel("Frequency")
        plt.title("Distribution of p1")
        
        plt.subplot(1, 3, 3)
        plt.hist(all_top_5_gaps[:, 0], bins=50, alpha=0.7, label='Gap p1-p2')
        plt.hist(all_top_5_gaps[:, 1], bins=50, alpha=0.5, label='Gap p2-p3')
        plt.xlabel("Probability Gap between Top Actions")
        plt.ylabel("Frequency")
        plt.title("Distribution of Gaps")
        plt.legend()
        
        save_path = "rl_certainty_analysis.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Certainty analysis plot saved to {save_path}")

        # 打印一些统计数据以帮助决策
        print("\n--- Statistical Analysis for Threshold Setting ---")
        print(f"Confidence Ratio Percentiles (p50, p75, p90): {np.percentile(all_confidence_ratios, [50, 75, 90])}")
        print(f"Gap (p1-p2) Percentiles (p10, p25, p50): {np.percentile(all_top_5_gaps[:, 0], [10, 25, 50])}")

    @torch.no_grad()
    def analyze_rl_certainty_by_step(self, td, env):
        """
        NEW & IMPROVED: Runs the RL policy to collect and visualize decision certainty METRICS PER STEP.
        This provides a much deeper insight into how uncertainty evolves during the decoding process.
        """
        print("--- Starting Step-wise RL Certainty Analysis ---")
        B, N, _ = td['locs'].shape
        
        # Use defaultdict to easily append data for each step
        stats_by_step = defaultdict(lambda: {
            'confidence_ratios': [],
            'p1_values': [],
            'gaps_p1_p2': []
        })

        td_step = env.reset(td.clone())
        node_embeds, _ = self.rl_policy.encoder(td_step)
        cached_embeds = self.rl_policy.decoder._precompute_cache(node_embeds)
        self.rl_policy.decoder.eval()

        while not td_step["done"].all() and td_step['i'][0].item() < 99:
            current_step_index = td_step['i'][0].item()
            active_mask = ~td_step["done"].squeeze(-1)
            if not active_mask.any(): break

            # ===================
            logits, _ = self.rl_policy.decoder(td_step, cached_embeds)
            mask = td_step["action_mask"]
            probs = F.softmax(logits + mask.log(), dim=-1)
            
            if active_mask.any():
                active_probs = probs[active_mask]
                sorted_probs, _ = torch.sort(active_probs, dim=-1, descending=True)
                
                p1 = sorted_probs[:, 0]
                p2 = sorted_probs[:, 1]
                confidence_ratio = p1 / (p2 + 1e-9)
                gap_p1_p2 = p1 - p2
                
                # Store metrics for the current step
                stats_by_step[current_step_index]['confidence_ratios'].extend(confidence_ratio.cpu().numpy())
                stats_by_step[current_step_index]['p1_values'].extend(p1.cpu().numpy())
                stats_by_step[current_step_index]['gaps_p1_p2'].extend(gap_p1_p2.cpu().numpy())

            best_next_nodes = probs.argmax(-1)
            td_step.set("action", best_next_nodes)
            td_step = env.step(td_step)["next"]
        
        # --- Create Step-wise Visualization ---
        fig, axes = plt.subplots(3, 1, figsize=(15, 18), sharex=True)
        steps = sorted(stats_by_step.keys())

        # 1. Confidence Ratio vs. Step
        ratios_data = [stats_by_step[s]['confidence_ratios'] for s in steps]
        axes[0].boxplot(ratios_data, labels=steps, showfliers=False) # Hide outliers for better scale
        axes[0].set_title("Confidence Ratio (p1/p2) vs. Decoding Step")
        axes[0].set_ylabel("Confidence Ratio")
        axes[0].axhline(y=1, color='r', linestyle='--', label='Ratio=1.0') # Example threshold line
        axes[0].legend()

        # 2. P1 Value vs. Step
        p1_data = [stats_by_step[s]['p1_values'] for s in steps]
        axes[1].boxplot(p1_data, labels=steps)
        axes[1].set_title("Probability of Best Action (p1) vs. Decoding Step")
        axes[1].set_ylabel("Probability (p1)")

        # 3. Gap (p1-p2) vs. Step
        gap_data = [stats_by_step[s]['gaps_p1_p2'] for s in steps]
        axes[2].boxplot(gap_data, labels=steps)
        axes[2].set_title("Gap (p1 - p2) vs. Decoding Step")
        axes[2].set_ylabel("Probability Gap")
        axes[2].axhline(y=0.1, color='g', linestyle='--', label='Gap=0.1') # Example threshold line
        axes[2].legend()

        axes[2].set_xlabel("Decoding Step Index (k)")
        fig.tight_layout()
        
        save_path = "rl_certainty_analysis_by_step.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Step-wise certainty analysis plot saved to {save_path}")

        print("\n--- How to Interpret the Step-wise Plot ---")
        print("1. Observe the TREND of the boxes (median and spread) in each plot.")
        print("2. Confidence Ratio (Top Plot): Does it generally increase with steps? This is expected.")
        print("   -> Your dynamic threshold for `confidence_ratio` could be a function of the step, e.g., a decreasing curve.")
        print("3. Gap p1-p2 (Bottom Plot): Does the median gap increase? Also expected.")
        print("   -> Your dynamic threshold for `max_gap_threshold` could also be a function, perhaps increasing slightly with steps.")
        print("4. This analysis allows you to define different thresholds for different stages, e.g., steps 0-20 vs 21-80 vs 81-99.")
    
    def _check_uncertainty_hierarchical(self, probs, active_mask, current_step):
        """
        MODIFIED: This function now ONLY decides IF a step is uncertain.
        The calculation of HOW MANY candidates to check is moved to solve_batch.
        """
        B, N_nodes = probs.shape
        final_uncertain_mask = torch.zeros_like(active_mask, dtype=torch.bool)

        if not active_mask.any():
            return final_uncertain_mask
        
        if current_step < 1:
            confidence_ratio_threshold = 115.0  # The parameter i set here is to make sure that only the first step will use diffusion model
            max_gap_threshold = 10.1
        elif current_step < 60:
            confidence_ratio_threshold = 1.0
            max_gap_threshold = 0.05
        else:
            confidence_ratio_threshold = 1.0
            max_gap_threshold = 0.05
            
        # =============================================================

        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        p1, p2 = sorted_probs[:, 0], sorted_probs[:, 1]
        
        # --- Tier 1 Check: Confidence Ratio ---
        is_confident_mask = (p1 / (p2 + 1e-9)) > confidence_ratio_threshold
        pool_for_second_check_mask = active_mask & ~is_confident_mask
        
        if not pool_for_second_check_mask.any():
            return final_uncertain_mask

        # --- Tier 2 Check: Group Hesitation ---
        # We now check up to a FIXED number of gaps, e.g., the top 10,
        # as the dynamic `n` is now handled later.
        # This is a simplified but effective check for "group hesitation".
        fixed_n_check = min(10, N_nodes - 1)
        gaps = sorted_probs[:, :fixed_n_check] - sorted_probs[:, 1:fixed_n_check+1]
        
        # If all gaps within this fixed check window are small, it's uncertain.
        all_gaps_are_small_mask = (gaps < max_gap_threshold).all(dim=-1)

        uncertain_in_second_check = pool_for_second_check_mask & all_gaps_are_small_mask
        final_uncertain_mask[uncertain_in_second_check] = True
        
        return final_uncertain_mask

    @torch.no_grad()
    def solve_batch_hybrid_vs_proposals(self, td, env):
        """
        MODIFIED HYBRID-vs-PROPOSALS VERSION WITH DETAILED TRACKING:
        This version tracks the origin of the final best tour.
        """
        print("\n--- Running in HYBRID-vs-PROPOSALS mode with detailed tracking ---")
        B, N, _ = td['locs'].shape
        device = self.device
        
        # --- Helper functions ---

        def decode_dm_heatmap_simple_greedy_batch(adj_matrices_probs, batch_prefix_nodes):
            """
            一个完全并行的、基于简单贪心策略的解码器。
            它避免了任何跨批次的Python循环，速度极快。
            
            adj_matrices_probs: (B, N, N) - DM输出的邻接矩阵概率
            batch_prefix_nodes: (B, N) - 每个实例的完整前缀路径
            """
            B, N, _ = adj_matrices_probs.shape
            device = adj_matrices_probs.device
            
            # 1. 初始化
            final_tours = torch.full((B, N), -1, dtype=torch.long, device=device)
            # 使用一个掩码来跟踪已访问的节点
            visited_mask = torch.zeros((B, N), dtype=torch.bool, device=device)
            
            # 2. 从前缀的第一个节点开始
            # 假设prefix_nodes的第一列是起始节点
            current_nodes = batch_prefix_nodes[:, 0]
            final_tours[:, 0] = current_nodes
            
            # 在掩码中标记起始节点为已访问
            # 使用scatter_来并行更新所有批次实例的掩码
            visited_mask.scatter_(1, current_nodes.unsqueeze(1), True)
        
            # 3. 并行地、一步步地构建路径
            for i in range(1, N):
                # 将已访问节点的概率设置为一个非常低的值，以避免选择它们
                step_probs = adj_matrices_probs.clone()
                step_probs.masked_fill_(visited_mask.unsqueeze(1), -1e9) # (B, N, N) -> (B, N_visited, N)
                
                # 从每个当前节点，找到到未访问节点中概率最高的下一个节点
                # gather用于并行地为每个实例选择其“当前节点”对应的行
                next_node_probs = step_probs.gather(1, current_nodes.view(B, 1, 1).expand(-1, -1, N)).squeeze(1) # (B, N)
                
                # 选出概率最高的下一个节点
                next_nodes = torch.argmax(next_node_probs, dim=1) # (B,)
                
                # 更新路径和状态
                final_tours[:, i] = next_nodes
                visited_mask.scatter_(1, next_nodes.unsqueeze(1), True)
                current_nodes = next_nodes
                
            # 检查解码是否成功 (所有节点都被访问过)
            # 对于这个简单的贪心算法，理论上总是成功的
            decoding_ok_mask = (final_tours != -1).all(dim=1)
            
            return final_tours, decoding_ok_mask

        
        def decode_dm_heatmap_edge_greedy_batch(adj_matrices_probs, instance_locs, batch_prefix_nodes):
            B_decode, N_decode, _ = adj_matrices_probs.shape
            device_decode = adj_matrices_probs.device
            adj_probs = (adj_matrices_probs + adj_matrices_probs.transpose(1, 2)) / 2.0
            dists = torch.cdist(instance_locs, instance_locs, p=2) + 1e-9
            edge_scores = adj_probs / dists
            indices = torch.triu_indices(N_decode, N_decode, offset=1, device=device_decode)
            flat_scores = edge_scores[:, indices[0], indices[1]]
            _, sorted_indices = torch.sort(flat_scores, dim=1, descending=True)
            sorted_edges_u = indices[0][sorted_indices]
            sorted_edges_v = indices[1][sorted_indices]
            final_tours = torch.full((B_decode, N_decode), -1, dtype=torch.long, device=device_decode)
            for i in range(B_decode):
                parent = torch.arange(N_decode, device=device_decode)
                def find_set(v):
                    if v == parent[v]: return v
                    parent[v] = find_set(parent[v]); return parent[v]
                def unite_sets(a, b):
                    a, b = find_set(a), find_set(b)
                    if a != b: parent[b] = a
                node_degrees = torch.zeros(N_decode, dtype=torch.int, device=device_decode)
                edges_in_tour = []
                prefix_nodes = batch_prefix_nodes[i]
                prefix_len = (prefix_nodes != -1).sum().item()
                prefix_nodes = prefix_nodes[:prefix_len]
                if prefix_len > 1:
                    for j in range(prefix_len - 1):
                        u, v = prefix_nodes[j].item(), prefix_nodes[j+1].item()
                        if node_degrees[u] >= 2 or node_degrees[v] >= 2: continue
                        edges_in_tour.append((u, v))
                        node_degrees[u] += 1
                        node_degrees[v] += 1
                        unite_sets(u, v)
                for u_tensor, v_tensor in zip(sorted_edges_u[i], sorted_edges_v[i]):
                    if len(edges_in_tour) >= N_decode - 1: break
                    u, v = u_tensor.item(), v_tensor.item()
                    is_prefix = False
                    if prefix_len > 1:
                        for j in range(prefix_len - 1):
                            p_u, p_v = prefix_nodes[j].item(), prefix_nodes[j+1].item()
                            if (u == p_u and v == p_v) or (u == p_v and v == p_u):
                                is_prefix = True; break
                    if is_prefix: continue
                    if node_degrees[u] < 2 and node_degrees[v] < 2 and find_set(u) != find_set(v):
                        edges_in_tour.append((u, v))
                        node_degrees[u] += 1
                        node_degrees[v] += 1
                        unite_sets(u, v)
                if len(edges_in_tour) == N_decode - 1:
                    endpoints = (node_degrees == 1).nonzero(as_tuple=True)[0]
                    if len(endpoints) == 2:
                        u, v = endpoints[0].item(), endpoints[1].item()
                        edges_in_tour.append((u, v))
                if len(edges_in_tour) == N_decode:
                    start_node = prefix_nodes[0].item() if prefix_len > 0 else 0
                    tour_sequence = construct_tour_from_edges(edges_in_tour, N_decode, start_node=start_node)
                    if tour_sequence and len(tour_sequence) == N_decode:
                        final_tours[i] = torch.tensor(tour_sequence, device=device_decode)
            return final_tours
        
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

        # --- Initialization ---
        td_step = env.reset(td.clone())
        hybrid_solutions = torch.zeros(B, N, device=device, dtype=torch.long)
        
        # NEW: Structure to track the best DM proposal and its metadata
        dm_proposal_stats = [{
            "cost": torch.tensor(float('inf'), device=device),
            "tour": torch.zeros(N, dtype=torch.long, device=device),
            "generation_step": -1,
            "prefix_node": -1,
            "rl_greedy_node_at_step": -1
        } for _ in range(B)]

        node_embeds, _ = self.rl_policy.encoder(td_step)
        cached_embeds = self.rl_policy.decoder._precompute_cache(node_embeds)
        self.rl_policy.decoder.eval()

        # --- Main Loop (builds the hybrid path) ---
        while td_step['i'][0] < N:
            step_idx = td_step['i'].squeeze(-1)
            logits, _ = self.rl_policy.decoder(td_step, cached_embeds)
            mask = td_step["action_mask"]
            probs = F.softmax(logits + mask.log(), dim=-1)
            
            rl_greedy_choice = probs.argmax(-1)
            best_next_nodes = rl_greedy_choice.clone()
            
            active_mask = ~td_step["done"].squeeze(-1)
            if not active_mask.any(): break
            
            current_step_scalar = step_idx[0].item()
            is_uncertain_mask = self._check_uncertainty_hierarchical(probs, active_mask, current_step_scalar)
            uncertain_indices = is_uncertain_mask.nonzero().squeeze(-1)
            print(f"Step {step_idx[0].item()}Uncertain step ratio: {uncertain_indices.numel()} / {active_mask.sum().item()}")

            if uncertain_indices.numel() > 0:
                # --- Standard DM call logic ---
                num_uncertain = uncertain_indices.numel()
                uncertain_probs = probs[uncertain_indices]
                # ... (code for dynamic_n_candidates, proposals, final_prefixes is the same)
                sorted_uncertain_probs, _ = torch.sort(uncertain_probs, dim=-1, descending=True)
                cum_probs = torch.cumsum(sorted_uncertain_probs, dim=-1)
                cumulative_threshold = self.cfg.solver.get("dynamic_n_cumulative_threshold", 0.8)
                dynamic_n_indices = torch.argmax((cum_probs >= cumulative_threshold).int(), dim=-1)
                dynamic_n_candidates = dynamic_n_indices + 1
                max_n_in_batch = dynamic_n_candidates.max().item()
                proposals = torch.topk(uncertain_probs, k=max_n_in_batch, dim=1)[1]
                path_so_far = hybrid_solutions[uncertain_indices, :current_step_scalar]
                expanded_paths = path_so_far.repeat_interleave(dynamic_n_candidates, dim=0)
                arange_mask = torch.arange(max_n_in_batch, device=self.device).unsqueeze(0)
                selection_mask = arange_mask < dynamic_n_candidates.unsqueeze(1)
                candidate_nodes = proposals[selection_mask]
                final_prefixes = torch.cat([expanded_paths, candidate_nodes.unsqueeze(1)], dim=1)
                # --- DM simulation setup and call ---
                # ... (code for prefix_lengths, expanded_locs, ..., p_sample_loop is the same)
                prefix_lengths = torch.full((final_prefixes.shape[0],), current_step_scalar + 1, device=self.device)
                total_dm_simulations = final_prefixes.shape[0]
                dm_to_instance_idx = torch.arange(num_uncertain, device=self.device).repeat_interleave(dynamic_n_candidates)
                expanded_locs = td['locs'][uncertain_indices][dm_to_instance_idx]
                node_prefix_state_dm = torch.zeros(total_dm_simulations, N, 1, device=self.device)
                index_for_scatter = final_prefixes.long().unsqueeze(-1)
                src_for_scatter = torch.ones_like(index_for_scatter, dtype=torch.float32)
                node_prefix_state_dm.scatter_(1, index_for_scatter, src_for_scatter)
                _, generated_adj_matrices_probs = self.diffusion_handler.p_sample_loop_ddim(
                    denoiser_model=self.dm_model, instance_locs=expanded_locs,
                    prefix_nodes=final_prefixes, prefix_lengths=prefix_lengths,
                    node_prefix_state=node_prefix_state_dm,
                    num_inference_steps=self.cfg.solver.dm_inference_steps,
                    schedule=self.cfg.eval.inference_schedule_type
                )
                decoded_tours, decoding_ok_mask = decode_dm_heatmap_simple_greedy_batch(
                    generated_adj_matrices_probs, 
                    final_prefixes # final_prefixes是我们在这里需要的完整前缀
                )
                costs = torch.full((total_dm_simulations,), float('inf'), device=self.device)
                if decoding_ok_mask.any():
                    # 仅为成功解码的实例计算成本
                    valid_locs = expanded_locs[decoding_ok_mask]
                    valid_tours = decoded_tours[decoding_ok_mask]
                    costs[decoding_ok_mask] = calculate_tsp_cost_batch(valid_locs, valid_tours)
                
                # --- Find best NEXT NODE for hybrid path AND best COMPLETE TOUR proposal ---
                costs_split = torch.split(costs, dynamic_n_candidates.cpu().tolist())
                tours_split = torch.split(decoded_tours, dynamic_n_candidates.cpu().tolist())
                
                dm_chosen_nodes = torch.zeros(num_uncertain, dtype=torch.long, device=self.device)
                for i in range(num_uncertain):
                    candidates_for_best = proposals[i].cpu().numpy()

                    if len(costs_split[i]) == 0: continue
                    best_local_idx = torch.argmin(costs_split[i])
                    
                    # 1. Determine best next node for the hybrid path
                    dm_chosen_nodes[i] = proposals[i, best_local_idx]

                    # 2. Check if the best complete tour from this call is the best proposal so far
                    best_dm_cost = costs_split[i][best_local_idx]
                    original_batch_idx = uncertain_indices[i].item()
                    if not torch.isinf(best_dm_cost) and best_dm_cost < dm_proposal_stats[original_batch_idx]["cost"]:
                        dm_proposal_stats[original_batch_idx] = {
                            "cost": best_dm_cost,
                            "tour": tours_split[i][best_local_idx],
                            "generation_step": current_step_scalar,
                            "candidates_for_the_step":candidates_for_best,
                            "prefix_node": proposals[i, best_local_idx].item(),
                            "rl_greedy_node_at_step": rl_greedy_choice[original_batch_idx].item()
                        }
                
                # Update the action for the hybrid path
                best_next_nodes[uncertain_indices] = dm_chosen_nodes

            hybrid_solutions[torch.arange(B), step_idx] = best_next_nodes
            td_step.set("action", best_next_nodes)
            td_step = env.step(td_step)["next"]
        
        # --- Final Selection and Statistics Logging ---
        final_hybrid_costs = calculate_tsp_cost_batch(td['locs'], hybrid_solutions)
        final_solutions = torch.zeros(B, N, device=device, dtype=torch.long)
        
        run_statistics = [{} for _ in range(B)] # Final stats to be returned

        for i in range(B):
            hybrid_cost = final_hybrid_costs[i]
            proposal_cost = dm_proposal_stats[i]["cost"]

            if proposal_cost < hybrid_cost:
                # The best tour was a complete proposal from a DM call
                final_solutions[i] = dm_proposal_stats[i]["tour"]
                run_statistics[i] = {
                    "best_cost": proposal_cost,
                    "best_tour": dm_proposal_stats[i]["tour"],
                    "source": "DM Proposal",
                    **dm_proposal_stats[i] # Copy all detailed stats
                }
            else:
                # The step-by-step hybrid path was better
                final_solutions[i] = hybrid_solutions[i]
                run_statistics[i] = {
                    "best_cost": hybrid_cost,
                    "best_tour": hybrid_solutions[i],
                    "source": "Hybrid Path",
                    "generation_step": -1,
                    "prefix_node": -1,
                    "rl_greedy_node_at_step": -1
                }
        
        print("--- Hybrid-vs-Proposals run finished. Final selection complete. ---")
        return final_solutions, run_statistics

def run(cfg: DictConfig):
    solver = HybridSolver(cfg)
    device = solver.device
    env = get_env(cfg.rl_model.problem, generator_params={"num_loc": cfg.model.num_nodes})
    dataset = env.dataset(filename=cfg.data.test_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.eval.batch_size, 
        shuffle=False,
    )

    all_final_tours, all_final_costs, all_gt_costs = [], [], []
    all_stats = [] # New list to store statistics from all batches
    start_time = time.time()
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Solving Batches")):
        td = TensorDict(batch, batch_size=batch['locs'].shape[0]).to(device)
        td['locs'] = td['locs'].float()
        
        # MODIFIED: Capture the statistics from the solver
        solved_tours, batch_stats = solver.solve_batch_hybrid_vs_proposals(td, env)
        all_stats.extend(batch_stats)

        if cfg.eval.get("apply_two_opt", True):
            print("Applying 2-opt post-processing...")
            solved_tours = apply_2opt_batch(solved_tours, td['locs'])
            
        final_costs = calculate_tsp_cost_batch(td['locs'], solved_tours)
        
        gt_tour_indices = torch.arange(cfg.model.num_nodes, device=device).unsqueeze(0).repeat(td.shape[0], 1)
        gt_costs = calculate_tsp_cost_batch(td['locs'], gt_tour_indices)
        
        all_final_tours.append(solved_tours.cpu())
        all_final_costs.append(final_costs.cpu())
        all_gt_costs.append(gt_costs.cpu())

        # Visualization logic (remains the same)
        if batch_idx == 0 and cfg.eval.get("num_samples_to_visualize", 0) > 0:
            num_to_viz = min(td.shape[0], cfg.eval.num_samples_to_visualize)
            print(f"\nVisualizing final tours for first {num_to_viz} samples...")
            save_dir = "./hybrid_solver_visualizations_am_new"
            os.makedirs(save_dir, exist_ok=True)
            for i in range(num_to_viz):
                hybrid_cost = final_costs[i].item()
                gt_cost = gt_costs[i].item()
                title = (f"Instance {i} | Hybrid Cost: {hybrid_cost:.3f} | GT Cost: {gt_cost:.3f}\n"
                         f"Optimality Gap: {((hybrid_cost/gt_cost - 1)*100):.2f}%")
                fig, ax = plt.subplots(figsize=(8, 8))
                visualize_tsp_tour(
                    instance_locs=td['locs'][i].cpu(),
                    tour_indices=solved_tours[i].cpu(),
                    title=title, ax=ax,
                    gt_tour_indices=gt_tour_indices[i].cpu()
                )
                save_path = os.path.join(save_dir, f"hybrid_solve_instance_{i}.png")
                plt.savefig(save_path)
                plt.close(fig)

    total_time = time.time() - start_time
    final_costs_tensor = torch.cat(all_final_costs)
    gt_costs_tensor = torch.cat(all_gt_costs)

    avg_final_cost = final_costs_tensor.mean().item()
    avg_gt_cost = gt_costs_tensor.mean().item()
    optimality_gap = ((avg_final_cost / avg_gt_cost) - 1) * 100 if avg_gt_cost > 0 else float('inf')
    
    print("\n" + "=" * 60)
    print("--- Hybrid Step-by-Step Solver Evaluation Summary ---")
    print(f"Total time: {total_time:.2f}s")
    print(f"Evaluated {len(final_costs_tensor)} instances.")
    print(f"Average Hybrid Solver Cost: {avg_final_cost:.4f}")
    print(f"Average Ground Truth (Ordered) Cost: {avg_gt_cost:.4f}")
    print(f"Optimality Gap vs Ordered GT: {optimality_gap:.2f}%")
    print("=" * 60)

    # NEW: Print the detailed per-instance statistics
    print("\n" + "=" * 60)
    print("--- Detailed Best Tour Generation Analysis ---")
    for i, stats in enumerate(all_stats):
        print(f"\n--- Instance {i} ---")
        best_cost_val = stats['best_cost'].item()
        print(f"  > Final Best Cost: {best_cost_val:.4f}")
        print(f"  > Tour Source: {stats['source']}")
        if stats['source'] == 'DM Proposal':
            candidates = stats['candidates_for_the_step']
            prefix_node = stats['prefix_node']
            
            # 找到 prefix_node 在 candidates 中的位置
            indices = np.where(candidates == prefix_node)[0]  # 返回的是 array of indices

            print(f"  > Generated at step: {stats['generation_step']}")
            print(f"  > Winning Prefix Node: {stats['prefix_node']}, its index is {indices}")
            print(f"  > candidates_for_best at step: {stats['candidates_for_the_step']}")

            rl_choice = stats['rl_greedy_node_at_step']
            print(f"  > RL's Greedy Choice at that step: {rl_choice}")
            print(f"  > DM's choice differed from RL's? {'YES' if stats['prefix_node'] != rl_choice else 'No'}")
        elif stats['source'] == 'Hybrid Path':
            print("  > The step-by-step Hybrid Path was the best solution found.")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid RL-DM Step-by-Step Solver")
    parser.add_argument(
        "--config", type=str, default="hybrid_eval_config.yaml",
        help="Path to the unified YAML configuration file."
    )
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.config)
    
    solver_cfg = {
        'solver': {
            'num_candidates': 8,
            'dm_inference_steps': 5,
            'confidence_ratio_threshold': 0.001, 
            'dynamic_n_cumulative_threshold': 0.8,
            'apply_two_opt': False
        }
    }
    cfg = OmegaConf.merge(OmegaConf.create(solver_cfg), cfg)

    print("--- Running Hybrid Step-by-Step Solver with New Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------------------------------")
    run(cfg)
