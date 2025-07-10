# evalutaion_GPU_v2.py
import torch
import numpy as np
import yaml
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt # For visualization
import os
# 假设你的这些模块和训练时使用的是一样的
from Backup_v2.data_loader_v2 import TSPConditionalSuffixDataset # construct_target_adj_from_prefix_suffix 可能不需要在评估中直接用，但 data_loader 可能需要
from Backup_v2.data_loader_v2 import construct_target_adj_from_prefix_suffix # construct_target_adj_from_prefix_suffix 可能不需要在评估中直接用，但 data_loader 可能需要

import time # Added for timing
from Backup_v2.model_components_v2 import ( SinusoidalTimestepEmbedding, # Keep this for explicit time embedding if preferred outside GNN
    PrefixEncoder,
    PositionEmbeddingSine,      # For node coordinate embedding
    DifuscoGNNEncoder           # The new GNN backbone
)

from Backup_v2.diffusion_model_v2 import ConditionalTSPSuffixDiffusionModel 
#from diffusion_model import ConditionalTSPSuffixDiffusionModel 

from Backup_v2.model_components_v2 import SinusoidalTimestepEmbedding, PrefixEncoder, PositionEmbeddingSine, DifuscoGNNEncoder, DifuscoGNNLayer, normalization, linear, timestep_embedding 
from Backup_v2.discrete_diffusion_v2 import AdjacencyMatrixDiffusion, InferenceSchedule
# -------- Helper Functions for Evaluation (Modified for Batching and GPU) --------

def calculate_segment_cost_gpu(node_idx1_batch, node_idx2_batch, instance_locs_batch):
    """
    Helper function to calculate cost between two specific nodes for a batch.
    node_idx1_batch: (B) tensor of first node indices.
    node_idx2_batch: (B) tensor of second node indices.
    instance_locs_batch: (B, N, 2) tensor of city coordinates.
    """
    B = instance_locs_batch.shape[0]
    # Create indices for gathering: (B, 1, 2) for node coordinates
    idx1 = node_idx1_batch.unsqueeze(-1).unsqueeze(-1).expand(B, 1, 2)
    idx2 = node_idx2_batch.unsqueeze(-1).unsqueeze(-1).expand(B, 1, 2)

    loc1_batch = torch.gather(instance_locs_batch, 1, idx1).squeeze(1) # (B, 2)
    loc2_batch = torch.gather(instance_locs_batch, 1, idx2).squeeze(1) # (B, 2)
    return torch.sqrt(((loc1_batch - loc2_batch)**2).sum(dim=1)) # (B)


def calculate_tsp_cost_batch(instance_locs_batch, tour_indices_batch):
    """
    Calculates the total length of TSP tours for a batch.
    instance_locs_batch: (B, N, 2) tensor of city coordinates.
    tour_indices_batch: (B, N_tour) tensor of city indices in tour order.
                       N_tour can be <= N if tours are partial or not full.
    """
    B, N, _ = instance_locs_batch.shape
    N_tour = tour_indices_batch.shape[1]

    if N_tour == 0:
        return torch.zeros(B, device=instance_locs_batch.device)
    if N_tour < 2: # A tour with less than 2 nodes has zero cost
        return torch.zeros(B, device=instance_locs_batch.device)

    # Gather tour locations: (B, N_tour, 2)
    # tour_indices_batch needs to be (B, N_tour, 1) then expanded for gather
    tour_indices_expanded = tour_indices_batch.unsqueeze(-1).expand(B, N_tour, 2)
    tour_locs_batch = torch.gather(instance_locs_batch, 1, tour_indices_expanded)

    # Calculate segment lengths: (B, N_tour-1)
    segment_lengths = torch.sqrt(((tour_locs_batch[:, :-1] - tour_locs_batch[:, 1:])**2).sum(dim=2))

    # Calculate closing segment lengths: (B)
    closing_segment_diff_sq = ((tour_locs_batch[:, -1] - tour_locs_batch[:, 0])**2) # (B, 2)
    closing_segment_lengths = torch.sqrt(closing_segment_diff_sq.sum(dim=1)) # (B)

    total_costs = segment_lengths.sum(dim=1) + closing_segment_lengths
    return total_costs # (B)

def apply_2opt_batch(initial_tours_batch, instance_locs_batch, max_iterations=100):
    """
    Applies the 2-opt local search algorithm to improve a batch of TSP tours on GPU.
    Optimized to vectorize tour reversal.

    initial_tours_batch: (B, N_nodes) tensor of node indices in the initial tour order.
    instance_locs_batch: (B, N_nodes, 2) tensor of city coordinates.
    max_iterations: Maximum number of iterations.

    Returns: (B, N_nodes) tensor of node indices in the optimized tour order.
    """
    device = instance_locs_batch.device
    B, num_nodes, _ = instance_locs_batch.shape

    if num_nodes < 4: # 2-opt requires at least 4 nodes
        return initial_tours_batch

    current_tours_tensor = initial_tours_batch.clone().long()
    best_tours_tensor = current_tours_tensor.clone()
    
    best_costs = calculate_tsp_cost_batch(instance_locs_batch, best_tours_tensor)

    for iter_count in range(max_iterations):
        improved_in_pass = torch.zeros(B, dtype=torch.bool, device=device) # Tracks if any tour improved in this pass
        
        for i in range(num_nodes - 2):
            for j in range(i + 2, num_nodes):
                # Current edges for all tours in the batch
                node_i_indices = current_tours_tensor[:, i]
                node_i_plus_1_indices = current_tours_tensor[:, i+1]
                node_j_indices = current_tours_tensor[:, j]
                node_j_plus_1_indices = current_tours_tensor[:, (j + 1) % num_nodes]

                # Cost of current two edges for the batch (GPU calculation)
                cost_edge_i_ip1 = calculate_segment_cost_gpu(node_i_indices, node_i_plus_1_indices, instance_locs_batch)
                cost_edge_j_jp1 = calculate_segment_cost_gpu(node_j_indices, node_j_plus_1_indices, instance_locs_batch)
                current_edge_pair_costs = cost_edge_i_ip1 + cost_edge_j_jp1 # (B)

                # Cost of new two edges if swapped (GPU calculation)
                cost_edge_i_j = calculate_segment_cost_gpu(node_i_indices, node_j_indices, instance_locs_batch)
                cost_edge_ip1_jp1 = calculate_segment_cost_gpu(node_i_plus_1_indices, node_j_plus_1_indices, instance_locs_batch)
                new_edge_pair_costs = cost_edge_i_j + cost_edge_ip1_jp1 # (B)

                # Mask for tours where swapping these two edges is beneficial (edge cost heuristic)
                edge_improvement_mask = new_edge_pair_costs < current_edge_pair_costs # (B)
                
                if edge_improvement_mask.any():
                    # Clone current tours to create a temporary version with potential swaps
                    temp_swapped_tours = current_tours_tensor.clone()

                    # Identify the actual indices in the batch that show edge improvement
                    indices_for_swap = torch.where(edge_improvement_mask)[0]
                    
                    if len(indices_for_swap) > 0:
                        # Extract the subset of tours that need modification
                        tours_to_modify_subset = current_tours_tensor[indices_for_swap]

                        # Perform vectorized 2-opt swap (reverse segment) on the GPU
                        # Segment is from index i+1 to j (inclusive)
                        idx_segment_start = i + 1
                        idx_segment_end = j 

                        prefix = tours_to_modify_subset[:, :idx_segment_start]
                        segment_to_flip = tours_to_modify_subset[:, idx_segment_start : idx_segment_end + 1]
                        suffix = tours_to_modify_subset[:, idx_segment_end + 1 :]
                        
                        flipped_segment = torch.flip(segment_to_flip, dims=[1])
                        
                        # Concatenate parts to form the new tours for the subset
                        modified_tours_subset = torch.cat([prefix, flipped_segment, suffix], dim=1)
                        
                        # Place the modified subset back into the temporary full batch tensor
                        temp_swapped_tours[indices_for_swap] = modified_tours_subset
                    
                    # Recalculate full tour costs for tours in temp_swapped_tours
                    # This confirms if the edge swap leads to an overall tour improvement.
                    new_total_costs = calculate_tsp_cost_batch(instance_locs_batch, temp_swapped_tours)
                    
                    # Mask for tours where the new total cost is better than the current best known cost
                    total_cost_improvement_mask = new_total_costs < best_costs
                    
                    # Final update mask: must satisfy both the edge heuristic AND overall cost reduction
                    final_update_mask = edge_improvement_mask & total_cost_improvement_mask
                                        
                    if final_update_mask.any():
                        best_tours_tensor[final_update_mask] = temp_swapped_tours[final_update_mask]
                        best_costs[final_update_mask] = new_total_costs[final_update_mask]
                        improved_in_pass[final_update_mask] = True # Mark that these tours improved in this pass
                        
                        # CRITICAL: Update current_tours_tensor for the next i,j iteration within this pass
                        # with the tours that have shown improvement.
                        current_tours_tensor[final_update_mask] = temp_swapped_tours[final_update_mask]

        if not improved_in_pass.any(): # If no tour in the batch was improved in this entire pass over i,j
            # print(f"  2-opt converged in {iter_count+1} iterations for this batch.")
            break # Exit max_iterations loop early
            
    # print(f"  2-opt finished after {iter_count+1} iterations for this batch.")
    return best_tours_tensor

def apply_2opt_batch_bk(initial_tours_batch, instance_locs_batch, max_iterations=100):
    """
    Applies the 2-opt local search algorithm to improve a batch of TSP tours on GPU.

    initial_tours_batch: (B, N_nodes) tensor of node indices in the initial tour order.
    instance_locs_batch: (B, N_nodes, 2) tensor of city coordinates.
    max_iterations: Maximum number of iterations.

    Returns: (B, N_nodes) tensor of node indices in the optimized tour order.
    """
    device = instance_locs_batch.device
    B, num_nodes, _ = instance_locs_batch.shape

    if num_nodes < 4: # 2-opt requires at least 4 nodes
        return initial_tours_batch

    current_tours_tensor = initial_tours_batch.clone().long()
    best_tours_tensor = current_tours_tensor.clone()
    
    # Calculate initial costs for the batch
    best_costs = calculate_tsp_cost_batch(instance_locs_batch, best_tours_tensor)

    for iter_count in range(max_iterations):
        improved_batch = torch.zeros(B, dtype=torch.bool, device=device)
        
        for i in range(num_nodes - 2):
            for j in range(i + 2, num_nodes):
                # Current edges: (node_i, node_i_plus_1) and (node_j, node_j_plus_1)
                node_i_indices = current_tours_tensor[:, i]
                node_i_plus_1_indices = current_tours_tensor[:, i+1]
                node_j_indices = current_tours_tensor[:, j]
                node_j_plus_1_indices = current_tours_tensor[:, (j + 1) % num_nodes] # Handle loop

                # Cost of current two edges for the batch
                cost_edge_i_ip1 = calculate_segment_cost_gpu(node_i_indices, node_i_plus_1_indices, instance_locs_batch)
                cost_edge_j_jp1 = calculate_segment_cost_gpu(node_j_indices, node_j_plus_1_indices, instance_locs_batch)
                current_edge_costs = cost_edge_i_ip1 + cost_edge_j_jp1 # (B)

                # Cost of new two edges if swapped: (node_i, node_j) and (node_i_plus_1, node_j_plus_1)
                cost_edge_i_j = calculate_segment_cost_gpu(node_i_indices, node_j_indices, instance_locs_batch)
                cost_edge_ip1_jp1 = calculate_segment_cost_gpu(node_i_plus_1_indices, node_j_plus_1_indices, instance_locs_batch)
                new_edge_costs = cost_edge_i_j + cost_edge_ip1_jp1 # (B)

                # Identify improvements for the batch
                improvement_mask = new_edge_costs < current_edge_costs # (B)
                
                if improvement_mask.any():
                    # Create new tours for those that improve
                    temp_new_tours = current_tours_tensor.clone()
                    
                    # Perform 2-opt swap (reverse segment) only for tours that improve
                    # This is tricky to vectorize perfectly without advanced indexing or loops.
                    # For simplicity, we can iterate here or use more complex tensor ops.
                    # Let's try a loop for clarity first, then consider vectorization.
                    for k in range(B):
                        if improvement_mask[k]:
                            tour_to_modify = temp_new_tours[k].tolist() # Convert to list for easy slicing/reversal
                            segment_to_reverse = tour_to_modify[i+1 : j+1]
                            segment_to_reverse.reverse()
                            temp_new_tours[k] = torch.tensor(
                                tour_to_modify[:i+1] + segment_to_reverse + tour_to_modify[j+1:],
                                device=device, dtype=torch.long
                            )
                    
                    # Recalculate costs for the modified tours
                    current_new_tours_costs = calculate_tsp_cost_batch(instance_locs_batch, temp_new_tours)
                    
                    # Update tours and costs where improvement actually lowered total cost
                    # (Edge cost delta is a heuristic, full cost check is more robust)
                    update_mask = current_new_tours_costs < best_costs 
                    final_update_mask = improvement_mask & update_mask # Ensure both edge heuristic and total cost improve

                    if final_update_mask.any():
                        best_tours_tensor[final_update_mask] = temp_new_tours[final_update_mask]
                        best_costs[final_update_mask] = current_new_tours_costs[final_update_mask]
                        improved_batch[final_update_mask] = True
                        current_tours_tensor[final_update_mask] = temp_new_tours[final_update_mask]


        if not improved_batch.any(): # If no tour in the batch improved
            break
            
    # print(f"2-opt batch finished in {iter_count+1} iterations.")
    return best_tours_tensor


# def decode_adj_matrices_to_tours_batch(adj_matrices_probs, batch_prefix_nodes, num_nodes):
#     """
#     Decodes a batch of probabilistic adjacency matrices into TSP tours on GPU.
#     adj_matrices_probs: (B, N, N) tensor of edge probabilities (e.g., after sigmoid).
#     batch_prefix_nodes: (B, k) tensor of fixed prefix node indices. k can be 0.
#     num_nodes: Total number of nodes (N).

#     Returns: A (B, N) tensor of node indices representing the tours.
#              Returns (B, N) tensor of -1 for failed decodings.
#     """
#     B, N, _ = adj_matrices_probs.shape
#     k = batch_prefix_nodes.shape[1]
#     device = adj_matrices_probs.device

#     # Initialize tours: (B, N), filled with -1 (or a placeholder for not set)
#     tours = torch.full((B, N), -1, dtype=torch.long, device=device)
    
#     # Visited mask: (B, N)
#     visited_mask = torch.zeros((B, N), dtype=torch.bool, device=device)
    
#     # Set prefix nodes
#     if k > 0:
#         tours[:, :k] = batch_prefix_nodes
#         # Mark prefix nodes as visited
#         # Need to use scatter_ along dim 1 for visited_mask
#         # Create indices for scatter: (B, k)
#         # Create src for scatter: True values (B, k)
#         visited_mask.scatter_(1, batch_prefix_nodes, True)

#     # current_nodes are the last nodes in the (partially built) tours
#     # If k=0, we need a starting node policy. Let's default to 0.
#     current_nodes = torch.zeros(B, dtype=torch.long, device=device)
#     if k > 0 :
#         current_nodes = batch_prefix_nodes[:, -1] # Last node of prefix
#     else: # No prefix, start all tours at node 0
#         if N > 0:
#             tours[:, 0] = 0
#             visited_mask[:, 0] = True
#             current_nodes[:] = 0 # current_nodes is already zeros(B)
#         else: # num_nodes is 0
#             return tours # Returns (B,0) tensor of -1s, or handle as error

#     num_filled_per_tour = torch.full((B,), k if k > 0 else (1 if N > 0 else 0) , dtype=torch.long, device=device)
    
#     # Active mask for batches that are not yet complete
#     active_batches = torch.ones(B, dtype=torch.bool, device=device)
#     if N == 0 : active_batches[:] = False


#     for step in range(k if k > 0 else (1 if N > 0 else 0), N): # Iterate to fill up to N nodes
#         if not active_batches.any(): # All tours in batch are complete
#             break

#         # For active batches, find the next node
#         # Gather probabilities from current_nodes: (B_active, N)
#         # Need to index adj_matrices_probs for active batches and their current_nodes
        
#         active_indices = torch.where(active_batches)[0]
#         if len(active_indices) == 0: break

#         current_adj_probs = adj_matrices_probs[active_indices] # (B_active, N, N)
#         current_step_nodes = current_nodes[active_indices]     # (B_active)
        
#         # Get probabilities from current_node to all others (and vice-versa for symmetry)
#         # Probs from current: current_adj_probs[batch_idx, current_node_val, next_node_candidate]
#         # Probs to current:   current_adj_probs[batch_idx, next_node_candidate, current_node_val]
        
#         # Use advanced indexing to get probabilities for each active batch from its current node
#         # B_active = len(active_indices)
#         # batch_arange = torch.arange(B_active, device=device)
#         # probs_from = current_adj_probs[batch_arange, current_step_nodes, :] # (B_active, N)
#         # probs_to   = current_adj_probs[batch_arange, :, current_step_nodes] # (B_active, N)
#         # symmetrized_probs = (probs_from + probs_to) / 2.0

#         # Alternative way to gather, perhaps simpler for this specific case:
#         symmetrized_probs_list = []
#         for i_active, original_batch_idx in enumerate(active_indices):
#             node_val = current_nodes[original_batch_idx].item()
#             p_from = adj_matrices_probs[original_batch_idx, node_val, :]
#             p_to = adj_matrices_probs[original_batch_idx, :, node_val]
#             symmetrized_probs_list.append((p_from + p_to) / 2.0)
#         if not symmetrized_probs_list: break # Should not happen if active_indices is not empty
#         symmetrized_probs = torch.stack(symmetrized_probs_list) # (B_active, N)


#         # Apply visited mask (for active batches)
#         # Set probabilities of already visited nodes to a very low value
#         symmetrized_probs[visited_mask[active_indices]] = -float('inf')

#         # Select best next node for each active tour
#         best_next_probs, best_next_nodes = torch.max(symmetrized_probs, dim=1) # (B_active)
        
#         # Update tours, visited_mask, current_nodes, and num_filled for active batches
#         # Check for failures (no valid next node, prob is -inf)
#         valid_next_node_mask_active = best_next_probs > -float('inf')
        
#         # Update only if a valid next node was found
#         active_indices_with_valid_next = active_indices[valid_next_node_mask_active]
#         nodes_to_add = best_next_nodes[valid_next_node_mask_active]
        
#         if len(active_indices_with_valid_next) > 0:
#             # Add to tour: tours[batch_idx, step_idx] = node_to_add
#             # We need the correct step_idx for each tour in active_indices_with_valid_next
#             current_fill_counts = num_filled_per_tour[active_indices_with_valid_next]
#             tours[active_indices_with_valid_next, current_fill_counts] = nodes_to_add
            
#             # Update visited_mask
#             # visited_mask[batch_idx, node_to_add] = True
#             visited_mask.scatter_(1, nodes_to_add.unsqueeze(1), True) # More robust for non-contiguous indices

#             # Update current_nodes
#             current_nodes[active_indices_with_valid_next] = nodes_to_add
            
#             # Increment num_filled
#             num_filled_per_tour[active_indices_with_valid_next] += 1

#         # Update active_batches: a tour becomes inactive if it's full or failed
#         failed_construction_mask_active = ~valid_next_node_mask_active
#         active_batches[active_indices[failed_construction_mask_active]] = False # Mark failed as inactive
#         active_batches[active_indices_with_valid_next[num_filled_per_tour[active_indices_with_valid_next] == N]] = False # Mark completed as inactive

#     # Check for tours that didn't complete to N nodes (failures)
#     # They will have -1 in some positions.
#     # For simplicity, we return as is. Caller can check for -1.
#     return tours

import torch
from collections import defaultdict

def construct_tour_from_edges(edge_list, num_nodes, start_node=0):
    """
    Given a list of edges representing a valid tour, construct the node sequence.
    """
    if not edge_list or len(edge_list) < num_nodes -1:
        return []
    
    adj = defaultdict(list)
    for u, v in edge_list:
        adj[u].append(v)
        adj[v].append(u)
        
    # Find a starting node, preferably one from the prefix if available
    if start_node not in adj:
        # Fallback if start_node is isolated
        start_node = next(iter(adj)) if adj else 0

    tour = [start_node]
    prev_node = -1
    curr_node = start_node
    
    # Using a set for faster checking of visited nodes
    visited_nodes = {start_node}
    
    while len(tour) < num_nodes:
        neighbors = adj.get(curr_node, [])
        next_node_found = False
        for neighbor in neighbors:
            if neighbor != prev_node:
                next_node = neighbor
                next_node_found = True
                break
        
        if not next_node_found or next_node in visited_nodes:
             # This indicates a problem, like a sub-tour or dead end.
             return [] 
            
        tour.append(next_node)
        visited_nodes.add(next_node)
        prev_node = curr_node
        curr_node = next_node
        
    return tour


def decode_adj_matrices_to_tours_batch(adj_matrices_probs, instance_locs, batch_prefix_nodes):
    """
    FINAL & CORRECTED VERSION: Decodes heatmaps using the edge-based greedy strategy from DIFUSCO,
    while rigorously enforcing the prefix constraint required by the hybrid solver.
    This replaces the old node-based greedy decoder.
    """
    B, N, _ = adj_matrices_probs.shape
    device = adj_matrices_probs.device
    
    # Symmetrize the probability matrix and calculate edge scores
    adj_probs = (adj_matrices_probs + adj_matrices_probs.transpose(1, 2)) / 2.0
    dists = torch.cdist(instance_locs, instance_locs, p=2) + 1e-6
    edge_scores = adj_probs / dists
    
    # Flatten scores and sort all possible edges (upper triangle)
    indices = torch.triu_indices(N, N, offset=1, device=device)
    flat_scores = edge_scores[:, indices[0], indices[1]]
    sorted_scores, sorted_indices = torch.sort(flat_scores, dim=1, descending=True)
    
    # Get the edge coordinates (u, v) for all sorted edges
    sorted_edges_u = indices[0][sorted_indices]
    sorted_edges_v = indices[1][sorted_indices]

    final_tours = torch.full((B, N), -1, dtype=torch.long, device=device)

    # --- Batch-wise Greedy Construction ---
    for i in range(B):
        # Union-Find data structure for cycle detection
        parent = torch.arange(N, device=device)
        def find_set(v):
            if v == parent[v]: return v
            parent[v] = find_set(parent[v])
            return parent[v]
        def unite_sets(a, b):
            a, b = find_set(a), find_set(b)
            if a != b: parent[b] = a

        node_degrees = torch.zeros(N, dtype=torch.int, device=device)
        edges_in_tour = []
        is_prefix_edge = torch.zeros(N, N, dtype=torch.bool, device=device)
        
        # === 1. ENFORCE PREFIX CONSTRAINT ===
        prefix_nodes = batch_prefix_nodes[i]
        # Handle potential padding in prefix_nodes if it comes from a collate_fn
        prefix_len = (prefix_nodes != -1).sum().item() 
        prefix_nodes = prefix_nodes[:prefix_len]
        
        if prefix_len > 1:
            for j in range(prefix_len - 1):
                u, v = prefix_nodes[j].item(), prefix_nodes[j+1].item()
                # Ensure u < v for consistency with is_prefix_edge matrix
                if u > v: u, v = v, u
                
                edges_in_tour.append((u, v))
                node_degrees[u] += 1
                node_degrees[v] += 1
                unite_sets(u, v)
                is_prefix_edge[u, v] = True
        # ====================================

        # === 2. GREEDY EDGE INSERTION for remaining edges ===
        num_edges_to_add = N - len(edges_in_tour)
        edges_added_count = 0

        for u_tensor, v_tensor in zip(sorted_edges_u[i], sorted_edges_v[i]):
            u, v = u_tensor.item(), v_tensor.item()
            
            # Skip if it's already a prefix edge
            if is_prefix_edge[u, v]:
                continue
            
            # Check conditions: no degree > 2 and no cycles
            if node_degrees[u] < 2 and node_degrees[v] < 2 and find_set(u) != find_set(v):
                edges_in_tour.append((u, v))
                node_degrees[u] += 1
                node_degrees[v] += 1
                unite_sets(u, v)
                edges_added_count += 1
                if edges_added_count == num_edges_to_add:
                    break
        # =======================================================
        
        # === 3. FINALIZE AND CONSTRUCT TOUR ===
        # The logic from cython_merge to close the tour if N-1 edges are found is implicitly handled
        # by the greedy search. If N-1 edges are added, the last two degree-1 nodes form the last available valid edge.

        if len(edges_in_tour) == N:
            start_node = prefix_nodes[0].item() if prefix_len > 0 else 0
            tour_sequence = construct_tour_from_edges(edges_in_tour, N, start_node=start_node)
            if tour_sequence and len(tour_sequence) == N:
                final_tours[i] = torch.tensor(tour_sequence, device=device)
        # =====================================
                
    return final_tours



def visualize_tsp_tour(instance_locs, tour_indices, title="TSP Tour", ax=None, gt_tour_indices=None):
    """
    Visualizes a TSP tour. (Assumed to be mostly CPU-based for plotting)
    instance_locs: (N, 2) tensor of city coordinates (CPU).
    tour_indices: (N) tensor or list of city indices in tour order (CPU).
    gt_tour_indices: (Optional N) tensor or list of ground truth tour for comparison (CPU).
    """
    if ax is None:
        fig, ax = plt.subplots()

    if isinstance(instance_locs, torch.Tensor):
        instance_locs = instance_locs.cpu()
    if isinstance(tour_indices, torch.Tensor):
        tour_indices = tour_indices.cpu()
    if isinstance(gt_tour_indices, torch.Tensor):
        gt_tour_indices = gt_tour_indices.cpu()
        
    if isinstance(tour_indices, list):
        tour_indices = torch.tensor(tour_indices)
    
    # Filter out -1s from incomplete tours for visualization
    valid_tour_indices = tour_indices[tour_indices != -1]
    if len(valid_tour_indices) == 0:
        print(f"Warning: No valid tour to visualize for '{title}'")
        ax.scatter(instance_locs[:, 0], instance_locs[:, 1], color='blue', s=50, zorder=2, label="Cities (No tour)")
        ax.set_title(title + " (No Valid Tour)")
        return

    valid_tour_indices = valid_tour_indices.long()
    
    ax.scatter(instance_locs[:, 0], instance_locs[:, 1], color='blue', s=50, zorder=2, label="Cities")
    for i in range(instance_locs.size(0)):
        ax.text(instance_locs[i, 0], instance_locs[i, 1], str(i), fontsize=8, zorder=3)

    tour_locs = instance_locs[valid_tour_indices]
    for i in range(len(tour_locs)):
        start_node = tour_locs[i]
        end_node = tour_locs[(i + 1) % len(tour_locs)] 
        ax.plot([start_node[0], end_node[0]], [start_node[1], end_node[1]], 'r-', lw=1.5, zorder=1, label="Generated Tour" if i == 0 else None)
    if len(tour_locs) > 0:
        ax.scatter(tour_locs[0,0], tour_locs[0,1], color='red', s=100, marker='x', zorder=4, label="Start/End")

    if gt_tour_indices is not None:
        if isinstance(gt_tour_indices, list):
            gt_tour_indices = torch.tensor(gt_tour_indices)
        gt_tour_indices = gt_tour_indices.long()
        gt_tour_locs = instance_locs[gt_tour_indices]
        for i in range(len(gt_tour_locs)):
            start_node = gt_tour_locs[i]
            end_node = gt_tour_locs[(i + 1) % len(gt_tour_locs)]
            ax.plot([start_node[0], end_node[0]], [start_node[1], end_node[1]], 'g--', lw=1, zorder=0.5, label="Ground Truth Tour" if i == 0 else None)

    ax.set_title(title)
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.legend()
    ax.axis('equal')

# -------- Main Evaluation Function (Modified for Batching) --------
@torch.no_grad()
def evaluate(cfg: DictConfig, model_checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ... (data loading setup as before) ...
    all_data = np.load(cfg.data.test_path) # Make sure cfg.data.test_path is in your config
    test_instances_locs_np = all_data['locs']
    num_total_test_samples, num_nodes_conf, _ = test_instances_locs_np.shape
    
    if num_nodes_conf != cfg.model.num_nodes:
        raise ValueError(f"Mismatch in num_nodes: data has {num_nodes_conf}, config expects {cfg.model.num_nodes}")

    test_instances_locs_tensor = torch.tensor(test_instances_locs_np, dtype=torch.float32)
    test_gt_tours_indices_tensor = torch.arange(cfg.model.num_nodes, dtype=torch.long).unsqueeze(0).repeat(num_total_test_samples, 1)
    
    rl_prefix_k = cfg.data.prefix_k
    test_prefix_nodes_tensor = test_gt_tours_indices_tensor[:, :rl_prefix_k]

    eval_batch_size = cfg.eval.get("batch_size", 1) 
    num_samples_to_evaluate = min(cfg.eval.get("num_samples_to_eval", num_total_test_samples), num_total_test_samples)
    
    dataset = torch.utils.data.TensorDataset(
        test_instances_locs_tensor[:num_samples_to_evaluate],
        test_gt_tours_indices_tensor[:num_samples_to_evaluate],
        test_prefix_nodes_tensor[:num_samples_to_evaluate]
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=eval_batch_size, shuffle=False)


    # 2. Load trained Diffusion Model (Updated Instantiation)
    model = ConditionalTSPSuffixDiffusionModel(
        num_nodes=cfg.model.num_nodes,
        node_coord_dim=cfg.model.node_coord_dim,
        pos_embed_num_feats=cfg.model.pos_embed_num_feats,
        node_embed_dim=cfg.model.node_embed_dim,
        prefix_max_len=cfg.data.prefix_k,
        prefix_node_embed_dim=cfg.model.node_embed_dim,
        prefix_enc_hidden_dim=cfg.model.prefix_enc_hidden_dim,
        prefix_cond_dim=cfg.model.prefix_cond_dim,
        gnn_n_layers=cfg.model.gnn_n_layers,
        gnn_hidden_dim=cfg.model.gnn_hidden_dim,
        gnn_aggregation=cfg.model.gnn_aggregation,
        gnn_norm=cfg.model.gnn_norm,
        gnn_learn_norm=cfg.model.gnn_learn_norm,
        gnn_gated=cfg.model.gnn_gated,
        time_embed_dim=cfg.model.time_embed_dim
    ).to(device)
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded trained model from {model_checkpoint_path}")

    # 3. Diffusion Process Handler
    diffusion_handler = AdjacencyMatrixDiffusion(
        num_nodes=cfg.model.num_nodes, # Pass num_nodes
        num_timesteps=cfg.diffusion.num_timesteps,
        schedule_type=cfg.diffusion.schedule_type, # Ensure this matches training if Q_bar_t depends on it
        device=device
    )

    # ... (rest of the evaluation loop: batching, timing, cost calculation) ...
    total_generated_cost_sum = 0.0
    total_gt_cost_sum = 0.0
    num_valid_tours_generated = 0
    batch_times = []
    total_samples_processed = 0
    overall_start_time = time.time()


    for batch_idx, (batch_locs, batch_gt_tours, batch_prefixes) in enumerate(dataloader):
        batch_start_time = time.time()
        current_batch_size = batch_locs.shape[0]
        instance_locs_batch = batch_locs.to(device)
        gt_tour_indices_batch = batch_gt_tours.to(device)
        prefix_nodes_batch = batch_prefixes.to(device)
        
        print(f"\nEvaluating batch {batch_idx+1}/{len(dataloader)}, size {current_batch_size}...")

        sample_shape = (current_batch_size, cfg.model.num_nodes, cfg.model.num_nodes)

        print(f"  Generating suffixes with diffusion model (prefix_k={rl_prefix_k})...")
        # Updated p_sample_loop call
        generated_adj_matrices_output = diffusion_handler.p_sample_loop(
            denoiser_model=model,
            shape=sample_shape,
            instance_locs=instance_locs_batch,
            prefix_nodes=prefix_nodes_batch,
            num_inference_steps=cfg.eval.num_inference_steps,
            inference_schedule_type=cfg.eval.inference_schedule_type
        ) # Output: (B, N, N) binary

        # The output of p_sample_loop is now binary {0,1} float matrix.
        # If your decoder expects probabilities, you might need to adjust.
        # However, a common approach is to use the model's x0_pred from the *final* step
        # as a heatmap if the decoder requires probabilities.
        # For now, let's assume generated_adj_matrices_output (the final binary sample x_0) is used.
        # If your decoder needs probabilities, you'd ideally get pred_x0_probs from the last step of p_sample_loop.
        
        # Let's assume generated_adj_matrices_output is the final binary sampled x0.
        # Your decode_adj_matrices_to_tours_batch expects probabilities.
        # To get probabilities, we'd need to run the model one last time on the sampled x_1 (if t_prev was 1)
        # or use the pred_x0_probs from the t_prev=0 step inside p_sample_loop.
        # For simplicity with the current p_sample_loop, let's treat its binary output as "hard" probabilities for the decoder.
        generated_adj_matrices_probs = generated_adj_matrices_output.float()


        print(f"  Decoding adjacency matrices to tours (batch)...")
        initial_decoded_tours_batch = decode_adj_matrices_to_tours_batch( # Your existing greedy decoder
            generated_adj_matrices_probs, 
            prefix_nodes_batch,          
            cfg.model.num_nodes
        ) 

        # ... (rest of 2-opt, cost calculation, visualization remains similar) ...
        # ... (Make sure to import `time` if you haven't in evalutaion_GPU.py)
        valid_initial_mask = (initial_decoded_tours_batch != -1).all(dim=1)
        optimized_tours_final_batch = initial_decoded_tours_batch.clone()

        if cfg.eval.get("apply_two_opt", True) and valid_initial_mask.any() :
            print(f"  Applying 2-opt to {valid_initial_mask.sum().item()} initially valid tours...")
            tours_for_2opt = initial_decoded_tours_batch[valid_initial_mask]
            locs_for_2opt = instance_locs_batch[valid_initial_mask]
            
            if tours_for_2opt.numel() > 0: 
                optimized_tours_from_2opt = apply_2opt_batch(
                    tours_for_2opt,
                    locs_for_2opt,
                    max_iterations=cfg.eval.get("two_opt_max_iterations", 100)
                )
                optimized_tours_final_batch[valid_initial_mask] = optimized_tours_from_2opt
        
        fully_decoded_mask = (optimized_tours_final_batch != -1).all(dim=1)
        costs_generated_batch = torch.full((current_batch_size,), float('inf'), device=device)
        costs_gt_batch = torch.full((current_batch_size,), float('inf'), device=device)

        if fully_decoded_mask.any():
            valid_tours = optimized_tours_final_batch[fully_decoded_mask]
            valid_locs = instance_locs_batch[fully_decoded_mask]
            valid_gt_tours = gt_tour_indices_batch[fully_decoded_mask]

            costs_generated_batch[fully_decoded_mask] = calculate_tsp_cost_batch(valid_locs, valid_tours)
            costs_gt_batch[fully_decoded_mask] = calculate_tsp_cost_batch(valid_locs, valid_gt_tours)

            total_generated_cost_sum += costs_generated_batch[fully_decoded_mask].sum().item()
            total_gt_cost_sum += costs_gt_batch[fully_decoded_mask].sum().item()
            num_valid_tours_generated += fully_decoded_mask.sum().item()

        batch_elapsed_time = time.time() - batch_start_time
        batch_times.append(batch_elapsed_time)
        total_samples_processed += current_batch_size
        
        # Visualization (select first few from the batch, if needed)
        # (ensure os and matplotlib.pyplot are imported)
        for vis_idx in range(min(current_batch_size, cfg.eval.get("num_samples_to_visualize_in_first_batches",1) if batch_idx == 0 else 0)):
            actual_sample_idx_in_dataset = batch_idx * eval_batch_size + vis_idx
            if actual_sample_idx_in_dataset < cfg.eval.get("num_samples_to_visualize", 5) :
                tour_to_viz = optimized_tours_final_batch[vis_idx]
                cost_GT_viz = costs_gt_batch[vis_idx].item() if costs_gt_batch[vis_idx] != float('inf') else -1
                cost_gen_viz = costs_generated_batch[vis_idx].item() if costs_generated_batch[vis_idx] != float('inf') else -1
                title = f"Instance {actual_sample_idx_in_dataset+1} (k={rl_prefix_k}) - Gen Cost: {cost_gen_viz:.2f}, GT Cost: {cost_GT_viz:.2f}"
                if not fully_decoded_mask[vis_idx].item():
                    title += " (Decode Failed)"
                
                fig, ax = plt.subplots(figsize=(8,8))
                # Make sure visualize_tsp_tour can handle CPU tensors
                visualize_tsp_tour( 
                    instance_locs_batch[vis_idx].cpu(), 
                    tour_to_viz.cpu(),
                    title=title,
                    ax=ax,
                    gt_tour_indices=gt_tour_indices_batch[vis_idx].cpu()
                )
                # print(f"DEBUG: GT tour for visualization (Instance {actual_sample_idx_in_dataset+1}):")
                # print(gt_tour_indices_batch[vis_idx])
                if cfg.eval.get("apply_two_opt", True):
                    placefig = "2-opt"
                else:
                    placefig = "greedy"
                os.makedirs("./generated_tsp_batch_difusco", exist_ok=True) # New viz folder
                plt.savefig(f"./generated_tsp_batch_difusco/tsp_tour_instance_{actual_sample_idx_in_dataset+1}_k{rl_prefix_k}_{placefig}.png")
                plt.close(fig)
                print(f"  Saved visualization for instance {actual_sample_idx_in_dataset+1}")
    
    overall_elapsed_time = time.time() - overall_start_time
    # ... (timing summary and evaluation summary as before) ...
    print("\n--------- Timing Summary ---------")
    if batch_times:
        avg_batch_time = sum(batch_times) / len(batch_times)
        avg_sample_time = sum(batch_times) / total_samples_processed if total_samples_processed > 0 else 0
        print(f"Processed {total_samples_processed} samples in {len(batch_times)} batches.")
        print(f"Total evaluation time: {overall_elapsed_time:.3f}s")
        print(f"Average time per batch: {avg_batch_time:.3f}s")
        print(f"Average time per sample: {avg_sample_time:.3f}s (Batch size: {eval_batch_size})")
        print(f"Samples per second: {total_samples_processed / overall_elapsed_time:.2f} samples/s")


    avg_generated_cost = total_generated_cost_sum / num_valid_tours_generated if num_valid_tours_generated > 0 else float('inf')
    avg_gt_cost = total_gt_cost_sum / num_valid_tours_generated if num_valid_tours_generated > 0 else float('inf')

    print("\n---------Diffusion Model Evaluation Summary ---------")
    print(f"Number of instances evaluated: {total_samples_processed}") 
    print(f"Number of valid tours successfully decoded and costed: {num_valid_tours_generated}")
    
    if num_valid_tours_generated > 0:
        print(f"Average Generated Tour Cost (for valid tours): {avg_generated_cost:.4f}")
        print(f"Average Ground Truth Tour Cost (for corresponding valid tours): {avg_gt_cost:.4f}")
        optimality_gap = ((avg_generated_cost / avg_gt_cost) - 1) * 100 if avg_gt_cost > 0 else float('inf')
        print(f"Optimality Gap: {optimality_gap:.2f}%")
    else:
        print("No valid tours were successfully generated and costed.")


if __name__ == "__main__":

    example_eval_config_dict = {
        'data': {
            'test_path': "./tsp_data_n100/tsp_solutions_test_n100_s1997_solver_concorde_100.npz" ,  #./tsp/tsp100-greedy-test-new-seed1234.npz
            'prefix_k': 1, 
        },
        'model': { 
            'num_nodes': 100, 'node_coord_dim': 2,
            'pos_embed_num_feats': 64, 'node_embed_dim': 128,
            'prefix_enc_hidden_dim': 128, 'prefix_cond_dim': 128,
            'gnn_n_layers': 12, 'gnn_hidden_dim': 256,
            'gnn_aggregation': "sum", 'gnn_norm': "layer",
            'gnn_learn_norm': True, 'gnn_gated': True,
            'time_embed_dim': 256
        },
        'diffusion': { 
            'num_timesteps': 1000, 
            'schedule_type': 'cosine',
        },
        'eval': { # Ensure these are in your loaded config
            'batch_size': 32, 
            'num_samples_to_eval': 100,
            'num_samples_to_visualize': 5,
            'num_samples_to_visualize_in_first_batches': 6,
            'num_inference_steps': 20, 
            'inference_schedule_type': 'cosine',
            'apply_two_opt': False,
            'two_opt_max_iterations': 80
        }
    }
    eval_cfg = OmegaConf.create(example_eval_config_dict)
    # --- End placeholder config ---

    trained_model_checkpoint =  "./ckpt_tsp_difusco_style_new/best_model_checkpoint.pth"

    # (Dummy data creation for testing structure, if needed)
    if not os.path.exists(eval_cfg.data.test_path):
        print(f"Test data file {eval_cfg.data.test_path} not found. Creating a dummy file.")
        num_samples = eval_cfg.eval.num_samples_to_eval 
        num_nodes = eval_cfg.model.num_nodes
        dummy_locs = np.random.rand(num_samples, num_nodes, 2)
        os.makedirs(os.path.dirname(eval_cfg.data.test_path), exist_ok=True)
        np.savez_compressed(eval_cfg.data.test_path, locs=dummy_locs)

    if not os.path.exists(trained_model_checkpoint):
         print(f"Model checkpoint {trained_model_checkpoint} not found. Evaluation may fail.")

    evaluate(eval_cfg, trained_model_checkpoint)