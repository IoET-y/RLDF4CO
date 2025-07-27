# discrete_diffusion_mis.py (Corrected)
import torch
import torch.nn.functional as F
import numpy as np
import math
from tqdm import tqdm         # New, more standard and robust line

class NodeLabelDiffusion:
    # ... (The __init__ and q_sample methods are the same as before) ...
    def __init__(self, num_timesteps, schedule_type='cosine', device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device

        if schedule_type == 'cosine':
            s = 0.008
            t = torch.arange(num_timesteps + 1, dtype=torch.float64, device=device)
            alpha_bar = torch.cos((t / num_timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            self.betas = torch.clip(betas, 0, 0.999).float()
        else: # linear
            self.betas = torch.linspace(1e-4, 0.02, num_timesteps, device=device, dtype=torch.float32)

        self.alphas = 1.0 - self.betas
        
        # This part for q_sample on single tensors needs to be available
        self.Q_bar_t = torch.zeros(num_timesteps + 1, 2, 2, device=device, dtype=torch.float32)
        self.Q_bar_t[0] = torch.eye(2, device=device)
        self.Q_t = torch.zeros(num_timesteps, 2, 2, device=device, dtype=torch.float32)
        for t_idx in range(num_timesteps):
            alpha_t, beta_t = self.alphas[t_idx], self.betas[t_idx]
            self.Q_t[t_idx] = torch.tensor([[alpha_t, beta_t], [beta_t, alpha_t]], device=device)
            self.Q_bar_t[t_idx+1] = torch.matmul(self.Q_bar_t[t_idx], self.Q_t[t_idx])

    def q_sample(self, x_0_labels, t_steps):
        # x_0_labels is now a long tensor of nodes, t_steps is per-graph
        Q_bar_t_selected = self.Q_bar_t[t_steps] # (B, 2, 2)
        
        prob_xt_eq_1_given_x0_eq_1 = Q_bar_t_selected[:, 1, 1]
        prob_xt_eq_1_given_x0_eq_0 = Q_bar_t_selected[:, 0, 1]
        
        prob_xt_is_one = x_0_labels.float() * prob_xt_eq_1_given_x0_eq_1 + \
                         (1 - x_0_labels.float()) * prob_xt_eq_1_given_x0_eq_0
        
        return torch.bernoulli(prob_xt_is_one).long()

    def training_loss(self, denoiser_model, graph_batch, t):
        x_0_labels = graph_batch.x.squeeze(-1) # (total_nodes,)
        
        # Create a t_steps vector for all nodes in the batch
        t_steps_per_node = t[graph_batch.batch]
        
        # Sample noise for each node based on its graph's timestep
        x_t_labels = self.q_sample(x_0_labels, t_steps_per_node)
        
        # Model expects {-1, 1}
        x_t_transformed = x_t_labels.float() * 2.0 - 1.0
        x_t_transformed = x_t_transformed * (1.0 + 0.05 * torch.rand_like(x_t_transformed))
        
        # Update the batch with the noised features
        graph_batch.x = x_t_transformed.unsqueeze(-1)
        
        # Get model prediction
        predicted_x_0_logits = denoiser_model(graph_batch, t) # (total_nodes, 2)
        
        # Create loss mask: only compute loss on non-prefix nodes
        is_prefix_mask = (graph_batch.node_prefix_state == 1.0).squeeze(-1) # (total_nodes,)
        loss_mask = ~is_prefix_mask
        
        target = x_0_labels[loss_mask]
        prediction = predicted_x_0_logits[loss_mask]

        if target.numel() == 0:
            return torch.tensor(0.0, device=prediction.device, requires_grad=True)

        return F.cross_entropy(prediction, target.long())



   # ===================================================================
    # =========== NEW FUNCTIONS FOR DDIM INFERENCE ======================
    # ===================================================================

    def get_selection_schedule(self, num_inference_steps, schedule_type):
        """
        Generates a sequence of (t_current, t_prev) pairs for accelerated inference.
        """
        timesteps = []
        if schedule_type == 'linear':
            t_points = np.linspace(self.num_timesteps, 0, num_inference_steps + 1).astype(int)
        elif schedule_type == 'cosine':
            s_points = np.linspace(1.0, 0.0, num_inference_steps + 1)
            t_points = (0.5 * self.num_timesteps * (1 + np.cos(s_points * np.pi))).astype(int)
        else:
            raise ValueError(f"Unknown inference schedule: {schedule_type}")
        
        unique_t_points = np.unique(t_points)
        t_points = np.sort(unique_t_points)[::-1]
        
        for i in range(len(t_points) - 1):
            t_current = max(1, t_points[i])
            t_prev = t_points[i+1]
            timesteps.append((t_current, t_prev))
        return timesteps

    @torch.no_grad()
    def p_sample_loop_ddim(self, denoiser_model, graph_batch, num_inference_steps=50, schedule='cosine'):
        """
        Implements a DDIM-like deterministic sampling loop for the MIS model.
        """
        device = self.device
        
        # Start with random node labels {0, 1}
        x_t = torch.randint(0, 2, (graph_batch.num_nodes,), device=device, dtype=torch.long)
        
        timesteps_pairs = self.get_selection_schedule(num_inference_steps, schedule)

        for t_current, t_prev in tqdm(timesteps_pairs, desc='DDIM Denoising (MIS)'):
            t_current_tensor = torch.full((graph_batch.num_graphs,), t_current, device=device, dtype=torch.long)
            
            # 1. Get the model's prediction for x_0
            # Transform current state x_t to {-1, 1} for model input
            x_t_transformed = x_t.float() * 2.0 - 1.0
            graph_batch.x = x_t_transformed.unsqueeze(-1) # Update batch with current state
            
            pred_x0_logits = denoiser_model(graph_batch, t_current_tensor)
            # Convert logits to probabilities for each node being in the MIS
            pred_x0_probs = F.softmax(pred_x0_logits, dim=-1)[:, 1] # Probability of label 1

            # --- THE FIX IS HERE ---
            is_prefix_mask = (graph_batch.node_prefix_state == 1.0).squeeze(-1)
            # Get the true labels for ALL nodes, not just the prefix ones
            true_labels_all_nodes = graph_batch.x_true

            # 1. Start with the model's predictions
            pred_x0_probs_clamped = pred_x0_probs.clone()
            # 2. Use the mask to overwrite the predictions for prefix nodes with their true labels
            pred_x0_probs_clamped[is_prefix_mask] = true_labels_all_nodes[is_prefix_mask].float()
            # --- END OF FIX ---
            
            if t_prev == 0:
                # If this is the last step, the prediction is our final answer
                x_t = torch.bernoulli(pred_x0_probs_clamped).long()
                continue

            # 3. Apply the core Discrete DDIM formula
            Q_bar_t_prev = self.Q_bar_t[t_prev]
            
            prob_xtm1_eq_1_given_x0_eq_1 = Q_bar_t_prev[1, 1]
            prob_xtm1_eq_1_given_x0_eq_0 = Q_bar_t_prev[0, 1]

            # Calculate probability of x_{t-1} being 1, based on our prediction of x_0
            probs_xt_minus_1_is_1 = (
                prob_xtm1_eq_1_given_x0_eq_1 * pred_x0_probs_clamped +
                prob_xtm1_eq_1_given_x0_eq_0 * (1 - pred_x0_probs_clamped)
            )
            
            # 4. Sample the next state x_{t-1}
            x_t = torch.bernoulli(probs_xt_minus_1_is_1).long()
            
        # Return the final predicted labels and probabilities
        return x_t, pred_x0_probs