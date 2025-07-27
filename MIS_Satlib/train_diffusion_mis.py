# train_diffusion_mis.py (Corrected)

import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
import os
import time
from torch_geometric.loader import DataLoader as GraphDataLoader # Use the right DataLoader

from data_loader_mis import MISConditionalPrefixDataset
from diffusion_model_mis import ConditionalMISSuffixDiffusionModel
from discrete_diffusion_mis import NodeLabelDiffusion

def run_training_stage(cfg: DictConfig, stage_name: str, prefix_k_options: list, epochs_for_stage: int, checkpoint_to_load: str = None):
    print(f"\n===== Starting Curriculum Stage: {stage_name} =====")
    print(f"Epochs: {epochs_for_stage}, Prefix Fractions: {prefix_k_options}")
    if checkpoint_to_load:
        print(f"Loading checkpoint from: {checkpoint_to_load}")
    time.sleep(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = cfg.train.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    train_dataset = MISConditionalPrefixDataset(
        data_dir=cfg.data.train_path,
        prefix_k_options=prefix_k_options,
        prefix_sampling_strategy=cfg.data.prefix_sampling_strategy
    )
    # Use GraphDataLoader, no custom collate_fn needed
    train_dataloader = GraphDataLoader(
        train_dataset, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.train.num_workers
    )

    model = ConditionalMISSuffixDiffusionModel(
        node_feature_dim=cfg.model.node_feature_dim,
        node_embed_dim=cfg.model.node_embed_dim,
        gnn_n_layers=cfg.model.gnn_n_layers, gnn_hidden_dim=cfg.model.gnn_hidden_dim,
        prefix_enc_hidden_dim=cfg.model.prefix_enc_hidden_dim,
        prefix_cond_dim=cfg.model.prefix_cond_dim,
        time_embed_dim=cfg.model.time_embed_dim
    ).to(device)

    if checkpoint_to_load and os.path.exists(checkpoint_to_load):
        model.load_state_dict(torch.load(checkpoint_to_load, map_location=device))
        print(f"Successfully loaded checkpoint from {checkpoint_to_load}")

    diffusion_handler = NodeLabelDiffusion(
        num_timesteps=cfg.diffusion.num_timesteps,
        schedule_type=cfg.diffusion.schedule_type,
        device=device
    )

    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    
    for epoch in range(epochs_for_stage):
        model.train()
        total_train_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Stage '{stage_name}' Epoch {epoch+1}/{epochs_for_stage}")

        for batch_data in progress_bar:
            optimizer.zero_grad()
            batch_data = batch_data.to(device)
            
            # Timestep is per-graph, not per-node
            t = torch.randint(1, diffusion_handler.num_timesteps + 1, (batch_data.num_graphs,), device=device).long()
            
            loss = diffusion_handler.training_loss(model, batch_data, t)
            
            if torch.isnan(loss):
                print("Warning: NaN loss detected. Skipping batch.")
                continue

            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=f'{loss.item():.5f}')

        avg_epoch_loss = total_train_loss / len(train_dataloader)
        print(f"Stage '{stage_name}', Epoch {epoch+1} completed. Average Training Loss: {avg_epoch_loss:.5f}")

        if (epoch + 1) % cfg.train.save_interval == 0:
            save_path = os.path.join(ckpt_dir, f"{stage_name}_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved model checkpoint to {save_path}")

    final_ckpt_path = os.path.join(ckpt_dir, f"best_model_{stage_name}.pth")
    torch.save(model.state_dict(), final_ckpt_path)
    print(f"Finished stage '{stage_name}'. Saved final model to {final_ckpt_path}")
    return final_ckpt_path

def train_with_curriculum(cfg: DictConfig):
    global tqdm
    from tqdm.auto import tqdm
    last_checkpoint = None
    
    for stage_key in cfg.train.curriculum:
        stage_cfg = cfg.train.curriculum[stage_key]
        prefix_options_list = list(stage_cfg.prefix_k_options)
        
        last_checkpoint = run_training_stage(
            cfg=cfg, stage_name=stage_cfg.name,
            prefix_k_options=prefix_options_list,
            epochs_for_stage=stage_cfg.epochs,
            checkpoint_to_load=last_checkpoint
        )
    
    print("\nCurriculum training finished!")
    if last_checkpoint:
        final_path = os.path.join(cfg.train.ckpt_dir, "final_best_model.pth")
        os.replace(last_checkpoint, final_path)
        print(f"Renamed final model to: {final_path}")

if __name__ == "__main__":
    from tqdm.auto import tqdm
    config = OmegaConf.load("mis_config.yaml")
    train_with_curriculum(config)