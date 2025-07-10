# RLDF4CO_v4/train_diffusion_new_2gpu_new.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import os
import time

from data_loader_new import TSPConditionalSuffixDataset, custom_collate_fn
from diffusion_model_new import ConditionalTSPSuffixDiffusionModel
from discrete_diffusion_new_new_new import AdjacencyMatrixDiffusion
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast

@torch.no_grad()
def validate_model(model, diffusion_handler, valid_dataloader, device):
    model.eval()
    total_valid_loss = 0
    num_batches = 0

    for batch_data in valid_dataloader:
        instance_locs = batch_data["instance_locs"].to(device)
        prefix_nodes = batch_data["prefix_nodes"].to(device)
        prefix_lengths = batch_data["prefix_lengths"].to(device)
        x_0_adj_matrix = batch_data["target_adj_matrix"].to(device)
        node_prefix_state = batch_data["node_prefix_state"].to(device) # <<< GET NEW STATE

        t = torch.randint(1, diffusion_handler.num_timesteps + 1, (instance_locs.size(0),), device=device).long()

        loss = diffusion_handler.training_loss(
            model, x_0_adj_matrix, t, instance_locs,
            prefix_nodes, prefix_lengths, node_prefix_state # <<< PASS NEW STATE
        )
        total_valid_loss += loss.item()
        num_batches += 1
    
    if num_batches == 0:
        return float('inf')

    # 将损失和批次数聚合到所有进程
    total_loss_tensor = torch.tensor([total_valid_loss, num_batches], dtype=torch.float64, device=device)
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    
    global_total_loss, global_num_batches = total_loss_tensor[0].item(), total_loss_tensor[1].item()
    
    avg_valid_loss = global_total_loss / global_num_batches if global_num_batches > 0 else float('inf')
    return avg_valid_loss


def ddp_setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank), local_rank


def run_training_stage(cfg: DictConfig, stage_name: str, prefix_k_options: list, epochs_for_stage: int, device, local_rank, checkpoint_to_load: str = None):
    """
    Executes a single stage of the training curriculum.
    """
    
    if dist.get_rank() == 0:
        print(f"\n===== Starting Curriculum Stage: {stage_name} =====")
        print(f"===== Epochs: {epochs_for_stage}, Prefix K Range: {prefix_k_options[0]}-{prefix_k_options[-1]} =====")
        print(f"prefix in this stage is {prefix_k_options}")
        if checkpoint_to_load:
            print(f"===== Loading checkpoint from: {checkpoint_to_load} =====")


    time.sleep(2) # Pause for readability

    ckpt_dir = cfg.train.get("ckpt_dir", "./ckpt_difusco_style")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    prefix_sampling_strategy = cfg.data.get('prefix_sampling_strategy', 'continuous_from_start')

    #global_batch_size = cfg.train.batch_size
    #per_gpu_batch_size = global_batch_size // dist.get_world_size()  # 分给每个进程

    # Setup Datasets for the current stage
    train_dataset = TSPConditionalSuffixDataset(
        npz_file_path=cfg.data.train_path,
        prefix_k_options=prefix_k_options,
        prefix_sampling_strategy=prefix_sampling_strategy
    )
    train_sampler = DistributedSampler(train_dataset)

    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.train.batch_size, shuffle=False, 
        sampler=train_sampler,
        num_workers=cfg.train.get("num_workers", 4), collate_fn=custom_collate_fn
    )

    valid_dataloader = None
    if cfg.data.get("valid_path"):
        
        valid_dataset = TSPConditionalSuffixDataset(
            npz_file_path=cfg.data.valid_path,
            prefix_k_options=prefix_k_options,
            prefix_sampling_strategy=prefix_sampling_strategy
        )


        valid_sampler = DistributedSampler(valid_dataset, shuffle=False) # shuffle=False 用于验证
        valid_dataloader = DataLoader(
            valid_dataset, 
            batch_size=cfg.train.batch_size,
            sampler=valid_sampler,shuffle=False, # 使用 Sampler
            num_workers=cfg.train.get("num_workers", 4), 
            collate_fn=custom_collate_fn
        )
    # Initialize Model
    model = ConditionalTSPSuffixDiffusionModel(
        num_nodes=cfg.model.num_nodes, node_coord_dim=cfg.model.node_coord_dim,
        pos_embed_num_feats=cfg.model.pos_embed_num_feats, node_embed_dim=cfg.model.node_embed_dim,
        prefix_node_embed_dim=cfg.model.node_embed_dim,
        prefix_enc_hidden_dim=cfg.model.prefix_enc_hidden_dim, prefix_cond_dim=cfg.model.prefix_cond_dim,
        gnn_n_layers=cfg.model.gnn_n_layers, gnn_hidden_dim=cfg.model.gnn_hidden_dim,
        gnn_aggregation=cfg.model.gnn_aggregation, gnn_norm=cfg.model.gnn_norm,
        gnn_learn_norm=cfg.model.gnn_learn_norm, gnn_gated=cfg.model.gnn_gated,
        time_embed_dim=cfg.model.time_embed_dim
    ).to(device)

    # 2. 加载你的单卡训练检查点 (它的 key 也没有 'module.' 前缀)
    #    因为两者都没有 'module.' 前缀，所以这里可以完美匹配！
    if checkpoint_to_load:
        model_checkpoint_path = checkpoint_to_load
        if os.path.exists(model_checkpoint_path):
            try:
                # 加载 state_dict 到原始 model
                model.load_state_dict(torch.load(model_checkpoint_path, map_location=device, weights_only=True))
                if dist.get_rank() == 0:
                    print(f"Successfully loaded single-GPU checkpoint into base model from {model_checkpoint_path}")
            except Exception as e:
                if dist.get_rank() == 0:
                    print(f"Could not load checkpoint: {e}. Starting from scratch.")
        else:
            if dist.get_rank() == 0:
                 print(f"Checkpoint file not found at {model_checkpoint_path}. Starting from scratch.")
    
    # (可选但推荐) 使用同步屏障确保所有进程都完成了加载
    dist.barrier()  # 确保所有进程都完成了加载

    
    # 3. 最后，将已经载入权重的模型用 DDP 包装
    #    DDP 会自动为所有 key 加上 'module.' 前缀，用于后续的梯度同步
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)  
    

    diffusion_handler = AdjacencyMatrixDiffusion(
        num_nodes=cfg.model.num_nodes, num_timesteps=cfg.diffusion.num_timesteps,
        schedule_type=cfg.diffusion.schedule_type, device=device
    )

    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    scaler = GradScaler()# 20250626
    
    best_valid_loss = float('inf')
    epochs_no_improve = 0
    early_stopping_patience = cfg.train.get("early_stopping_patience", 10)
    min_delta = cfg.train.get("early_stopping_min_delta", 0.00001)
    
    # Main training loop for the stage
    for epoch in range(epochs_for_stage):
        train_sampler.set_epoch(epoch)
        model.train()
        total_train_loss = 0
        num_train_batches = 0
        is_main_process = (dist.get_rank() == 0)
        
        for batch_idx, batch_data in enumerate(train_dataloader):
            optimizer.zero_grad()
            instance_locs = batch_data["instance_locs"].to(device)
            prefix_nodes = batch_data["prefix_nodes"].to(device)
            prefix_lengths = batch_data["prefix_lengths"].to(device)
            x_0_adj_matrix = batch_data["target_adj_matrix"].to(device)
            node_prefix_state = batch_data["node_prefix_state"].to(device) # <<< GET NEW STATE

            t = torch.randint(1, diffusion_handler.num_timesteps + 1, (instance_locs.size(0),), device=device).long()

            # loss = diffusion_handler.training_loss(
            #     model, x_0_adj_matrix, t, instance_locs,
            #     prefix_nodes, prefix_lengths, node_prefix_state # <<< PASS NEW STATE
            # )
            # loss.backward()
            # optimizer.step()

            with autocast():
                loss = diffusion_handler.training_loss(
                    model, x_0_adj_matrix, t, instance_locs,
                    prefix_nodes, prefix_lengths, node_prefix_state
                )
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            
            total_train_loss += loss.item()
            num_train_batches += 1

            if is_main_process and batch_idx % cfg.train.log_interval == 0 and batch_idx > 0:
                print(f"Stage '{stage_name}', Epoch {epoch+1}/{epochs_for_stage}, Batch {batch_idx}/{len(train_dataloader)}, Avg Train Loss: {(total_train_loss/num_train_batches):.5f}")
        
        print(f"Stage '{stage_name}', Epoch {epoch+1} completed. Average Training Loss: {(total_train_loss/num_train_batches):.5f}")

        if valid_dataloader:
            current_valid_loss = validate_model(model, diffusion_handler, valid_dataloader, device)
            print(f"Stage '{stage_name}', Epoch {epoch+1}: Validation Loss: {current_valid_loss:.5f}")
            if is_main_process:

                if current_valid_loss < best_valid_loss - min_delta:
                    best_valid_loss = current_valid_loss
                    epochs_no_improve = 0
                    best_model_path_stage = os.path.join(ckpt_dir, f"best_model_{stage_name}.pth")
                    torch.save(model.module.state_dict(), best_model_path_stage)
                    print(f"Validation loss improved. Saved best model for this stage to {best_model_path_stage}")
                else:
                    epochs_no_improve += 1
    
                if epochs_no_improve >= early_stopping_patience:
                    print(f"Early stopping triggered for stage '{stage_name}'.")
                    break
        if is_main_process:
            if (epoch + 1) % cfg.train.save_interval == 0:
                periodic_save_path = os.path.join(ckpt_dir, f"{stage_name}_epoch_{epoch+1}.pth")
                torch.save(model.module.state_dict(), periodic_save_path)
                
                print(f"Saved model checkpoint (periodic) at epoch {epoch+1} to {periodic_save_path}")
    print(f"Finished stage '{stage_name}'. Best validation loss for this stage: {best_valid_loss:.5f}")
    return os.path.join(ckpt_dir, f"best_model_{stage_name}.pth")

def ddp_cleanup():
    dist.destroy_process_group()

    
def train_with_curriculum(cfg: DictConfig):
    """
    Main function to orchestrate the curriculum learning process.
    """    
    device, local_rank = ddp_setup()
    print(f"[Rank {dist.get_rank()}] DDP setup complete. Using device: {device}")

    try:
        # Stage 1: Easy task - long prefixes
        stage1_k_options = list(range(60, 91))
        stage1_epochs = 10
        stage1_best_ckpt = run_training_stage(
            cfg=cfg,
            stage_name="stage1_k60_90",
            prefix_k_options=stage1_k_options,
            epochs_for_stage=stage1_epochs,
            device=device,              # <<< 传递 device
            local_rank=local_rank,      # <<< 传递 local_rank
            checkpoint_to_load=None
        )
        #stage1_best_ckpt = "./ckpt_tsp_difusco_style_new_prefix_new_new/stage1_k60_90_epoch_30.pth"
        # Stage 2: Medium task - short prefixes
        stage2_k_options = list(range(1, 61))
        stage2_epochs = 10
        stage2_best_ckpt = run_training_stage(
            cfg=cfg,
            stage_name="stage2_k1_60",
            prefix_k_options=stage2_k_options,
            epochs_for_stage=stage2_epochs,
            device=device,              # <<< 传递 device
            local_rank=local_rank,      # <<< 传递 local_rank
            checkpoint_to_load=stage1_best_ckpt
        )

        # Stage 3: Full task - all prefixes
        stage3_k_options = list(range(1, cfg.model.num_nodes))
        stage3_epochs = 20
        final_best_ckpt = run_training_stage(
            cfg=cfg,
            stage_name="stage3_k1_99_final",
            prefix_k_options=stage3_k_options,
            epochs_for_stage=stage3_epochs,
            device=device,              # <<< 传递 device
            local_rank=local_rank,      # <<< 传递 local_rank
            checkpoint_to_load=stage2_best_ckpt
        )

        # Stage 4: Front task - [1-30]
        stage4_k_options = list(range(1, 30))
        stage4_epochs = 20
        final_last_best_ckpt = run_training_stage(
            cfg=cfg,
            stage_name="stage4_k1_30_last",
            prefix_k_options=stage4_k_options,
            epochs_for_stage=stage4_epochs,
            device=device,              # <<< 传递 device
            local_rank=local_rank,      # <<< 传递 local_rank
            checkpoint_to_load=final_best_ckpt
        )

        stage5_k_options = list(range(1, 20))
        stage5_epochs = 20
        final_last_best_ckpt = run_training_stage(
            cfg=cfg,
            stage_name="stage5_k1_20_last",
            prefix_k_options=stage5_k_options,
            epochs_for_stage=stage5_epochs,
            device=device,              # <<< 传递 device
            local_rank=local_rank,      # <<< 传递 local_rank
            checkpoint_to_load=final_last_best_ckpt
        )

        if dist.get_rank() == 0:
            print("\nCurriculum training finished!")
            final_generic_path = os.path.join(os.path.dirname(final_last_best_ckpt), "Stage5_1_20_best_model_checkpoint.pth")
            if os.path.exists(final_last_best_ckpt):
                os.rename(final_last_best_ckpt, final_generic_path)
                print(f"Renamed final model to: {final_generic_path}")

    # finally:
    #     # === 新增：在程序结束时清理DDP进程组 ===
    #     ddp_cleanup()
    #     print(f"[Rank {dist.get_rank()}] DDP resources cleaned up.")
    finally:
        # It's important to get the rank *before* cleaning up the process group.
        # Once ddp_cleanup() is called, `dist.get_rank()` will fail.
        # We also check if the process group is initialized to be safe.
        if dist.is_initialized():
            rank = dist.get_rank()
            ddp_cleanup()
            print(f"[Rank {rank}] DDP resources cleaned up.")
        else:
            # This branch would execute if setup failed in the first place
            print("DDP was not initialized, no cleanup needed.")

if __name__ == "__main__":
    config_path = "tsp100_config.yaml" 
    try:
        config = OmegaConf.load(config_path)
        print("Loaded configuration from:", config_path)
    except FileNotFoundError:
        print(f"ERROR: Configuration file '{config_path}' not found.")
        exit()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        exit()
        
    train_with_curriculum(config)
