python tsp_gendata.py --mode generate --num_nodes 200 --num_samples 25000 --seed 7991
python tsp_gendata.py --mode solve --solver ortools --num_nodes 100 --num_samples 10000 --seed 1997

python tsp_gendata.py --mode solve --solver concorde --num_nodes 50 --num_samples 15000 --seed 7991 > solve_n50_training.log 2>&1 &
python tsp_gendata.py --mode solve --solver concorde --num_nodes 50 --num_samples 10000 --seed 7991  > solve_n50_test.log 2>&1 &

## old version v3 
python train_diffusion.py > output_train_df_v5.log 2>&1 &

python evaluate_diffusion_GPU_v3

python hybrid_solver_am_new_v2.py --config hybrid_eval_config.yaml     > output_evaluationFULL.log 2>&1 &
python hybrid_solver_am_new.py --config hybrid_eval_config.yaml   
python hybrid_solver_pomo.py --config hybrid_eval_config_pomo.yaml
python hybrid_new_attempt_v4.py --config hybrid_eval_config.yaml

## new version v4
python train_diffusion_new.py > output_train_newdf_v7.log 2>&1 &
python evaluate_diffusion_GPU_v4

python hybrid_solver_am_new.py --config hybrid_eval_config.yaml   
python hybrid_solver_pomo.py --config hybrid_eval_config_pomo.yaml


torchrun --nproc_per_node=2 --master_port=29515 train_diffusion_new_2gpu_new_old_fintune.py  > output_train_newdf_v9_2gpu_fintune.log 2>&1 &