time CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python engines/main/main_baseline_load-frozen-adapter_train-task-head.py train_task_head --cfg configs/baseline_load_fronzen_adapter.yml --checkpoint output_checkpoints/baseline_kg-no-edges-bag-of-steps/Adapter-2022-07-15T05-20-20Z_e116.pth --use_wandb 1