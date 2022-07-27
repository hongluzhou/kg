time CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python engines/train_adapter/main_baseline_bag_of_steps.py train_adapter --cfg configs/baseline_bag_of_steps.yml --use_wandb 1
# time CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python engines/train_adapter/main_baseline_bag_of_steps.py train_adapter --cfg configs/baseline_bag_of_steps.yml --load_pretrained 1 --checkpoint output_checkpoints/baseline_kg-no-edges-bag-of-steps/Adapter-2022-07-13T05-32-16Z_e134.pth --use_wandb 1


 