time CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python engines/train_adapter/main_ours.py train_adapter --cfg configs/ours.yml --use_wandb 1
# time CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python engines/train_adapter/main_ours.py train_adapter --cfg configs/ours.yml --load_pretrained 1 --checkpoint output_checkpoints/?.pth --use_wandb 1


 