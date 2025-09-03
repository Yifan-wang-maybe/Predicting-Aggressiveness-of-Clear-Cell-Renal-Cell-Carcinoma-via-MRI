1. Image Proprecess: Volume Resample and Volume Normalization

2. Pre-training Foundation model: 

Training:
torchrun --nproc_per_node=1 main.py --model mae_vit_tiny_tiny_patch16 --output_dir XXX/output_dir_tiny_tiny_augment --log_dir XXX/Kindey/Pre_train/V1/output_dir_tiny_tiny_augment

Refer to Job_tiny_tiny.sh for detailed training setting.

3. Cross-validation finetuning

torchrun --nproc_per_node=1 main_finetune.py mae_vit_tiny_tiny_patch16 --output_dir XXX --log_dir XXX

Notes:
   Change XXX to the according path in all codes

