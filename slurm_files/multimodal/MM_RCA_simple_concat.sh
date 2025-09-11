#!/bin/bash
#SBATCH --gpus-per-node=v100l:1
#SBATCH --cpus-per-task=16       # CPU cores/threads
#SBATCH --mem=64000M             # memory per node
#SBATCH --time=16:00:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
wandb login $WANDB_API_KEY
module load apptainer
apptainer run --nv -C \
 -B /scratch/jocazar/:/home \
 -B /project/def-rmsouza/jocazar \
 -B /home/jocazar:/jocazar \
 /project/def-rmsouza/jocazar/CVPR_2024.sif \
 python /project/def-rmsouza/jocazar/ENSF_619_02_Final_project_jose_cazarin/main_both.py \
 --dataset_folder_name=/jocazar/Final_dataset_W2025 \
 --late_fusion=MM_RCA \
 --ft_epochs=40 \
 --epochs=30 \
 --prob_aug=1.00 \
 --acc_steps=10 \
 --acc_steps_FT=10 \
 --opt=sgd \
 --text_model=distilbert \
 --fraction_lr=3 \
 --image_text_dropout=0.0 \
 --balance_weights \
 --reg=0.03 \
 --lr=0.0016 \
 --features-only