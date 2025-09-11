#!/bin/bash
#SBATCH --gpus-per-node=v100l:1
#SBATCH --cpus-per-task=16       # CPU cores/threads
#SBATCH --mem=64000M             # memory per node
#SBATCH --time=10:00:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
wandb login $WANDB_API_KEY
module load apptainer
apptainer run --nv -C \
 -B /scratch/jocazar/:/home \
 -B /project/def-rmsouza/jocazar \
 -B /home/jocazar:/jocazar \
 /project/def-rmsouza/jocazar/CVPR_2024.sif \
 python /project/def-rmsouza/jocazar/ENSF_619_02_Final_project_jose_cazarin/main_image.py \
 --image_model=transformer_L16 \
 --dataset_folder_name=/jocazar/Final_dataset_W2025 \
 --ft_epochs=40 \
 --opt=adamw \
 --epochs=70 \
 --balance_weights \
 --reg=0.1 \
 --prob_aug=0.8 \