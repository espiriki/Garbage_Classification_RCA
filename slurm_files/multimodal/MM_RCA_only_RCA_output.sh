#!/bin/bash
#SBATCH --gpus=h100:1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --time=6:00:00
#SBATCH --mem=64000M

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
wandb login $WANDB_API_KEY
module load apptainer/1.3.5
apptainer run --nv --bind /etc/pki:/etc/pki \
 -B /scratch/jocazar/:/home \
 -B /project/def-rmsouza/jocazar \
 -B /home/jocazar:/jocazar \
 /project/def-rmsouza/jocazar/Final_2025.sif \
 python /project/def-rmsouza/jocazar/ENSF_619_02_Final_project_jose_cazarin/main_both.py \
 --dataset_folder_name=/project/def-rmsouza/jocazar/ENSF_619_02_Final_project_jose_cazarin/Final_dataset_W2025 \
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
 --reverse \
 --cross_attention_only