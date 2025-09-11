#!/bin/bash
#SBATCH --gpus-per-node=v100l:1
#SBATCH --cpus-per-task=8       # CPU cores/threads
#SBATCH --mem=64000M             # memory per node
#SBATCH --time=06:00:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
wandb login $WANDB_API_KEY
module load apptainer
apptainer run --nv -C \
 -B /scratch/jocazar/:/home \
 -B /project/def-rmsouza/jocazar \
 /project/def-rmsouza/jocazar/CVPR_2024.sif \
 python /project/def-rmsouza/jocazar/ENSF_619_02_Final_project_jose_cazarin/main_image.py \
 --image_model=mb \
 --dataset_folder_name=CVPR_2024_dataset \
 --ft_epochs=70 \
 --opt=adamw \
 --epochs=70 \
 --balance_weights \
 --reg=0.1