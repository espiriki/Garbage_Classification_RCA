#!/bin/bash
#SBATCH --gpus=h100:1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --time=23:59:59

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
wandb login $WANDB_API_KEY
module load apptainer
export TRANSFORMERS_NO_SAFE_TENSORS=1
apptainer run --nv \
 -B /etc/pki:/etc/pki \
 -B /project/def-rmsouza/jocazar \
 -B /scratch/jocazar/jocazar/:/scratch \
 /project/def-rmsouza/jocazar/cazarin_blip.sif \
  python /project/def-rmsouza/jocazar/ENSF_619_02_Final_project_jose_cazarin/blip_2_from_alexandre.py \
 --dataset_folder_name=/project/def-rmsouza/jocazar/ENSF_619_02_Final_project_jose_cazarin/Final_dataset_W2025_Train \
 --dataset_folder_name_val=/project/def-rmsouza/jocazar/ENSF_619_02_Final_project_jose_cazarin/Final_dataset_W2025_Val \
 --batch_size=32 \
 --epochs=16
