#!/bin/bash
#SBATCH --gpus-per-node=v100l:1
#SBATCH --cpus-per-task=32       # CPU cores/threads
#SBATCH --mem=64000M             # memory per node
#SBATCH --time=22:00:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
wandb login $WANDB_API_KEY
module load apptainer
export TRANSFORMERS_NO_SAFE_TENSORS=1
apptainer run --nv \
 -B /etc/pki:/etc/pki \
 -B /project/def-rmsouza/jocazar \
 -B /scratch/jocazar/jocazar/:/scratch \
 /project/def-rmsouza/jocazar/cazarin_blip.sif \
  python /project/def-rmsouza/jocazar/ENSF_619_02_Final_project_jose_cazarin/q_former_alexandre.py \
 --dataset_folder_name=/project/def-rmsouza/jocazar/ENSF_619_02_Final_project_jose_cazarin/Final_dataset_W2025_Train \
 --dataset_folder_name_val=/project/def-rmsouza/jocazar/ENSF_619_02_Final_project_jose_cazarin/Final_dataset_W2025_Val \
 --batch_size=4 \
 --epochs=14
