#!/bin/bash
#SBATCH --job-name=PIMO
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=sxmq
#SBATCH --nodelist=sxm[002]
#SBATCH --output=./SLURM/R-%SLURM.%j.out

source /home/parsas1/conda/bin/activate
conda activate pt_dcam

python3 -u run.py --gpu_num=$gpu_num --cancer_type=$cancer_type --exp_num=$exp_num --learning_rate=$learning_rate --decay_rate=$decay_rate --decay_epochs=$decay_epochs --weight_decay=$weight_decay --num_epochs=$num_epochs --num_kernels1=$num_kernels1 --dropout1=$dropout1 --num_kernels2=$num_kernels2 --dropout2=$dropout2 --activation_fn=$activation_fn --fc_nodes=$fc_nodes --batch_size=$batch_size --min_epochs=$min_epochs --patience=$patience