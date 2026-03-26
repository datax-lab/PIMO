#!/bin/bash
#SBATCH --job-name=PIMO
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpuq-a30
#SBATCH --nodelist=gpu[002]
#SBATCH --output=./SLURM/R-%SLURM.%j.out

source /home/parsas1/conda/bin/activate
conda activate pt_dcam
python3 -u data_preparation_pipeline.py --cancer_type="brca"