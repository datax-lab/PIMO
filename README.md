# PIMO: Pathway-based Interpretable Multi-Omics interactions for multi-omics integration

## Abstract
We propose a Pathway-based Interpretable deep learning Multi-Omics interaction model, PIMO, that explicitly captures regulatory effects across omics layers. Experiments on multiple TCGA cancer datasets showed that PIMO consistently outperformed state-of-the-art baselines in survival analysis, up to 13% increase in the C-index. PIMO provides biologically interpretable analyses that identify important pathways, genes, and inter-omics interactions with DNA methylation and copy number alterations

## Overview
![abstract_figure_jpg](https://github.com/user-attachments/assets/83199a87-bbed-4e88-bbaf-57621f8c2051)

## Data Preparation
The `tcga_data_preparation/` directory contains the following subfolders:

- `raw_brca/` — raw input data  
- `processed_brca/` — processed multi-omics data  
- `splits_brca/` — train/validation/test splits  
- `kegg_pathway_data/` — KEGG pathway and gene annotations  

In addition, this directory includes:
- `tcga_data_preparation.ipynb` — main notebook for data preprocessing  
- `data_preparation_pipeline.py` — pipeline for dataset construction and splitting  
- helper scripts for running preprocessing  

### Required Inputs
Users must download the following data from cBioPortal for Cancer Genomics:

- Gene expression  
- DNA methylation  
- Copy number alteration (CNA)  
- Clinical data  

These files should be placed in:
- `raw_brca/` (for BRCA), or  
- `raw_<cancer>/` (for other cancer types)

The `kegg_pathway_data/` folder (already provided) contains pathway definitions and gene mappings required for model input construction.

### Processing Steps
Running `tcga_data_preparation.ipynb` will:

1. Integrate multi-omics data into a unified representation  
2. Generate merged datasets stored in:
   - `processed_brca/` or `processed_<cancer>/`  
3. Apply preprocessing and pathway-based transformations  

The `data_preparation_pipeline.py` script further:

- Splits the data into training, validation, and test sets  
- Applies pathway ordering (PathCNN-style preprocessing)  
- Saves outputs to:
  - `splits_brca/` or `splits_<cancer>/`

---

## Model Training
The root directory contains:

- `run.py` — main script for training the PIMO model  
- `run.sh` — SLURM batch script for cluster execution  
- `run_command.txt` — example command for submitting jobs  

### Running the Model (SLURM)

```bash
sbatch --export=gpu_num=0,cancer_type="brca",exp_num=1,learning_rate=0.00005,decay_rate=0.3,decay_epochs=100,weight_decay=0.9,num_epochs=500,num_kernels1=32,dropout1=0.58,num_kernels2=4,dropout2=0.025,activation_fn="tanh",fc_nodes=256,batch_size=256,min_epochs=200,patience=25 run.sh
