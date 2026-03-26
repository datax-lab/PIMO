import argparse

parser = argparse.ArgumentParser(description="PIMO Training")
parser.add_argument("--gpu_num", type=int, default=0)
parser.add_argument("--cancer_type", type=str, default="brca")
parser.add_argument("--exp_num", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--decay_rate", type=float, default=0.5)
parser.add_argument("--decay_epochs", type=int, default=50)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--num_epochs", type=int, default=300)
parser.add_argument("--num_kernels1", type=int, default=16)
parser.add_argument("--dropout1", type=float, default=0.58)
parser.add_argument("--num_kernels2", type=int, default=8)
parser.add_argument("--dropout2", type=float, default=0.1)
parser.add_argument("--activation_fn", type=str, default='tanh')
parser.add_argument("--fc_nodes", type=int, default=256)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--min_epochs", type=int, default=200)
parser.add_argument("--patience", type=int, default=25)
parser.add_argument("--dataset_path", type=str, default='tcga_data_preparation')
parser.add_argument("--chkpoint_path", type=str, default='chkpoints')
parser.add_argument("--results_path", type=str, default='results')
args = parser.parse_args()
print(args)

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)

DATA_PATH = f'{args.dataset_path}/splits_{args.cancer_type}'
PATHWAY_MASK_PATH = f'{args.dataset_path}/processed_{args.cancer_type}'
CHKPOINT_PATH = args.chkpoint_path
RESULTS_PATH  = args.results_path 

import copy
import torch
import pickle
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from scipy.interpolate import interp1d

import time
import psutil
from typing import Tuple
from typing import Any
from typing import Sequence
from typing import Optional

import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from Model import PIMO
from DataLoader import load_data, load_pathway, MultiOmicsDataset, sort_collate_fn
from Survival_CostFunc_CIndex import R_set, neg_par_log_likelihood, c_index, CIndex, AUC


def exponential_lr_decay(
    learning_rate: float,
    optimizer: Optimizer,
    lr_decay_rate: float,
    step: int
) -> None:
    """
    Applies exponential decay to the learning rate and updates the optimizer in-place.

    The learning rate is scaled according to:
        lr_new = learning_rate * (lr_decay_rate ** step)

    This function directly modifies the learning rate of all parameter groups
    within the given optimizer.

    Args:
        learning_rate (float): Initial/base learning rate.
        optimizer (Optimizer): PyTorch optimizer whose learning rate will be updated.
        lr_decay_rate (float): Multiplicative decay factor applied at each step (e.g., 0.5).
        step (int): Current decay step, typically derived from epoch progression
                    (e.g., step = epoch // decay_epochs).

    Returns:
        None
    """

    new_lr = learning_rate * (lr_decay_rate ** step)

    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def load_data(
    exp_num: int,
    split_type: str,
    cancer_type: str,
    data_path: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads multi-omics data (RNA, DNA, CNA) along with survival outcomes.

    This function reads preprocessed NumPy arrays for each omics modality
    and a CSV file containing survival information. The survival data is
    split into time-to-event and event indicator arrays.

    Args:
        exp_num (int): Experiment number corresponding to a specific data split.
        split_type (str): Dataset split identifier (e.g., "train", "val", "test").
        cancer_type (str): Cancer type identifier (e.g., "brca").
        data_path (str): Root directory containing dataset files.

    Returns:
        Tuple containing:
            - rna (np.ndarray): gene expression data.
            - dna (np.ndarray): DNA methylation data.
            - cna (np.ndarray): Copy number alteration data.
            - time (np.ndarray): Survival times (e.g., OS_MONTHS).
            - event (np.ndarray): Event indicators (1 = event occurred, 0 = censored).
    """

    rna = np.load(f'{data_path}/exp_{exp_num}_samples_3d_rna_{split_type}.npy')
    dna = np.load(f'{data_path}/exp_{exp_num}_samples_3d_dna_{split_type}.npy')
    cna = np.load(f'{data_path}/exp_{exp_num}_samples_3d_cna_{split_type}.npy')

    y = pd.read_csv(f'{data_path}/exp_{exp_num}_y_{split_type}.csv')

    time  = y['OS_MONTHS'].values
    event = y['OS_STATUS'].values

    return rna, dna, cna, time, event


def plot_all_metrics(
    train_loss: Sequence[float],
    val_loss: Sequence[float],
    train_cindex: Sequence[float],
    val_cindex: Sequence[float],
    train_auc: Sequence[float],
    val_auc: Sequence[float],
    config: dict
) -> None:
    """
    Plots training and validation curves for loss, C-index, and AUC
    in a single figure with multiple subplots.

    This function generates a consolidated visualization of model
    performance across epochs, enabling direct comparison between
    training and validation trends for multiple evaluation metrics.

    The resulting figure contains three subplots:
        (1) Loss curves
        (2) C-index curves
        (3) AUC curves

    Args:
        train_loss (Sequence[float]): Training loss values per epoch.
        val_loss (Sequence[float]): Validation loss values per epoch.
        train_cindex (Sequence[float]): Training C-index values per epoch.
        val_cindex (Sequence[float]): Validation C-index values per epoch.
        train_auc (Sequence[float]): Training AUC values per epoch.
        val_auc (Sequence[float]): Validation AUC values per epoch.
        config (dict): Configuration dictionary containing experiment
            settings such as 'cancer_type' and 'exp_num'.

    Returns:
        None
    """

    save_dir = f"plots_{config['cancer_type']}"
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # -------- Loss --------
    axes[0].plot(train_loss, label="train_loss")
    axes[0].plot(val_loss, label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # -------- C-index --------
    axes[1].plot(train_cindex, label="train_cindex")
    axes[1].plot(val_cindex, label="val_cindex")
    axes[1].set_title("C-Index")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("C-Index")
    axes[1].legend()
    axes[1].grid(True)

    # -------- AUC --------
    axes[2].plot(train_auc, label="train_auc")
    axes[2].plot(val_auc, label="val_auc")
    axes[2].set_title("AUC")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AUC")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/metrics_plot_{config['exp_num']}.png")
    plt.close()
    

def fit_model(config: dict[str, Any]) -> None:
    """
    Trains the model using the provided configuration settings.

    This function serves as the main training entry point. It expects a
    configuration dictionary containing all required hyperparameters,
    dataset settings, and training options, such as learning rate,
    weight decay, number of epochs, model dimensions, and experiment
    identifiers.

    Args:
        config (dict[str, Any]): Configuration dictionary containing
            training, model, and dataset parameters.

    Returns:
        None
    """
    
    dtype = torch.FloatTensor
    
    #Load the pathway order - this is determined based on PathCNN paper logic
    with open(f"{DATA_PATH}/exp_{config['exp_num']}_pathway_order.pkl", "rb") as file:
        pathway_order = pickle.load(file)

    # Load the pathway mask
    pathway_mask = load_pathway(f"{PATHWAY_MASK_PATH}/{config['cancer_type']}_pathway_mask.csv", dtype)

    # Update the order of pathway_mask
    pathway_mask = pathway_mask[pathway_order]

    # Load the training data
    split_type = 'train'
    train_rna, train_dna, train_cna, train_time, train_event = load_data(config['exp_num'], split_type, config['cancer_type'], DATA_PATH)
    train_dataset    = MultiOmicsDataset(train_rna, train_dna, train_cna, train_time, train_event, pathway_order)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=sort_collate_fn)

    # Load the validation data
    split_type = 'val'
    val_rna, val_dna, val_cna, val_time, val_event = load_data(config['exp_num'], split_type, config['cancer_type'], DATA_PATH)
    val_dataset    = MultiOmicsDataset(val_rna, val_dna, val_cna, val_time, val_event, pathway_order)
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, collate_fn=sort_collate_fn)

    # These are required for creating the model object
    num_genes = train_rna.shape[2]
    num_pathways = train_rna.shape[1]
    
    net = PIMO(num_genes, 
               num_pathways, 
               config['num_kernels1'], 
               config['dropout1'], 
               config['num_kernels2'], 
               config['dropout2'], 
               config['activation_fn'], 
               config['fc_nodes'], 
               pathway_mask)

    # if gpu is being used
    if torch.cuda.is_available():
        net.cuda()

    opt = optim.Adam(net.parameters(),
                     lr=config['learning_rate'],
                     betas=(0.99, 0.999),
                     weight_decay=config['weight_decay'],
                     amsgrad=True)
    
    # Used to keep track of training loss and c_index values
    train_loss_values, train_cindex_values, train_auc_values = [], [], []
    
    # Used to keep track of validation loss and c_index values
    val_loss_values, val_cindex_values, val_auc_values = [], [], []
    
    # Keep track of the best val loss
    best_val_loss = np.inf
    
    # counter for early stopping
    counter = 0

    # Training loop
    for epoch in range(1, config['num_epochs'] + 1):
        
        # Early stopping criteria
        if (epoch - counter) > config['patience'] and epoch > config['min_epochs']:
            
            # plot before breaking the loop
            plot_all_metrics(
                train_loss_values,
                val_loss_values,
                train_cindex_values,
                val_cindex_values,
                train_auc_values,
                val_auc_values,
                config
            )
            break
            
        if epoch % config['decay_epochs'] == 0:
            exponential_lr_decay(config['learning_rate'], opt, config['decay_rate'], int(epoch/config['decay_epochs']))

        # training mode
        net.train()

        train_total_loss = 0
        train_num_batches = 0
        train_preds = []
        train_times = []
        train_events = []

        for idx, (train_rna, train_dna, train_cna, train_time, train_event) in enumerate(train_dataloader):

            if torch.cuda.is_available():
                train_rna, train_dna, train_cna, train_time, train_event = train_rna.cuda(), train_dna.cuda(), train_cna.cuda(), train_time.cuda(), train_event.cuda()

            opt.zero_grad()

            # Forward Prop
            train_pred = net(train_rna, train_dna, train_cna)

            # Skip if the batch does not contain any events.
            # Adopted from PCLSurv paper
            if train_event.sum(0)==0:
                continue

            loss = neg_par_log_likelihood(train_pred, train_time, train_event)
            loss.backward()
            opt.step()

            train_total_loss  += loss.item()
            train_num_batches += 1
            
            train_preds.extend(train_pred.detach().cpu())
            train_times.extend(train_time.detach().cpu().unsqueeze(0))
            train_events.extend(train_event.detach().cpu().unsqueeze(0))

        train_preds = torch.cat(train_preds, dim=0)
        train_times = torch.cat(train_times, dim=0)
        train_events = torch.cat(train_events, dim=0)

        train_total_loss /= train_num_batches
        train_loss_values.append(train_total_loss)

        # Eval mode
        net.eval()

        val_total_loss = 0
        val_num_batches = 0
        val_preds = []
        val_times = []
        val_events = []

        for idx, (val_rna, val_dna, val_cna, val_time, val_event) in enumerate(val_dataloader):

            if torch.cuda.is_available():
                val_rna, val_dna, val_cna, val_time, val_event = val_rna.cuda(), val_dna.cuda(), val_cna.cuda(), val_time.cuda(), val_event.cuda()

            # Forward Prop
            val_pred = net(val_rna, val_dna, val_cna)

            loss = neg_par_log_likelihood(val_pred, val_time, val_event)

            val_total_loss += loss.item()
            val_num_batches += 1

            val_preds.extend(val_pred.detach().cpu())
            val_times.extend(val_time.detach().cpu().unsqueeze(0))
            val_events.extend(val_event.detach().cpu().unsqueeze(0))

        val_preds = torch.cat(val_preds, dim=0)
        val_times = torch.cat(val_times, dim=0)
        val_events = torch.cat(val_events, dim=0)

        val_total_loss /= val_num_batches
        val_loss_values.append(val_total_loss)

        if epoch % 10 == 0:
            print(f"Epoch# {epoch} Train-Loss {train_loss_values[-1]} Val-Loss {val_loss_values[-1]}")

        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            counter = epoch
            
            os.makedirs(f"{CHKPOINT_PATH}/{config['cancer_type']}", exist_ok=True)
            torch.save(net.state_dict(), f"{CHKPOINT_PATH}/{config['cancer_type']}/best_model_{config['exp_num']}.pt")

        # Sorting the data in the descending order of events is required if using c_index function for computing the Concordance Index
        # train_times, indices = torch.sort(train_times, descending=True)
#         train_events = train_events[indices]
#         train_preds  = train_preds[indices]

#         val_times, indices = torch.sort(val_times, descending=True)
#         val_events = val_events[indices]
#         val_preds  = val_preds[indices]
    
        train_cindex = CIndex(train_preds, train_times, train_events)
        train_auc    = AUC(train_preds, train_times, train_events)
        
        val_cindex = CIndex(val_preds, val_times, val_events)
        val_auc    = AUC(val_preds, val_times, val_events)

        train_cindex_values.append(train_cindex)
        train_auc_values.append(train_auc)
        
        val_cindex_values.append(val_cindex)
        val_auc_values.append(val_auc)
    
    # End timing and memory tracking
    measurement_results = end_measurement(start_time, process)
    
    # Save combined metric plot
    # If the code is trained for num_epochs
    plot_all_metrics(
        train_loss_values,
        val_loss_values,
        train_cindex_values,
        val_cindex_values,
        train_auc_values,
        val_auc_values,
        config
    )

    results = {
        
        "best_val_loss": best_val_loss,
        
        "train_loss_values": train_loss_values,
        "val_loss_values": val_loss_values,
        
        "train_cindex_values": train_cindex_values,
        "val_cindex_values": val_cindex_values,
        
        "train_auc_values": train_auc_values,
        "val_auc_values": val_auc_values,
    }

    return results


if __name__ == "__main__":    

    config = {
        'cancer_type': args.cancer_type,
        'exp_num'    : int(args.exp_num),
        'learning_rate': float(args.learning_rate),
        'decay_rate': float(args.decay_rate),
        'decay_epochs': int(args.decay_epochs),
        'weight_decay': float(args.weight_decay),
        'num_epochs': int(args.num_epochs),
        'num_kernels1': int(args.num_kernels1),
        'dropout1': float(args.dropout1),
        'num_kernels2': int(args.num_kernels2),
        'dropout2': float(args.dropout2),
        'activation_fn': str(args.activation_fn),
        'fc_nodes': int(args.fc_nodes),
        'batch_size': int(args.batch_size),
        'min_epochs': int(args.min_epochs),
        'patience': int(args.patience)
    }

    results = fit_model(config)
    
    os.makedirs(f"{RESULTS_PATH}/{config['cancer_type']}", exist_ok=True)
    with open(f"{RESULTS_PATH}/{config['cancer_type']}/results_{config['fold_num']}.pickle", 'wb') as file:
        pickle.dump(results, file)