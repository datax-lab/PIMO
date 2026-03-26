import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader


class MultiOmicsDataset(Dataset):
    
    def __init__(self, rna, dna, cna, time, event, pathway_order):
        
        self.dtype = torch.FloatTensor
        
        self.rna = torch.from_numpy(rna[:, pathway_order, :]).type(self.dtype)
        self.dna = torch.from_numpy(dna[:, pathway_order, :]).type(self.dtype)
        self.cna = torch.from_numpy(cna[:, pathway_order, :]).type(self.dtype)
        
        self.time = torch.from_numpy(time).type(self.dtype)
        self.event = torch.from_numpy(event).type(self.dtype)


    def __getitem__(self, index):
        return self.rna[index], self.dna[index], self.cna[index], self.time[index], self.event[index]

    def __len__(self):
        return len(self.rna)


def sort_collate_fn(batch):
    
    batch.sort(key=lambda x: x[3], reverse=True)

    rna, dna, cna, time, event = zip(*batch)
    return (
        torch.stack(rna),
        torch.stack(dna),
        torch.stack(cna),
        torch.stack(time),
        torch.stack(event),
    )


def sort_data(path):
    ''' sort the genomic and clinical data w.r.t. survival time (OS_MONTHS) in descending order
    Input:
        path: path to input dataset (which is expected to be a csv file).
    Output:
        x: sorted genomic inputs.
        ytime: sorted survival time (OS_MONTHS) corresponding to 'x'.
        yevent: sorted censoring status (OS_EVENT) corresponding to 'x', where 1 --> deceased; 0 --> censored.
        age: sorted age corresponding to 'x'.
    '''
    
    data = pd.read_csv(path)

    data.sort_values("OS_MONTHS", ascending=False, inplace=True)

    x = data.drop(["OS_MONTHS", "OS_STATUS"], axis=1).values
    ytime = data.loc[:, ["OS_MONTHS"]].values
    yevent = data.loc[:, ["OS_STATUS"]].values
    
    return (x, ytime, yevent)



def load_data(path, dtype):
    '''Load the sorted data, and then covert it to a Pytorch tensor.
    Input:
        path: path to input dataset (which is expected to be a csv file).
        dtype: define the data type of tensor (i.e. dtype=torch.FloatTensor)
    Output:
        X: a Pytorch tensor of 'x' from sort_data().
        YTIME: a Pytorch tensor of 'ytime' from sort_data().
        YEVENT: a Pytorch tensor of 'yevent' from sort_data().
        AGE: a Pytorch tensor of 'age' from sort_data().
    '''
    #x, ytime, yevent, age = sort_data(path)
    x, ytime, yevent = sort_data(path)
    X = torch.from_numpy(x).type(dtype)
    YTIME = torch.from_numpy(ytime).type(dtype)
    YEVENT = torch.from_numpy(yevent).type(dtype)
    #AGE = torch.from_numpy(age).type(dtype)
    ###if gpu is being used
    if torch.cuda.is_available():
        X = X.cuda()
        YTIME = YTIME.cuda()
        YEVENT = YEVENT.cuda()
        #AGE = AGE.cuda()
    ###
    return (X, YTIME, YEVENT) #(X, YTIME, YEVENT, AGE)


def load_pathway(path, dtype):
    '''Load a bi-adjacency matrix of pathways, and then covert it to a Pytorch tensor.
    Input:
        path: path to input dataset (which is expected to be a csv file).
        dtype: define the data type of tensor (i.e. dtype=torch.FloatTensor)
    Output:
        PATHWAY_MASK: a Pytorch tensor of the bi-adjacency matrix of pathways.
    '''
    pathway_mask = pd.read_csv(path).values
    pathway_mask = pathway_mask[:, 1:].astype(float)

    PATHWAY_MASK = torch.from_numpy(pathway_mask).type(dtype)
    ###if gpu is being used
    # if torch.cuda.is_available():
    #     PATHWAY_MASK = PATHWAY_MASK.cuda()
    ###
    return(PATHWAY_MASK)