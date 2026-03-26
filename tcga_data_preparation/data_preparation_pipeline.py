import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--cancer_type", type=str, default="brca")
args        = parser.parse_args()

CANCER_TYPE = args.cancer_type
PROCESSED_DATA_PATH   = f'processed_{CANCER_TYPE}'
SPLIT_DATA_PATH       = f'splits_{CANCER_TYPE}'

import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the order of the genes
with open(f"{PROCESSED_DATA_PATH}/{CANCER_TYPE}_gene_columns.pickle", "rb") as file:
    common_genes = pickle.load(file)

print(f"common_genes: {len(common_genes)}")

# Read the complete data
df_mor = pd.read_csv(f"{PROCESSED_DATA_PATH}/{CANCER_TYPE}_merged_multi_omics.csv")
print(f"len(df_mor): {len(df_mor)}")
print(f"df_mor.head()")
print(df_mor.head())

# Load the pathway mask
pathway_mask = pd.read_csv(f"{PROCESSED_DATA_PATH}/{CANCER_TYPE}_pathway_mask.csv", index_col=0).values
print(f"pathway_mask.shape: {pathway_mask.shape}")


# Generates a unique number between 1 and 100
random_seeds1 = random.sample(range(1, 101), 1)  

with open(f"{SPLIT_DATA_PATH}/random_seeds1.pickle", "wb") as file:
    pickle.dump(random_seeds1, file)

# Generates a unique numbers between 101 and 200
random_seeds2 = random.sample(range(101, 201), 1)

with open(f"{SPLIT_DATA_PATH}/random_seeds2.pickle", "wb") as file:
    pickle.dump(random_seeds2, file)
    

def convert_2d_to_3d(data_2d, pathway_mask):
    
    data_3d = []

    for i in tqdm(range(data_2d.shape[0])):
        data_3d.append(data_2d[i] * pathway_mask)
        
    return np.asarray(data_3d)


for i, (seed1, seed2) in enumerate(zip(random_seeds1, random_seeds2)):
    
    print(f"#######################  {i+1} experiment  #######################\n")
    
    all_indices = pd.DataFrame({"indices": [j for j in range(len(df_mor))]})
    y = df_mor[['OS_MONTHS', 'OS_STATUS']]
    
    # Train and Test Split
    train_indices, test_indices, y_train, y_test = train_test_split(all_indices, y, test_size=0.2, stratify=y["OS_STATUS"], random_state=seed1)
    
    # Val and Test Split
    val_indices, test_indices, y_val, y_test = train_test_split(test_indices, y_test, test_size=0.5, stratify=y_test["OS_STATUS"], random_state=seed2)
    
    # Saving the indices
    train_indices.to_csv(f"{SPLIT_DATA_PATH}/exp_{i+1}_train.csv", index=False)
    val_indices.to_csv(f"{SPLIT_DATA_PATH}/exp_{i+1}_val.csv", index=False)
    test_indices.to_csv(f"{SPLIT_DATA_PATH}/exp_{i+1}_test.csv", index=False)
    
    # Separate Data
    X_train = df_mor.loc[train_indices["indices"].values, df_mor.columns].drop(columns=["SAMPLE_ID", "Patient_ID", "OS_MONTHS", "OS_STATUS"]).values
    y_train = df_mor.loc[train_indices["indices"].values, ['OS_MONTHS', 'OS_STATUS']]
    
    X_val   = df_mor.loc[val_indices["indices"].values, df_mor.columns].drop(columns=["SAMPLE_ID", "Patient_ID", "OS_MONTHS", "OS_STATUS"]).values
    y_val   = df_mor.loc[val_indices["indices"].values, ['OS_MONTHS', 'OS_STATUS']]
    
    X_test  = df_mor.loc[test_indices["indices"].values, df_mor.columns].drop(columns=["SAMPLE_ID", "Patient_ID", "OS_MONTHS", "OS_STATUS"]).values
    y_test  = df_mor.loc[test_indices["indices"].values, ['OS_MONTHS', 'OS_STATUS']]
    
    data_scaler = StandardScaler()
    data_scaler.fit(X_train)
    
    with open(f"{SPLIT_DATA_PATH}/scaler_{i+1}.pickle", 'wb') as file:
        pickle.dump(data_scaler, file)
                        
    X_train_scaled = data_scaler.transform(X_train)
    X_val_scaled   = data_scaler.transform(X_val)
    X_test_scaled  = data_scaler.transform(X_test)
    
    y_train.to_csv(f'{SPLIT_DATA_PATH}/exp_{i+1}_y_train.csv', index=False)
    y_val.to_csv(f'{SPLIT_DATA_PATH}/exp_{i+1}_y_val.csv', index=False)
    y_test.to_csv(f'{SPLIT_DATA_PATH}/exp_{i+1}_y_test.csv', index=False)

    
    samples_2d_rna_train = X_train_scaled[:, 0: len(common_genes)]
    samples_3d_rna_train = convert_2d_to_3d(samples_2d_rna_train, pathway_mask)
    np.save(f"{SPLIT_DATA_PATH}/exp_{i+1}_samples_3d_rna_train.npy", samples_3d_rna_train)
        
    samples_2d_rna_val = X_val_scaled[:, 0: len(common_genes)]
    samples_3d_rna_val = convert_2d_to_3d(samples_2d_rna_val, pathway_mask)
    np.save(f"{SPLIT_DATA_PATH}/exp_{i+1}_samples_3d_rna_val.npy", samples_3d_rna_val)
    
    samples_2d_rna_test = X_test_scaled[:, 0: len(common_genes)]
    samples_3d_rna_test = convert_2d_to_3d(samples_2d_rna_test, pathway_mask)
    np.save(f"{SPLIT_DATA_PATH}/exp_{i+1}_samples_3d_rna_test.npy", samples_3d_rna_test)
    
    samples_2d_dna_train = X_train_scaled[:, len(common_genes): 2*len(common_genes)]
    samples_3d_dna_train = convert_2d_to_3d(samples_2d_dna_train, pathway_mask)    
    np.save(f"{SPLIT_DATA_PATH}/exp_{i+1}_samples_3d_dna_train.npy", samples_3d_dna_train)
        
    samples_2d_dna_val = X_val_scaled[:, len(common_genes): 2*len(common_genes)]
    samples_3d_dna_val = convert_2d_to_3d(samples_2d_dna_val, pathway_mask)
    np.save(f"{SPLIT_DATA_PATH}/exp_{i+1}_samples_3d_dna_val.npy", samples_3d_dna_val)
    
    samples_2d_dna_test = X_test_scaled[:, len(common_genes): 2*len(common_genes)]
    samples_3d_dna_test = convert_2d_to_3d(samples_2d_dna_test, pathway_mask)
    np.save(f"{SPLIT_DATA_PATH}/exp_{i+1}_samples_3d_dna_test.npy", samples_3d_dna_test)
    
    samples_2d_cna_train = X_train_scaled[:, 2*len(common_genes): 3*len(common_genes)]
    samples_3d_cna_train = convert_2d_to_3d(samples_2d_cna_train, pathway_mask)
    np.save(f"{SPLIT_DATA_PATH}/exp_{i+1}_samples_3d_cna_train.npy", samples_3d_cna_train)
    
    samples_2d_cna_val = X_val_scaled[:, 2*len(common_genes): 3*len(common_genes)]
    samples_3d_cna_val = convert_2d_to_3d(samples_2d_cna_val, pathway_mask)
    np.save(f"{SPLIT_DATA_PATH}/exp_{i+1}_samples_3d_cna_val.npy", samples_3d_cna_val)
    
    samples_2d_cna_test = X_test_scaled[:, 2*len(common_genes): 3*len(common_genes)]
    samples_3d_cna_test = convert_2d_to_3d(samples_2d_cna_test, pathway_mask)
    np.save(f"{SPLIT_DATA_PATH}/exp_{i+1}_samples_3d_cna_test.npy", samples_3d_cna_test)
    
    # PathCNN-based code logic to order the pathways based on correlation.
    q = 5

    print("Current q value: ", q)

    """For a pathway pi, the mRNA expression data of associated genes 
    were extracted from the mRNA expression matrix(G), 
    producing an intermediate matrix B 2 R_nxri, 
    where ri is the number of genes involved in the pathway pi. 
    That is, the matrix B consists of samples in rows and genes for a given pathway in columns.""" 

    ## RNA DATA PREP
    train_G_p = []
    val_G_p = []
    test_G_p = []
    
    for pi in range(samples_3d_rna_train.shape[1]):
        B_pi = samples_3d_rna_train[:, pi, :]

        """Using principal component analysis (PCA), the matrix B was 
        decomposed into uncorrelated components, yielding G_pi => R_nxq,
        where q is the number of principal components (PCs)."""
        G_pi = PCA(n_components=q)
        train_G_p.append(G_pi.fit_transform(B_pi))
        
        B_pi = samples_3d_rna_val[:, pi, :]
        val_G_p.append(G_pi.transform(B_pi))
        
        B_pi = samples_3d_rna_test[:, pi, :]
        test_G_p.append(G_pi.transform(B_pi))
        

    """Lastly, by rearranging the matrices for each sample sj, 
    a set of matrices, G_sj ; C_sj ; M_sj 2 R_numpathwaysXq, was produced"""
    train_G_p = np.array(train_G_p)
    print("Shape of train_G_p: ", train_G_p.shape)
    
    val_G_p = np.array(val_G_p)
    print("Shape of val_G_p: ", val_G_p.shape)
    
    test_G_p = np.array(test_G_p)
    print("Shape of test_G_p: ", test_G_p.shape)
    
    train_G_s = train_G_p.reshape(train_G_p.shape[1], train_G_p.shape[0], train_G_p.shape[2])
    np.save(f"{SPLIT_DATA_PATH}/exp_{i+1}_train_rna_pcas" + str(q) + ".npy", train_G_s)
    
    val_G_s   = val_G_p.reshape(val_G_p.shape[1], val_G_p.shape[0], val_G_p.shape[2])
    test_G_s  = test_G_p.reshape(test_G_p.shape[1], test_G_p.shape[0], test_G_p.shape[2])
    
    
    ## DNA DATA PREP
    train_M_p = []
    val_M_p = []
    test_M_p = []
    
    for pi in range(samples_3d_dna_train.shape[1]):
        B_pi = samples_3d_dna_train[:, pi, :]

        M_pi = PCA(n_components=q)
        train_M_p.append(M_pi.fit_transform(B_pi))
        
        B_pi = samples_3d_dna_val[:, pi, :]
        val_M_p.append(M_pi.transform(B_pi))
        
        B_pi = samples_3d_dna_test[:, pi, :]
        test_M_p.append(M_pi.transform(B_pi))

    
    """Lastly, by rearranging the matrices for each sample sj, 
    a set of matrices, G_sj ; C_sj ; M_sj 2 R_numpathwaysXq, was produced"""
    train_M_p = np.array(train_M_p)
    print("Shape of train_M_p: ", train_M_p.shape)
    
    val_M_p = np.array(val_M_p)
    print("Shape of val_M_p: ", val_M_p.shape)
    
    test_M_p = np.array(test_M_p)
    print("Shape of test_M_p: ", test_M_p.shape)
    
    train_M_s = train_M_p.reshape(train_M_p.shape[1], train_M_p.shape[0], train_M_p.shape[2])
    np.save(f"{SPLIT_DATA_PATH}/exp_{i+1}_train_dna_pcas" + str(q) + ".npy", train_M_s)
    
    val_M_s = val_M_p.reshape(val_M_p.shape[1], val_M_p.shape[0], val_M_p.shape[2])
    test_M_s = test_M_p.reshape(test_M_p.shape[1], test_M_p.shape[0], test_M_p.shape[2])
    
    
    train_C_p = []
    val_C_p = []
    test_C_p = []
    
    for pi in range(samples_3d_cna_train.shape[1]):
        B_pi = samples_3d_cna_train[:, pi, :]

        """Using principal component analysis (PCA), the matrix B was 
        decomposed into uncorrelated components, yielding G_pi => R_nxq,
        where q is the number of principal components (PCs)."""
        C_pi = PCA(n_components=q)
        train_C_p.append(C_pi.fit_transform(B_pi))
        
        B_pi = samples_3d_cna_val[:, pi, :]
        val_C_p.append(C_pi.transform(B_pi))
        
        B_pi = samples_3d_cna_test[:, pi, :]
        test_C_p.append(C_pi.transform(B_pi))
        

    """Lastly, by rearranging the matrices for each sample sj, 
    a set of matrices, G_sj ; C_sj ; M_sj 2 R_numpathwaysXq, was produced"""
    train_C_p = np.array(train_C_p)
    val_C_p   = np.array(val_C_p)
    test_C_p  = np.array(test_C_p)
    
    train_C_s = train_C_p.reshape(train_C_p.shape[1], train_C_p.shape[0], train_C_p.shape[2])
    np.save(f"{SPLIT_DATA_PATH}/exp_{i+1}_train_cna_pcas" + str(q) + ".npy", train_C_s)
    
    val_C_s   = val_C_p.reshape(val_C_p.shape[1], val_C_p.shape[0], val_C_p.shape[2])
    test_C_s  = test_C_p.reshape(test_C_p.shape[1], test_C_p.shape[0], test_C_p.shape[2])
    
    
    print(f"Running pathway ordering logic...")
    pathway_order = []
    
    # print("Current q value: ", q)
    rna_data = np.load(f"{SPLIT_DATA_PATH}/exp_{i+1}_train_rna_pcas" + str(q) + ".npy")
    dna_data = np.load(f"{SPLIT_DATA_PATH}/exp_{i+1}_train_dna_pcas" + str(q) + ".npy")
    cna_data = np.load(f"{SPLIT_DATA_PATH}/exp_{i+1}_train_cna_pcas" + str(q) + ".npy")
    
    combined_data = np.dstack((rna_data, dna_data, cna_data))
    
    print("RNA Data.shape: ", rna_data.shape)
    print("DNA Data.shape: ", dna_data.shape)
    print("CNA Data.shape: ", cna_data.shape)
    print("Combined Data.shape: ", combined_data.shape)
    
    pairwise_correlations = {}

    for j in tqdm(range(combined_data.shape[1]-1)):
        for k in range(j+1, combined_data.shape[1]):
            pairwise_correlations["p" + str(j) + "_p" + str(k)]  = np.corrcoef(combined_data[:, j, :].flatten(), combined_data[:, k, :].flatten())[0][1]

    pairwise_correlations = dict(sorted(pairwise_correlations.items(), key=lambda item: item[1], reverse=True))
    #print(f"pairwise_correlations: {pairwise_correlations}")
    
    first_pathway_pair = list(pairwise_correlations.keys())[0]
    removable_pathway_idx = [int(first_pathway_pair.split("_")[0][1:]), int(first_pathway_pair.split("_")[1][1:])]
    del pairwise_correlations[first_pathway_pair]
    
    pathway_id = removable_pathway_idx[0]
    current_pathway_dict = {key: value for key, value in pairwise_correlations.items() if pathway_id==int(key.split("_")[0][1:]) or pathway_id==int(key.split("_")[1][1:])}
    pathway_pair1 = list(current_pathway_dict.keys())[0]
    
    pathway_id = removable_pathway_idx[1]
    current_pathway_dict = {key: value for key, value in pairwise_correlations.items() if pathway_id==int(key.split("_")[0][1:]) or pathway_id==int(key.split("_")[1][1:])}
    pathway_pair2 = list(current_pathway_dict.keys())[0]
    
    selected_pathway_pair = None

    if pairwise_correlations[pathway_pair1] > pairwise_correlations[pathway_pair2]:
        selected_pathway_pair = pathway_pair1
    else:
        selected_pathway_pair = pathway_pair2

    pathway1 = int(selected_pathway_pair.split("_")[0][1:])
    pathway2 = int(selected_pathway_pair.split("_")[1][1:])
    to_remove = None

    if pathway1 in removable_pathway_idx:
        removable_pathway_idx.append(pathway2)
        to_remove = pathway1
    else:
        removable_pathway_idx.append(pathway1)
        to_remove = pathway2
        
    for key in current_pathway_dict.keys():
        del pairwise_correlations[key]
        
    pathway_id = removable_pathway_idx[0]
    current_pathway_dict = {key: value for key, value in pairwise_correlations.items() if pathway_id==int(key.split("_")[0][1:]) or pathway_id==int(key.split("_")[1][1:])}

    for key in current_pathway_dict.keys():
        if key in pairwise_correlations.keys():
            del pairwise_correlations[key]
            
    pathway_id = removable_pathway_idx[1]
    current_pathway_dict = {key: value for key, value in pairwise_correlations.items() if pathway_id==int(key.split("_")[0][1:]) or pathway_id==int(key.split("_")[1][1:])}

    for key in current_pathway_dict.keys():
        if key in pairwise_correlations.keys():
            del pairwise_correlations[key]
            
    while len(removable_pathway_idx)!=231:
        pathway_id = removable_pathway_idx[-1]
        current_pathway_dict = {key: value for key, value in pairwise_correlations.items() if pathway_id==int(key.split("_")[0][1:]) or pathway_id==int(key.split("_")[1][1:])}
        #current_pathway_dict = dict(sorted(current_pathway_dict.items(), key=lambda item: item[1], reverse=True))
        pathway_pair = list(current_pathway_dict.keys())[0]
        pathway1 = int(pathway_pair.split("_")[0][1:])
        pathway2 = int(pathway_pair.split("_")[1][1:])
        to_remove = None

        if pathway1 in removable_pathway_idx:
            removable_pathway_idx.append(pathway2)
            to_remove = pathway1
        else:
            removable_pathway_idx.append(pathway1)
            to_remove = pathway2

        for key in current_pathway_dict.keys():
            del pairwise_correlations[key]
    
    # print("Unique pathways: ", len(np.unique(removable_pathway_idx)))
    # print("top 15 pathways: ", removable_pathway_idx[:15])
    
    with open(f"{SPLIT_DATA_PATH}/exp_{i+1}_pathway_order.pkl", "wb") as f:
        pickle.dump(removable_pathway_idx, f)
    
    for j in range(combined_data.shape[0]):
        combined_data[j] = combined_data[j][removable_pathway_idx]
        
    #np.save(f"{DATA_PATH}/{CANCER_TYPE}/exp_{i+1}_combined_train_" + str(q) + ".npy", combined_data)
    print("-"*30)