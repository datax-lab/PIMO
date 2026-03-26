import torch
import numpy as np

def R_set(x):
    '''Create an indicator matrix of risk sets, where T_j >= T_i.
    Note that the input data have been sorted in descending order.
    Input:
        x: a PyTorch tensor that the number of rows is equal to the number of samples.
    Output:
        indicator_matrix: an indicator matrix (which is a lower traiangular portions of matrix).
    '''
    n_sample = x.size(0)
    matrix_ones = torch.ones(n_sample, n_sample)
    indicator_matrix = torch.tril(matrix_ones)

    return(indicator_matrix)


def neg_par_log_likelihood(pred, ytime, yevent):
    '''Calculate the average Cox negative partial log-likelihood.
    Input:
        pred: linear predictors from trained model.
        ytime: true survival time from load_data().
        yevent: true censoring status from load_data().
    Output:
        cost: the cost that is to be minimized.
    '''
    n_observed = yevent.sum(0)
    ytime_indicator = R_set(ytime)
    ###if gpu is being used
    if torch.cuda.is_available():
        ytime_indicator = ytime_indicator.cuda()
    ###
    risk_set_sum = ytime_indicator.mm(torch.exp(pred)) 
    diff = pred - torch.log(risk_set_sum)
    sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(yevent.view(-1, 1))
    cost = (- (sum_diff_in_observed / n_observed)).reshape((-1,))
    return cost


def c_index(pred, ytime, yevent):
    '''Calculate concordance index to evaluate models.
    Input:
        pred: linear predictors from trained model.
        ytime: true survival time from load_data().
        yevent: true censoring status from load_data().
    Output:
        concordance_index: c-index (between 0 and 1).
    '''
    n_sample = len(ytime)
    ytime_indicator = R_set(ytime)
    ytime_matrix = ytime_indicator - torch.diag(torch.diag(ytime_indicator))
    ###T_i is uncensored
    censor_idx = (yevent == 0).nonzero()
    zeros = torch.zeros(n_sample)
    ytime_matrix[censor_idx, :] = zeros
    ###1 if pred_i < pred_j; 0.5 if pred_i = pred_j
    pred_matrix = torch.zeros_like(ytime_matrix)
    for j in range(n_sample):
        for i in range(n_sample):
            if pred[i] < pred[j]:
                pred_matrix[j, i]  = 1
            elif pred[i] == pred[j]: 
                pred_matrix[j, i] = 0.5

    concord_matrix = pred_matrix.mul(ytime_matrix)
    ###numerator
    concord = torch.sum(concord_matrix)
    ###denominator
    epsilon = torch.sum(ytime_matrix)
    ###c-index = numerator/denominator
    concordance_index = torch.div(concord, epsilon)
    ###if gpu is being used
    if torch.cuda.is_available():
        concordance_index = concordance_index.cuda()
    ###
    return(concordance_index)


# Used this from PCLSurv code implementation
def AUC(pred, ytime_test, ystatus_test):
    N_test = ystatus_test.shape[0]
    ystatus_test = np.squeeze(ystatus_test)
    ytime_test = np.squeeze(ytime_test)
    theta = np.squeeze(pred)
    total = 0
    count = 0

    for i in range(N_test):
        if ystatus_test[i] == 1:
            for j in range(N_test):
                # if ytime_test[i] < quantile_2 < ytime_test[j]:
                if ytime_test[i] < 12 * 1 < ytime_test[j]:
                    total = total + 1
                    if theta[j] < theta[i]:
                        count = count + 1
                    elif theta[j] == theta[i]:
                        count = count + 0.5

    for i in range(N_test):
        if ystatus_test[i] == 1:
            for j in range(N_test):
                # if ytime_test[i] < quantile_2 < ytime_test[j]:
                if ytime_test[i] < 12 * 5 < ytime_test[j]:
                    total = total + 1
                    if theta[j] < theta[i]:
                        count = count + 1
                    elif theta[j] == theta[i]:
                        count = count + 0.5

    for i in range(N_test):
        if ystatus_test[i] == 1:
            for j in range(N_test):
                # if ytime_test[i] < quantile_2 < ytime_test[j]:
                if ytime_test[i] < 12 * 10 < ytime_test[j]:
                    total = total + 1
                    if theta[j] < theta[i]:
                        count = count + 1
                    elif theta[j] == theta[i]:
                        count = count + 0.5

    if total == 0:
        return 0

    return count / total


# Used this from PCLSurv code implementation
def CIndex(pred, ytime_test, ystatus_test):
    N_test = ystatus_test.shape[0]
    ystatus_test = np.squeeze(ystatus_test)
    ytime_test = np.squeeze(ytime_test)
    theta = np.squeeze(pred)
    concord = 0.
    total = 0.
    eav_count = 0
    
    for i in range(N_test):
        
        #print(f'i:{i}, theta[i]:{theta[i]}')
        if ystatus_test[i] == 1:
            
            for j in range(N_test):
                #print(f'\tj:{j}, theta[j]:{theta[j]}')
                if ytime_test[j] > ytime_test[i]:
                    total = total + 1
                    if theta[j] < theta[i]:
                        concord = concord + 1
                        #print(f'i, j: {i},{j}')
                    elif theta[j] == theta[i]:
                        concord = concord + 0.5
                        eav_count = eav_count + 1
                        #print("相等的对数")
                        #print(eav_count)
    if total == 0:
        return 0
    #print("concord:", concord,  "total:", total)
    return concord / total