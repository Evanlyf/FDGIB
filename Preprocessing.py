'''
This file is for data preprocessing and data analysis
'''
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from matplotlib import pyplot
import numpy as np
from numpy import cov
from scipy.stats import pearsonr
from scipy import spatial
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import scipy.io as scio

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(path_root, dataset):
    if dataset == 'bail':
        sens_attr = "WHITE"  
        sens_idx = 0  # column number after feature process is 0
        predict_attr = "RECID"
        label_number = 100
        path_bail = "./dataset/bail"
        adj, features, labels, train_mask, val_mask, test_mask, sens = load_bail(dataset, sens_attr,
                                                                              predict_attr, path=path_bail,
                                                                              label_number=label_number,
                                                                              )
        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features

    elif dataset == 'income':
        sens_attr = "race"  
        sens_idx = 8
        predict_attr = "income"
        label_number = 100
        path_income = "./dataset/income"
        adj, features, labels, train_mask, val_mask, test_mask, sens = load_income(dataset, sens_attr,
                                                                                predict_attr, path=path_income,
                                                                                label_number=label_number,
                                                                                )
        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features

    elif dataset == 'pokec_z' or dataset == 'pokec_n':
        sens_attr = "region"
        sens_idx = 3
        predict_attr = "I_am_working_in_field"
        label_number = 1000
        path_german = "./dataset/pokec"
        adj, features, labels, train_mask, val_mask, test_mask, sens = load_pokec(dataset, sens_attr,
                                                                               predict_attr, path=path_german,
                                                                               label_number=label_number,
                                                                               )
    else:
        print('Invalid dataset name!!')
        exit(0)

    print("loaded dataset: ", dataset, "num of node: ", len(features), ' feature dim: ', features.shape[1])

    return adj, features, labels, train_mask, val_mask, test_mask, sens, sens_idx
