import os
import time
import argparse
import numpy as np
import random
import math
import csv
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score 

import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import pickle
import warnings

warnings.filterwarnings('ignore')

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')

parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--pred_epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--hidden_size', type=int, help='hidden size', default=32)
parser.add_argument('--hidden_size_2', type=int, help='hidden size', default=32)
parser.add_argument('--hidden_size_3', type=int, help='hidden size', default=16)
parser.add_argument('--dataset', type=str, default='bail',
                    choices=['bail', "income", "pokec_z", "pokec_n"])
parser.add_argument('--encoder', type=str, default='sage', choices=['gcn', 'sage', 'jk', 'gat'])
parser.add_argument('--rec', type=float, default=1, help='rec_coeff')
parser.add_argument('--ib', type=float, default=1, help='rec_coeff')
parser.add_argument('--dis', type=float, default=1, help='rec_coeff')

parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--proj_hidden', type=int, default=64, help='Number of hidden units in projection layer.')
parser.add_argument('--sim_coeff', type=float, default=0.3, help='regularization similarity')
parser.add_argument('--cls_coeff', type=float, default=0.6, help='regularization classifier')
parser.add_argument('--batch_size', type=int, help='batch size', default=100)
parser.add_argument('--subgraph_size', type=int, help='subgraph size', default=30)
parser.add_argument('--n_order', type=int, help='order of neighbor nodes', default=10)
parser.add_argument('--exp_times', type=int, default=3)

parser.add_argument('--step', type=int, default='5')
parser.add_argument('--rate', type=float, default=0.5, help='rate')
parser.add_argument('--head', type=int, default=2, help='gat head')
parser.add_argument('--gpu', type=str, default='7')


args = parser.parse_known_args()[0]
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

from model import Encoder, Classifier, Discriminator, Generator, GraphCF, GNN
from utils_mp import Subgraph, preprocess
import Preprocessing as dpp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

args.cuda = not args.no_cuda and torch.cuda.is_available()

# set seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

torch.backends.cudnn.allow_tf32 = False

# set device
cpu = torch.device("cpu")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def add_list_in_dict(key, dict, elem):
    if key not in dict:
        dict[key] = elem
    else:
        dict[key] += elem
    return dict


def save_to_excel(file_path, result_mean_df, result_all_df):
    if os.path.isfile(file_path):
        df_mean = pd.read_excel(file_path, sheet_name="mean")
        df_all = pd.read_excel(file_path, sheet_name="all")
        result_mean_df = pd.concat([df_mean, result_mean_df], ignore_index=True)
        result_all_df = pd.concat([df_all, result_all_df], ignore_index=True)
    with pd.ExcelWriter(file_path) as writer:
        result_mean_df.to_excel(writer, sheet_name="mean", index=False)
        result_all_df.to_excel(writer, sheet_name="all", index=False)
        
        
def save_result(file_name, the_header, result_dict):
    file_exists = os.path.isfile(file_name)
    with open(file_name, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=the_header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result_dict)


def fair_metric(pred, labels, sens):
    idx_s0 = sens == 0
    idx_s1 = sens == 1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)
    parity = abs(sum(pred[idx_s0]) / (sum(idx_s0) + 1e-10) - sum(pred[idx_s1]) / (sum(idx_s1)) + 1e-10)
    equality = abs(sum(pred[idx_s0_y1]) / (sum(idx_s0_y1) + 1e-10) - sum(pred[idx_s1_y1]) / (sum(idx_s1_y1)) + 1e-10)
    return parity.item(), equality.item()


def prediction_metric(lab, pred, preds, predcf, predcfs):
    acc = accuracy_score(lab.cpu().numpy(), preds.cpu().numpy())
    f1_s = f1_score(lab.cpu().numpy(), preds.cpu().numpy())
    auc_roc = roc_auc_score(lab.cpu().numpy(), pred.detach().cpu().numpy())

    acc2 = accuracy_score(lab.cpu().numpy(), predcfs.cpu().numpy())
    f1_s2 = f1_score(lab.cpu().numpy(), predcfs.cpu().numpy())
    auc_roc2 = roc_auc_score(lab.cpu().numpy(), predcf.detach().cpu().numpy())
    return acc, f1_s, auc_roc, acc2, f1_s2, auc_roc2


def subgraph_x_adj(ini_idx, subg, labels):
    sample_size = min(args.batch_size, len(ini_idx))  # sample central node
    sample_idx = random.sample(list(ini_idx.cpu().numpy()), sample_size)  # select |batch size| central nodes
    batch, index = subg.search(sample_idx)
    label = labels[sample_idx]
    node_num = batch.x.shape[0]

    # build batch_adj
    idx = np.arange(node_num)
    idx_map = {j: i for i, j in enumerate(idx)}
    batch_edge_index = batch.edge_index.T
    edges = np.array(list(map(idx_map.get, batch_edge_index.flatten().cpu().numpy())),
                     dtype=int).reshape(batch_edge_index.shape)
    bat_adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(node_num, node_num), dtype=np.float32)
    batch_adj = bat_adj + bat_adj.T.multiply(bat_adj.T > bat_adj) - bat_adj.multiply(bat_adj.T > bat_adj)

    adj_coo = batch_adj.tocoo()
    indices_adj = torch.LongTensor([adj_coo.row, adj_coo.col])
    adj_t = torch.sparse_coo_tensor(indices_adj, adj_coo.data, size=(adj_coo.shape[0], adj_coo.shape[1])).float()
    adj_dense = adj_t.to_dense().to(device)
    return batch, index, adj_dense, label


def disen_train(epochs, net_T, net_R, net_Cls, net_Disc, net_Gen, optimizer_1, optimizer_2, data, subgraph, sen_idx,
                idx_train, exp_id):
    print("start training!")
    all_labels = data.y
    for epoch in range(1, epochs + 1):
        torch.cuda.empty_cache()
        batch, index, batch_adj, _ = subgraph_x_adj(idx_train, subgraph, all_labels)
        S = batch.x[:, sen_idx].reshape(-1, 1).to(device)

        _, _, _, _ = [x.train() for x in (net_T, net_Cls, net_Disc, net_Gen)]
        net_R.eval()
        optimizer_1.zero_grad()

        Tmean, Tlogvar, T = net_T(batch.x.to(device), batch.edge_index.to(device))
        Rmean, Rlogvar, R = net_R(batch.x.to(device), batch.edge_index.to(device))
        loss_GT = net_T.loss(Tmean, Tlogvar, T)

        logits = net_Cls(T)
        loss_c = net_Cls.loss(logits, S)

        score = net_Disc(R.detach())
        loss_d = net_Disc.loss(score, S)

        z_x, z_A = net_Gen(R, T, S)
        loss_reconst_x = net_Gen.loss_x(z_x, batch.x.to(device))
        loss_reconst_A = net_Gen.loss_adj(z_A, batch_adj.to(device))

        a_rec, a_ib, a_dis = args.rec, args.ib, args.dis
        loss = a_rec*(loss_reconst_x / loss_reconst_x.detach() + loss_reconst_A / loss_reconst_A.detach()) + \
               a_ib*(loss_c / loss_c.detach() - loss_GT / loss_GT.detach()) + a_dis*(loss_d / loss_d.detach())

        # loss = loss_reconst_x + loss_reconst_A + loss_c - loss_GT + loss_d
        loss.backward(retain_graph=True)
        optimizer_1.step()

        net_R.train()
        net_Disc.eval()
        optimizer_2.zero_grad()
        Rmean, Rlogvar, R = net_R(batch.x.to(device), batch.edge_index.to(device))
        score = net_Disc(R)
        loss_R = - net_Disc.loss(score, S)  # F.mse_loss(score.view(-1), 0.5 * torch.ones_like(score.view(-1)))
        loss_R.backward(retain_graph=True)
        optimizer_2.step()

        if epoch == 1 or epoch % 100 == 0:
            print(epoch,
                  '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(loss_reconst_x.item(), loss_reconst_A.item(),
                                                                     loss_c.item(), loss_GT.item(), loss_d.item(),
                                                                     loss_R.item()))
    return net_T, net_R, net_Gen


def get_all_node_emb(model, subgraph, idx_select, sen_idx, net_T, net_R, net_Gen):
    node_list = idx_select.tolist()
    list_size = len(node_list)
    z = torch.Tensor(list_size, args.hidden_size).to(device)
    z_cf = torch.Tensor(list_size, args.hidden_size).to(device)
    sen = torch.Tensor(list_size).to(device)
    group_nb = math.ceil(list_size / args.batch_size)  # num of batches
    for i in range(group_nb):
        with torch.no_grad():
            maxx = min(list_size, (i + 1) * args.batch_size)
            minn = i * args.batch_size
            batch, index = subgraph.search(node_list[minn:maxx])
            x, edge_ind = batch.x.to(device), batch.edge_index.to(device)

            node = model(x, edge_ind, index.to(device)) 
            z[minn:maxx] = node

            S = x[:, sen_idx]
            S_cf = 1 - S
            x_cf = x.clone()
            x_cf[:, sen_idx] = S_cf

            S_cf = S_cf.reshape(-1, 1).to(device)
            _, _, R = net_R(x.to(device), edge_ind.to(device))
            _, _, T_cf = net_T(x_cf.to(device), edge_ind.to(device))
            x_cf, A_cf = net_Gen(R, T_cf.to(device), S_cf.to(device))
            adj_cf = sp.coo_matrix(A_cf.cpu().detach().numpy())
            indices = np.vstack((adj_cf.row, adj_cf.col))
            edge_index_cf = torch.LongTensor(indices)
            node_cf = model(x_cf.to(device), edge_index_cf.to(device), index.to(device))
            z_cf[minn:maxx] = node_cf
            sen[minn:maxx] = S[index].to(device)
    return z, z_cf, sen


def compute_loss_sim(model, z1, z2):
    # projector
    p1 = model.projection(z1)
    p2 = model.projection(z2)

    # predictor
    h1 = model.prediction(p1)
    h2 = model.prediction(p2)

    l1 = model.sim(h1, p2) / 2
    l2 = model.sim(h2, p1) / 2
    sim_loss = args.sim_coeff * (l1 + l2)

    return sim_loss


def evaluation(model, net_T, net_R, net_Gen, data, subgraph, sen_idx, idx_select, types="all"):
    emb, emb_cf, sen = get_all_node_emb(model, subgraph, idx_select, sen_idx, net_T, net_R, net_Gen)
    label = data.y[idx_select]

    pred_z, pred_z_cf = model.classifier(emb), model.classifier(emb_cf)

    l1 = F.binary_cross_entropy_with_logits(pred_z, label.unsqueeze(1).float().to(device)) / 2
    l2 = F.binary_cross_entropy_with_logits(pred_z_cf, label.unsqueeze(1).float().to(device)) / 2
    loss_sim = compute_loss_sim(model, emb, emb_cf)

    pred_zs = (pred_z.squeeze() > 0).type_as(label)
    pred_z_cfs = (pred_z_cf.squeeze() > 0).type_as(label)

    if sum(pred_zs) == 0 or sum(pred_zs) == len(pred_zs):
        return {'acc': 0, 'auc': 0, 'f1': 0, 'acc2': 0, 'auc2': 0, 'f12': 0,
                'parity': 1, 'equality': 1, 'loss_pred': l1, 'loss_cf': l2, 'loss_sim': loss_sim}

    acc, f1_s, auc_roc, acc2, f1_s2, auc_roc2 = prediction_metric(label, pred_z, pred_zs, pred_z_cf, pred_z_cfs)
    parity, equality = fair_metric(pred_zs.cpu().numpy(), label.cpu().numpy(), sen.cpu().numpy())

    eval_result = {'acc': acc, 'auc': auc_roc, 'f1': f1_s, 'acc2': acc2, 'auc2': auc_roc2, 'f12': f1_s2,
                   'parity': parity, 'equality': equality, 'loss_pred': l1, 'loss_cf': l2, 'loss_sim': loss_sim}
    return eval_result


def train_graph_cf(epochs, net_T, net_R, net_Gen, data, subgraph, sen_idx, idx_train, idx_val, idx_test, exp_id):
    print('start training cf')
    torch.cuda.empty_cache()
    ft_num = data.x.shape[1]
    hidden_size = args.hidden_size
    gcf_model = GraphCF(ft_num, hidden_size, 1, gnn=GNN(ft_num, hidden_size, args, args.encoder)).to(device)
    par1 = list(gcf_model.gnn.parameters()) + list(gcf_model.fc1.parameters()) + list(gcf_model.fc2.parameters()) \
           + list(gcf_model.fc3.parameters()) + list(gcf_model.fc4.parameters())

    par2 = list(gcf_model.c1.parameters()) + list(gcf_model.gnn.parameters())
    optimizer1 = torch.optim.Adam(par1, lr=args.lr, weight_decay=args.weight_decay)
    optimizer2 = torch.optim.Adam(par2, lr=args.lr, weight_decay=args.weight_decay)
    best_trade_off = 1

    for epoch in range(1, epochs + 1):
        torch.cuda.empty_cache()
        _, _, _ = [x.eval() for x in (net_T, net_R, net_Gen)]
        gcf_model.train()
        optimizer1.zero_grad()
        optimizer2.zero_grad()

        batch, index, batch_adj, label = subgraph_x_adj(idx_train, subgraph, data.y)
        label = label.unsqueeze(1).float().to(device)

        x, edge_index = batch.x, batch.edge_index
        S = x[:, sen_idx]
        S_cf = 1 - S
        x_cf = x.clone()
        x_cf[:, sen_idx] = S_cf
        S_cf = S_cf.reshape(-1, 1).to(device)

        _, _, R = net_R(x.to(device), edge_index.to(device))
        _, _, T_cf = net_T(x_cf.to(device), edge_index.to(device))
        x_cf, A_cf = net_Gen(R, T_cf.to(device), S_cf.to(device))
        x_cf[:, sen_idx] = 1 - x[:, sen_idx]
        adj_cf = sp.coo_matrix(A_cf.cpu().detach().numpy())
        edge_index_cf = torch.LongTensor(np.vstack((adj_cf.row, adj_cf.col)))

        z = gcf_model(x.to(device), edge_index.to(device), index.to(device))
        p1 = gcf_model.projection(z)
        h1 = gcf_model.prediction(p1)

        z_cf = gcf_model(x_cf.to(device), edge_index_cf.to(device), index.to(device))
        p2 = gcf_model.projection(z_cf)
        h2 = gcf_model.prediction(p2)

        sim_loss = args.sim_coeff * (gcf_model.sim(h1, p2) + gcf_model.sim(h2, p1)) / 2
        sim_loss.backward(retain_graph=True)
        optimizer1.step()

        z = gcf_model(x.to(device), edge_index.to(device), index.to(device))
        z_cf = gcf_model(x_cf.to(device), edge_index_cf.to(device), index.to(device))
        pred_z = gcf_model.classifier(z)
        pred_z_cf = gcf_model.classifier(z_cf)

        loss1 = F.binary_cross_entropy_with_logits(pred_z, label) / 2
        loss2 = F.binary_cross_entropy_with_logits(pred_z_cf, label) / 2
        loss = (1 - args.sim_coeff) * (args.cls_coeff * loss1 + (1 - args.cls_coeff) * loss2)

        loss1.backward()
        optimizer2.step()

        if epoch % args.step == 0:
            gcf_model.eval()
            eval_res = evaluation(gcf_model, net_T, net_R, net_Gen, data, subgraph, sen_idx, idx_val)
            val_loss = (eval_res['loss_pred']).item()
            print(epoch, eval_res['loss_pred'].item(), eval_res['loss_sim'].item())
            val_acc, val_f1, val_auc, val_pa, val_eo = eval_res['acc'], eval_res['f1'], eval_res['auc'], \
                                                       eval_res['parity'], eval_res['equality']
            trade_off = val_acc + val_f1 + val_auc - val_pa - val_eo
            if trade_off > best_trade_off and (val_auc > args.auc or val_f1 > args.f1):
                best_trade_off = trade_off
                test_result = test(gcf_model, net_T, net_R, net_Gen, data, subgraph, sen_idx, idx_test, exp_i)

            print('{}\nloss1: {:.4f}\tloss2: {:.4f}\nacc: {:.4f}\tf1_s: {:.4f}\tauc_roc: {:.4f}\nacc: {:.4f}\t'
                  'f1_s: {:.4f}\tauc_roc: {:.4f}\nparity: ''{:.4f}\tequality: {:.4f}'
                  .format(epoch, eval_res['loss_pred'], eval_res['loss_cf'], eval_res['acc'], eval_res['f1'],
                          eval_res['auc'], eval_res['acc2'], eval_res['f12'], eval_res['auc2'], eval_res['parity'],
                          eval_res['equality']))
    try:
        return test_result
    except UnboundLocalError:
        return test(gcf_model, net_T, net_R, net_Gen, data, subgraph, sen_idx, idx_test, exp_i)


def test(gcfmodel, net_T, net_R, net_Gen, data, subgraph, sen_idx, idx_test, exp_id):
    print('Start test!')
    ac, f1, auc, pa, eq = [], [], [], [], []
    ac2, f12, auc2 = [], [], []
    _, _, _, _ = [x.eval() for x in (gcfmodel, net_T, net_R, net_Gen)]

    emb, emb_cf, sen = get_all_node_emb(gcfmodel, subgraph, idx_test, sen_idx, net_T, net_R, net_Gen)
    label = data.y[idx_test]
    pred_z, pred_z_cf = gcfmodel.classifier(emb), gcfmodel.classifier(emb_cf)

    pred_zs = (pred_z.squeeze() > 0).type_as(label)
    pred_z_cfs = (pred_z_cf.squeeze() > 0).type_as(label)

    acc, f1_s, auc_roc, acc2, f1_s2, auc_roc2 = prediction_metric(label, pred_z, pred_zs, pred_z_cf, pred_z_cfs)
    parity, equality = fair_metric(pred_zs.cpu().numpy(), label.cpu().numpy(), sen.cpu().numpy())

    results = [(ac, acc), (f1, f1_s), (auc, auc_roc), (ac2, acc2), (f12, f1_s2), (auc2, auc_roc2),
               (pa, parity), [eq, equality]]
    for lst, item in results:
        lst.append(item)

    eval_result = {'acc': ac, 'f1': f1, 'auc_roc': auc, 'parity': pa, 'equality': eq,
                   'acc_cf': ac2, 'f1_cf': f12, 'auc_roc_cf': auc2}
    return eval_result


if __name__ == '__main__':
    data_path_root = '../'
    model_path = 'models_save/'
    
    if args.dataset == 'credit':
        args.auc, args.f1 = 0.70, 0.82
    elif args.dataset == 'bail':
        args.auc, args.f1 = 0.9, 0.8
    elif args.dataset == 'income':
        args.auc, args.f1 = 0.8, 0.54
    elif args.dataset == 'pokec_z' or args.dataset == 'pokec_n':
        args.auc, args.f1 = 0.65, 0.63


    adj, features, labels, idx_train, idx_val, idx_test, sens, sens_idx = dpp.load_data(
        data_path_root, args.dataset)

    edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)  
    num_class = 1

    n = features.shape[0]
    data = Data(x=features, y=labels, edge_index=edge_index)
    ft_dim = data.x.size(1)  

    # Subgraph: Setting up the subgraph extractor
    ppr_path = './graphFair_subgraph/' + args.dataset
    subgraph = Subgraph(data.x, data.edge_index, ppr_path, args.subgraph_size, args.n_order)
    subgraph.build()

    results_all_exp = {}
    exp_num = args.exp_times

    for exp_i in range(0, exp_num):
        idx_train, _ = torch.sort(idx_train)
        idx_val, _ = torch.sort(idx_val)
        idx_test, _ = torch.sort(idx_test)

        hidden_dim_1 = 256
        hidden_dim_2 = args.hidden_size_2  
        hidden_dim_3 = args.hidden_size_3  
        net_T = Encoder(ft_dim, hidden_dim_2).to(device)
        net_R = Encoder(ft_dim, hidden_dim_2).to(device)
        net_cls = Classifier(hidden_dim_2, hidden_dim_3, num_class).to(device)
        net_disc = Discriminator(hidden_dim_2, hidden_dim_3).to(device)
        net_gen = Generator(ft_dim, hidden_dim_2 * 2).to(device)

        par_1 = list(net_T.parameters()) + list(net_cls.parameters()) + \
                list(net_disc.parameters()) + list(net_gen.parameters())

        par_2 = list(net_R.parameters())

        optimizer_1 = torch.optim.Adam(par_1, lr=args.lr, weight_decay=args.weight_decay)
        optimizer_2 = torch.optim.Adam(par_2, lr=args.lr, weight_decay=args.weight_decay)
        net_T, net_R, net_gen = disen_train(args.epochs, net_T, net_R, net_cls, net_disc, net_gen,
                                            optimizer_1, optimizer_2, data, subgraph, sens_idx, idx_train, exp_i)

        eval_results = train_graph_cf(args.pred_epochs, net_T, net_R, net_gen, data, subgraph, sens_idx, idx_train,
                                        idx_val, idx_test, exp_i)

        print('======================= ', str(args.dataset), '_Exp_', str(exp_i), ' ==============================')

        for key in eval_results:
            results_all_exp = add_list_in_dict(key, results_all_exp, eval_results[key])
            if isinstance(eval_results[key], list):
                print(key, f":{np.mean(eval_results[key]):.4f}")
            else:
                print(key, f": {eval_results[key]:.4f}")

    print('======================= ', str(args.dataset), ' Overall =======================')
    res_all, res_mean = {}, {}
    for k in results_all_exp:
        res_all[k] = results_all_exp[k]
        results_all_exp[k] = np.array(results_all_exp[k])*100
        mean, std = np.mean(results_all_exp[k]), np.std(results_all_exp[k])
        print(k, f": mean: {mean:.4f} | std: {std:.4f}")
        tmp_res = str(round(mean, 4)) + "±" + str(round(std, 4))
        res_all[k].append(tmp_res)
        res_all[k] =  [ res_all[k] ]
        res_mean[k] = [round(mean, 4)]

    paras = ['dataset', 'epochs', 'pred_epochs', 'encoder', 'dropout', 'hidden_size', 'sim_coeff', 'cls_coeff', 'rate', 'hidden_size_2', 'hidden_size_3']
    for k in args.__dict__:
        if k in paras:
            res_mean[k] = args.__dict__[k]
    res_mean['time'] = [time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))]
    res_all['time'] = [time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))]
    
    result_mean_df = pd.DataFrame.from_dict(res_mean)
    result_all_df = pd.DataFrame.from_dict(res_all)
    save_to_excel('./reports/result_{}.xlsx'.format(args.dataset), result_mean_df, result_all_df)
    save_to_excel('./reports/backup/result_{}.xlsx'.format(args.dataset), result_mean_df, result_all_df)
    
