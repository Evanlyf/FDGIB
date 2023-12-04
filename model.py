import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
from torch.autograd import Variable
import numpy as np
import pyro

from GNNs import *

from sklearn.linear_model import LogisticRegression
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool, SAGPooling, GATConv, \
    GINConv, SAGEConv, DeepGraphInfomax, JumpingKnowledge

from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score
from torch.nn.utils import spectral_norm
from torch.nn.parameter import Parameter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module): 
    def __init__(self, feature_dim, h_dim):
        super(Encoder, self).__init__()
        self.gcn = GCNConv(feature_dim, h_dim*2)
        self.gcn_mean = GCNConv(h_dim*2, h_dim)
        self.gcn_logvar = GCNConv(h_dim*2, h_dim)
        self.line = nn.Sequential(nn.Linear(h_dim, h_dim), nn.Tanh())

    def encode_G(self, x, edge_index):
        num = x.shape[0]
        hidden = self.gcn(x, edge_index)
        hidden = F.relu(hidden)
        mean = self.gcn_mean(hidden, edge_index)
        # mean = F.relu(mean)
        logvar = self.gcn_logvar(hidden, edge_index)
        logvar = F.relu(logvar)
        logvar = self.line(logvar)

        gaussian_noise = torch.randn(*mean.shape).to(device)
        z = gaussian_noise * torch.exp(0.5*logvar) + mean
        return mean, logvar, z

    def forward(self, x, edge_index):
        z = self.encode_G(x, edge_index)
        return z

    def loss(self, mu, logvar, z):
        var = logvar.exp()
        positive = -(mu - z)**2/var
        idx0 = np.random.permutation(z.shape[0])
        z_shuffle = z[idx0, :]
        negative = -(mu - z_shuffle)**2/var

        positive, negative = torch.mean(positive), torch.mean(negative)
        # positive, negative = positive.sum(dim=1).mean(dim=0), negative.sum(dim=1).mean(dim=0)
        loss_club = positive - negative
        return loss_club


class Classifier(nn.Module):
    def __init__(self, input_dim, h_dim, n_class):
        super(Classifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, n_class),
            nn.Sigmoid(),
        )

    def forward(self, t):
        res = self.layer(t)
        return res

    def loss(self, s_pred, s_true):
        loss_bce = nn.BCELoss()
        loss_cls = loss_bce(s_pred, s_true)
        return loss_cls


class Discriminator(nn.Module):
    def __init__(self, input_dim, h_dim):
        super(Discriminator, self).__init__()
        self.dc = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(h_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, r):
        score = self.dc(r)
        return score

    def loss(self, s_pred, s_true):
        loss_bce = nn.BCELoss()
        loss_cls = loss_bce(s_pred, s_true)
        return loss_cls


class Generator(nn.Module):
    def __init__(self, feature_dim, h_dim, h_dim2=16):
        super(Generator, self).__init__()
        self.pred_x = nn.Sequential(nn.Linear(h_dim, feature_dim))
        self.pred_A = nn.Sequential(nn.Linear(h_dim, h_dim2), nn.ReLU(0.2))

    def forward(self, R, T, s):
        z = torch.cat((R, T), dim=-1)
        x = self.pred_x(z)
        A = self.decode_a(z)
        return x, A

    def decode_a(self, z):
        logits = self.pred_A(z)
        adj_logits = logits @ logits.T
        z_a = self.concrete_sample(adj_logits, 1)
        return z_a

    def concrete_sample(self, logit, tau=1.0):
        random_noise = torch.rand_like(logit)
        gumbel = -torch.log(-torch.log(random_noise))
        gate_inputs = (gumbel + logit) / tau
        gate_inputs = torch.sigmoid(gate_inputs)
        return gate_inputs

    def loss_adj(self, A_pred, adj_dense):
        pos_weight = float(adj_dense.shape[0] * adj_dense.shape[0] - adj_dense.sum()) / adj_dense.sum()
        weight_mask = adj_dense.view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0))
        weight_tensor[weight_mask] = pos_weight

        bce_loss = F.binary_cross_entropy(A_pred.view(-1), adj_dense.view(-1), reduction='mean', weight=weight_tensor.to(device))
        loss_reconst_a = bce_loss
        return loss_reconst_a

    def loss_x(self, x_pred, x_true):
        loss_mse = nn.MSELoss(reduction='mean')
        loss_reconst_x = loss_mse(x_pred, x_true)
        return loss_reconst_x

    def loss_A(self, A_pred, adj):
        adj_coo = adj.tocoo()
        indices_adj = torch.LongTensor([adj_coo.row, adj_coo.col])
        adj_t = torch.sparse_coo_tensor(indices_adj, adj_coo.data, size=(adj_coo.shape[0], adj_coo.shape[1])).float()
        adj_dense = adj_t.to_dense()
        loss_bce = nn.BCELoss(reduction='mean')
        loss_reconst_a = loss_bce(A_pred.reshape(-1), adj_dense.reshape(-1))
        return loss_reconst_a


class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, args, base_model='sage'):
        super(GNN, self).__init__()
        self.base_model = base_model
        if self.base_model == 'gcn':
            self.conv = GCN(in_channels, out_channels, args.mid)
        elif self.base_model == 'sage':
            self.conv = SAGE(in_channels, out_channels, args.dropout)
        elif self.base_model == 'jk':
            self.conv = JK(in_channels, out_channels)
        elif self.base_model == 'gat':
            head = args.head
            self.conv = GAT(in_channels, int(out_channels/head), head, args.dropout)

        self.relu = nn.ReLU()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        torch.cuda.empty_cache()
        x = self.conv(x, edge_index)
        return x


class GraphCF(torch.nn.Module):
    def __init__(self, num_features, hidden_size, num_class, gnn):
        super(GraphCF, self).__init__()
        self.num_features = num_features  # to get center node rep
        self.hidden_size = hidden_size
        self.num_class = num_class
        self.num_proj_hidden = 64

        self.gnn = gnn

        # Classifier
        self.c1 = spectral_norm(nn.Linear(self.hidden_size, self.num_class), )

        # Projection
        self.fc1 = nn.Sequential(
            spectral_norm(nn.Linear(self.hidden_size, self.num_proj_hidden)),
            nn.BatchNorm1d(self.num_proj_hidden),
            nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(
            spectral_norm(nn.Linear(self.num_proj_hidden, self.hidden_size)),
            nn.BatchNorm1d(self.hidden_size))

        # Prediction
        self.fc3 = nn.Sequential(
            spectral_norm(nn.Linear(self.hidden_size, self.hidden_size)),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(inplace=True))
        self.fc4 = spectral_norm(nn.Linear(self.hidden_size, self.hidden_size))

    def forward(self, x, edge_index, index=None):
        torch.cuda.empty_cache()
        hidden = self.gnn(x, edge_index)
        if index is None:
            return hidden
        z = hidden[index]
        return z

    def projection(self, z):
        z = self.fc1(z)
        z = self.fc2(z)
        return z 

    def prediction(self, z):
        z = self.fc3(z)
        z = self.fc4(z)
        return z

    def classifier(self, z):
        x = self.c1(z)
        return x

    def sim(self, x1, x2):  # negative cosine similarity
        return -F.cosine_similarity(x1, x2.detach(), dim=-1).mean()
