import random
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
from torch_geometric.nn import GCNConv, SAGEConv, DeepGraphInfomax, JumpingKnowledge, GATConv
from torch.nn.utils import spectral_norm


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, heads=2, dropout=0.1):
        super(GAT, self).__init__()
        self.gc1 = GATConv(nfeat, nhid, heads=heads, dropout=dropout)

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        return x


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nmid=128, dropout=0.5):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(nfeat, nmid)
        self.gc2 = GCNConv(nmid, nhid)

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        x = nn.ReLU()(x)
        x = self.gc2(x, edge_index)
        x = nn.ReLU()(x)
        return x


class SAGE(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(nfeat, nhid, aggr='mean', normalize=True)
        self.transition = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(nhid),
            nn.Dropout(p=dropout)
        )
        self.conv2 = SAGEConv(nhid, nhid, aggr='mean', normalize=True)

        for m in self.modules():
            weights_init(m)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.transition(x)
        x = self.conv2(x, edge_index)
        return x


class JK(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(JK, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid)
        self.convx = GCNConv(nhid, nhid)
        self.jk = JumpingKnowledge(mode='max')
        self.transition = nn.Sequential(nn.ReLU(), )

        for m in self.modules():
            weights_init(m)

    def forward(self, x, edge_index):
        xs = []
        x = self.conv1(x, edge_index)
        x = self.transition(x)
        xs.append(x)
        for _ in range(1):
            x = self.convx(x, edge_index)
            x = self.transition(x)
            xs.append(x)
        x = self.jk(xs)
        return x
    

class Encoder_DGI(nn.Module):
    def __init__(self, nfeat, nhid):
        super(Encoder_DGI, self).__init__()
        self.hidden_ch = nhid
        self.conv = GCNConv(nfeat, self.hidden_ch)
        self.activation = nn.PReLU()

    def corruption(self, x, edge_index):
        # corrupted features are obtained by row-wise shuffling of the original features
        # corrupted graph consists of the same nodes but located in different places
        return x[torch.randperm(x.size(0))], edge_index

    def summary(self, z, *args, **kwargs):
        return torch.sigmoid(z.mean(dim=0))

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.activation(x)
        return x


class GraphInfoMax(nn.Module):
    def __init__(self, enc_dgi):
        super(GraphInfoMax, self).__init__()
        self.dgi_model = DeepGraphInfomax(enc_dgi.hidden_ch, enc_dgi, enc_dgi.summary, enc_dgi.corruption)

    def forward(self, x, edge_index):
        pos_z, neg_z, summary = self.dgi_model(x, edge_index)
        return pos_z
