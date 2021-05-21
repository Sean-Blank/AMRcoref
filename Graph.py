import torch
import torch.nn as nn
#from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
from grn import *


class GraphEncoder(nn.Module):
    def __init__(self, args):
        super(GraphEncoder, self).__init__()
        self.args = args
        self.gnn_type = self.args.gnn_type
        self.node_dim = self.args.concept_dim
        self.gnn_layers = nn.ModuleList()
        self.num_layers = self.args.gnn_layer_num
        if self.gnn_type == 'gcn':
            for i in range(self.num_layers):
                self.gnn_layers.append(GCNConv(self.node_dim, self.node_dim))
        elif self.gnn_type == 'gat':
            self.hidden_dim = self.node_dim
            # hidden layers
            for i in range(self.num_layers):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gnn_layers.append(GATConv(self.node_dim, self.node_dim, self.args.heads, concat=False))
        elif self.gnn_type == 'grn':
            self.gnn_layers = GRN(self.args)
        self.dropout = nn.Dropout(self.args.gnn_dropout)

    def forward(self, data):

        if self.gnn_type == 'gcn':
            x, edge_index = data[0], data[8]
            # edge_index = edge_index.transpose(0, 1)
            for i in range(self.num_layers):
                x = F.relu(self.gnn_layers[i](x.squeeze(0), edge_index))
            return x.unsqueeze(0)
        elif self.gnn_type == 'gat':
            x, edge_index = data[0], data[8]
            # edge_index = edge_index.transpose(0, 1)
            for i in range(self.num_layers):
                x = F.elu(self.gnn_layers[i](x.squeeze(0), edge_index))
            return x.unsqueeze(0)
        elif self.gnn_type == 'grn':
            _, x, _ = self.gnn_layers(data)
            return x