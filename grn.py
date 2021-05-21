import torch
import torch.nn as nn
import argparse
from config import parse_config



class GRN(nn.Module):
    def __init__(self, args):
        super(GRN, self).__init__()
        # debug
        self.args = args
        self.edge_vocab_size = args.edge_vocab_size
        self.edge_dim = args.embed_dim
        self.node_dim = args.embed_dim
        self.hidden_dim = args.embed_dim
        self.gnn_layers = args.gnn_layer_num
        self.dropout = nn.Dropout(self.args.gnn_dropout)
        self.edge_embedding = nn.Embedding(self.edge_vocab_size, self.edge_dim)
        # input gate
        self.W_ig_in = nn.Linear(self.node_dim + self.edge_dim, self.hidden_dim)
        self.W_ig_out = nn.Linear(self.node_dim + self.edge_dim, self.hidden_dim)
        self.U_ig_in = nn.Linear(self.node_dim + self.edge_dim, self.hidden_dim)
        self.U_ig_out = nn.Linear(self.node_dim + self.edge_dim, self.hidden_dim)

        # forget gate
        self.W_fg_in = nn.Linear(self.node_dim + self.edge_dim, self.hidden_dim)
        self.W_fg_out = nn.Linear(self.node_dim + self.edge_dim, self.hidden_dim)
        self.U_fg_in = nn.Linear(self.node_dim + self.edge_dim, self.hidden_dim)
        self.U_fg_out = nn.Linear(self.node_dim + self.edge_dim, self.hidden_dim)

        # output gate
        self.W_og_in = nn.Linear(self.node_dim + self.edge_dim, self.hidden_dim)
        self.W_og_out = nn.Linear(self.node_dim + self.edge_dim, self.hidden_dim)
        self.U_og_in = nn.Linear(self.node_dim + self.edge_dim, self.hidden_dim)
        self.U_og_out = nn.Linear(self.node_dim + self.edge_dim, self.hidden_dim)

        # cell
        self.W_cell_in = nn.Linear(self.node_dim + self.edge_dim, self.hidden_dim)
        self.W_cell_out = nn.Linear(self.node_dim + self.edge_dim, self.hidden_dim)
        self.U_cell_in = nn.Linear(self.node_dim + self.edge_dim, self.hidden_dim)
        self.U_cell_out = nn.Linear(self.node_dim + self.edge_dim, self.hidden_dim)


    def forward(self, batch_data):
        # indices: batch_size, node_num, neighbor_num_max
        # edges: batch_size, node_num, egde_labels

        node_reps, mask, in_indices, in_edges, in_mask, out_indices, out_edges, out_mask, _ = batch_data
        node_reps = self.dropout(node_reps)
        batch_size = node_reps.size(0)
        node_num_max = node_reps.size(1)

        # ==== input from in neighbors
        # [batch_size, node_num, neighbor_num_max, edge_dim]
        in_edge_reps = self.edge_embedding(in_edges)
        # [batch_size, node_num, neighbor_num_max, node_dim]
        in_node_reps = self.collect_neighbors(node_reps, in_indices)
        # [batch_size, node_num, neighbor_num_max, node_dim + edge_dim]
        in_reps = torch.cat([in_node_reps, in_edge_reps], 3)

        in_reps = in_reps.mul(in_mask.unsqueeze(-1))
        # [batch_size, node_num, word_dim + edge_dim]
        in_reps = in_reps.sum(dim=2)
        in_reps = in_reps.reshape([-1, self.node_dim + self.edge_dim])

        # ==== input from out neighbors
        # [batch_size, node_num, neighbor_num_max, edge_dim]
        out_edge_reps = self.edge_embedding(out_edges)
        # [batch_size, node_num, neighbor_num_max, node_dim]
        out_node_reps = self.collect_neighbors(node_reps, out_indices)
        # [batch_size, node_num, neighbor_num_max, node_dim + edge_dim]
        out_reps = torch.cat([out_node_reps, out_edge_reps], 3)

        out_reps = out_reps.mul(out_mask.unsqueeze(-1))
        # [batch_size, node_num, word_dim + edge_dim]
        out_reps = out_reps.sum(2)
        out_reps = out_reps.reshape([-1, self.node_dim + self.edge_dim])

        node_hidden = node_reps
        node_cell = torch.zeros(batch_size, node_num_max, self.hidden_dim).to(self.args.device)

        # node_reps = node_reps.reshape([-1, self.word_dim])

        graph_representations = []
        for i in range(self.gnn_layers):
            # in neighbor hidden
            # [batch_size, node_num, neighbor_num_max, node_dim + edge_dim]
            in_pre_hidden = self.collect_neighbors(node_hidden, in_indices)
            in_pre_hidden = torch.cat([in_pre_hidden, in_edge_reps], 3)
            in_pre_hidden = in_pre_hidden.mul(in_mask.unsqueeze(-1))
            # [batch_size, node_num, u_input_dim]
            in_pre_hidden = in_pre_hidden.sum(2)
            in_pre_hidden = in_pre_hidden.reshape([-1, self.node_dim + self.edge_dim])

            # out neighbor hidden
            # [batch_size, node_num, neighbors_size_max, node_dim + edge_dim]
            out_pre_hidden = self.collect_neighbors(node_hidden, out_indices)
            out_pre_hidden = torch.cat([out_pre_hidden, out_edge_reps], 3)
            out_pre_hidden = out_pre_hidden.mul(out_mask.unsqueeze(-1))
            # [batch_size, node_num, node_dim + edge_dim]
            out_pre_hidden = out_pre_hidden.sum(2)
            out_pre_hidden = out_pre_hidden.reshape([-1, self.node_dim + self.edge_dim])

            # in gate
            edge_ig = torch.sigmoid(self.W_ig_in(in_reps)
                                    + self.U_ig_in(in_pre_hidden)
                                    + self.W_ig_out(out_reps)
                                    + self.U_ig_out(out_pre_hidden))
            edge_ig = edge_ig.reshape([batch_size, node_num_max, self.hidden_dim])

            # forget gate
            edge_fg = torch.sigmoid(self.W_fg_in(in_reps)
                                    + self.U_fg_in(in_pre_hidden)
                                    + self.W_fg_out(out_reps)
                                    + self.U_fg_out(out_pre_hidden))
            edge_fg = edge_fg.reshape([batch_size, node_num_max, self.hidden_dim])

            # out gate
            edge_og = torch.sigmoid(self.W_og_in(in_reps)
                                    + self.U_og_in(in_pre_hidden)
                                    + self.W_og_out(out_reps)
                                    + self.U_og_out(out_pre_hidden))
            edge_og = edge_og.reshape([batch_size, node_num_max, self.hidden_dim])

            # input
            edge_cell_input = torch.tanh(self.W_cell_in(in_reps)
                                         + self.U_cell_in(in_pre_hidden)
                                         + self.W_cell_out(out_reps)
                                         + self.U_cell_out(out_pre_hidden))
            edge_cell_input = edge_cell_input.reshape([batch_size, node_num_max, self.hidden_dim])

            temp_cell = edge_fg * node_cell + edge_ig * edge_cell_input
            temp_hidden = edge_og * torch.tanh(temp_cell)

            node_cell = temp_cell.mul(mask.unsqueeze(-1))
            node_hidden = temp_hidden.mul(mask.unsqueeze(-1))

            graph_representations.append(node_hidden)

        return graph_representations, node_hidden, node_cell

    def collect_neighbors(self, node_reps, index):
        # node_rep: [batch_size, node_num, node_dim]
        # index: [batch_size, node_num, neighbors_num]
        batch_size = index.size(0)
        node_num = index.size(1)
        neighbor_num = index.size(2)
        rids = torch.arange(0, batch_size).to(self.args.device)  # [batch]
        rids = rids.reshape([-1, 1, 1])  # [batch, 1, 1]
        rids = rids.repeat(1, node_num, neighbor_num)  # [batch, nodes, neighbors]
        indices = torch.stack((rids, index), 3)  # [batch, nodes, neighbors, 2]
        return node_reps[indices[:, :, :, 0], indices[:, :, :, 1], :]

if __name__ == '__main__':

    # temp args: debug
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.device = torch.device('cpu')
    args.gnn_layer_num = 7
    args.gnn_dropout = 0.5
    args.batch_size = 2
    args.node_num_max = 3
    args.embed_dim = 5
    args.gnn_hidden_dim = 3
    args.edge_vocab_size = 6

    # node
    x = torch.ones((args.batch_size, args.node_num_max, args.embed_dim)).to(args.device)
    mask = [[1, 1, 1],
            [1, 0, 0]]
    mask = torch.tensor(mask).to(args.device)

    # in and out
    in_index = torch.tensor([[[-1, -1], [0, -1], [0, 1]],
                             [[-1, -1], [0, -1], [-1, -1]]]).to(args.device)
    in_edges = torch.tensor([[[0, 0], [1, 0], [2, 3]],
                             [[0, 0], [1, 0], [0, 0]]]).to(args.device)
    in_mask = torch.tensor([[[0, 0], [1, 0], [1, 1]],
                            [[0, 0], [1, 0], [0, 0]]]).to(args.device)

    out_index = torch.tensor([[[1, 2], [2, -1], [-1, -1]],
                              [[1, -1], [-1, -1], [-1, -1]]]).to(args.device)
    out_edges = torch.tensor([[[1, 2], [3, 0], [0, 0]],
                              [[1, 0], [1, 0], [0, 0]]]).to(args.device)
    out_mask = torch.tensor([[[1, 1], [1, 0], [0, 0]],
                             [[1, 0], [0, 0], [0, 0]]]).to(args.device)
    data = [x, mask, in_index, in_edges, in_mask, out_index, out_edges, out_mask, x]

    grn = GRN(args).to(args.device)
    graph_representations, node_hidden, node_cell = grn(data)
    print('graph reps\n', graph_representations)
    print('node hidden\n', node_hidden)
    print('node cell\n', node_cell)
