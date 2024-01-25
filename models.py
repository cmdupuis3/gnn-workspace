
import torch
import torch.nn.functional as F

from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing, GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 20)
        self.conv2 = GCNConv(20, 20)
        self.conv3 = GCNConv(20, 10)
        self.conv4 = GCNConv(10, out_channels)

    def forward(self, features, edges, weights):
        x = self.conv1(features, edges, weights)
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x = self.conv2(x, edges, weights)
        x = F.relu(x)
        x = self.conv2(x, edges, weights)
        x = F.relu(x)
        x = self.conv3(x, edges, weights)
        x = F.relu(x)
        x = self.conv4(x, edges, weights)

        return F.log_softmax(x, dim=1)

class _MPD_in(MessagePassing):
    def __init__(self, in_channels, out_channels, message_channels):
        super().__init__(aggr='add')
        self.lin_1 = Linear(in_channels, out_channels)
        self.lin_2 = Linear(in_channels, out_channels)
        self.mlp = Seq(Linear(2 * in_channels, message_channels),
                       ReLU(),
                       Linear(message_channels, in_channels))

    def forward(self, x, edge_index, edge_attr):  # edge_attr
        out = self.propagate(x=x, edge_index=edge_index, edge_attr=edge_attr)  # edge_attr=edge_attr
        out += self.lin_1(x)  # Using += here is like a skip-connection, as opposed to = (according to Alex)
        return out

    def message(self, x_i, x_j, edge_attr):  # edge_attr
        tmp = torch.cat([x_i, x_j], 1)  # edge_attr
        return self.lin_2(self.mlp(tmp) * (x_i - x_j))

class MsgModelDiff(torch.nn.Module):

    def __init__(self, num_in=1, num_out=1, num_message=200):
        super().__init__()
        self.layer_diff = _MPD_in(num_in, 20, num_message)
        self.layer_h = _MPD_in(20, 20, num_message)
        self.layer_out = _MPD_in(20, num_out, num_message)
        self.relu = torch.nn.ReLU()

    def forward(self, features, edges, weights):
        x = self.layer_diff(features, edges, weights)
        x = self.relu(x)
        x = self.layer_h(x, edges, weights)
        x = self.relu(x)
        x = self.layer_h(x, edges, weights)
        x = self.relu(x)
        x = self.layer_h(x, edges, weights)
        x = self.relu(x)
        x = self.layer_out(x, edges, weights)
        return x

