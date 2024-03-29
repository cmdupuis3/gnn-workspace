
import numpy as np
import torch
import torch.nn.functional as F

from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing, GCNConv


def get_halo_mask(coords):
    """
    Simple halo ATM; need to implement coastline halos
    """
    ii, jj = [], []
    [(ii.append(i), jj.append(j)) for i, j in coords]
    imin = np.min(ii)
    imax = np.max(ii)
    jmin = np.min(jj)
    jmax = np.max(jj)

    imask = [(i == imin) | (i == imax) for i in ii]
    jmask = [(j == jmin) | (j == jmax) for j in jj]

    mask = [not (i | j) for i, j in zip(imask, jmask)]
    return (mask)

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
    def __init__(self, in_channels, out_channels, message_multiplier):
        super().__init__(aggr='add')
        self.lin_1 = Linear(in_channels, out_channels)
        self.lin_2 = Linear(in_channels, out_channels)
        self.mlp = Seq(Linear(2 * in_channels, 2 * in_channels * message_multiplier),
                       ReLU(),
                       Linear(2 * in_channels * message_multiplier, in_channels))

    def forward(self, x, edge_index, edge_attr):  # edge_attr
        out = self.propagate(x=x, edge_index=edge_index, edge_attr=edge_attr)
        out += self.lin_1(x)  # Using += here is like a skip-connection, as opposed to = (according to Alex)
        return out

    def message(self, x_i, x_j, edge_attr):  # edge_attr
        tmp = torch.cat([x_i, x_j], 1)  # edge_attr
        return self.lin_2(self.mlp(tmp) * (x_i - x_j))

class MsgModelDiff(torch.nn.Module):

    def __init__(self, num_in, num_channels, num_out,
                 num_conv=0, num_conv_channels=0,
                 message_multiplier=2):
        super().__init__()

        self.layer_conv = _MPD_in(num_conv, num_conv_channels, message_multiplier)

        self.layer_1 = _MPD_in(num_in - num_conv + num_conv_channels,
                               num_channels[0], message_multiplier)
        self.layer_2 = _MPD_in(num_channels[0], num_channels[1], message_multiplier)
        self.layer_3 = _MPD_in(num_channels[1], num_out, message_multiplier)
        # self.layer_4 = _MPD_in(num_channels[2], num_channels[3], message_multiplier)
        # self.layer_5 = _MPD_in(num_channels[3], num_out, message_multiplier)


    def forward(self, convs, features, edges, weights, coords=None):

        preconv = self.layer_conv(convs, edges, weights)
        x = torch.concat((features, preconv), 1) # TODO: Check concat dimension

        x = self.layer_1(x, edges, weights)
        x = torch.nn.ReLU()(x)
        x = self.layer_2(x, edges, weights)
        x = torch.nn.ReLU()(x)
        x = self.layer_3(x, edges, weights)
        # x = torch.nn.ReLU()(x)
        # x = self.layer_4(x, edges, weights)
        # x = torch.nn.ReLU()(x)
        # x = self.layer_5(x, edges, weights)
        return x

class ModelLikeAnirbans(torch.nn.Module):

    def __init__(self, num_in, num_channels, num_out,
                 num_conv=0, num_conv_channels=0,
                 message_multiplier=2):
        super().__init__()

        self.layer_conv = _MPD_in(num_conv, num_conv_channels, message_multiplier)

        self.layer_1 = Linear(num_in - num_conv + num_conv_channels, num_channels[0])
        self.layer_2 = Linear(num_channels[0], num_channels[1])
        self.layer_3 = Linear(num_channels[1], num_out)

    def forward(self, convs, features, edges, weights, coords):
        preconv = self.layer_conv(convs, edges, weights)
        grad_fn = preconv.grad_fn

        halo = get_halo_mask(coords)
        preconv = [x for i, x in enumerate(preconv) if halo[i]]
        preconv = torch.stack(preconv)
        print(preconv.shape)

        x = torch.concat((features, preconv), 1)  # TODO: Check concat dimension
        x = self.layer_1(x)
        x = torch.nn.ReLU()(x)
        x = self.layer_2(x)
        x = torch.nn.ReLU()(x)
        x = self.layer_3(x)
        return x