import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import networkx as nx

from dataclasses import dataclass
from typing import Iterable
from itertools import product
from intake import open_catalog

# Need a cruft check on these

import torch
import torch.nn as nn
import torch.utils.data as data
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Sequential as Seq, Linear, ReLU

from torch_geometric.utils.convert import to_networkx, from_networkx

import xbatcher as xb

from surface_currents_prep import *
from scenario import Scenario, sc5
######


ds_training = load_training_data(sc5)
ds_training = just_the_data(ds_training)
ds_training = select_from(ds_training)



def point_to_graph(mask, i, j, imax, jmax, weight=1.0):
    edges = []

    if not (j == jmax):
        if mask[i, j+1]:
            edges.append([(i,j), (i,j+1), weight])

    if not (i == imax):
        if mask[i+1,j]:
            edges.append([(i,j), (i+1,j), weight])

    return edges


def np_array_to_graph(array):
    graph = nx.Graph()
    mask = ~np.isnan(array)

    imax = array.shape[0] - 1
    jmax = array.shape[1] - 1

    for i in range(0, imax + 1):
        for j in range(0, jmax + 1):
            if mask[i, j]:
                graph.add_node((i, j))
                edges = point_to_graph(mask, i, j, imax, jmax)
                graph.add_weighted_edges_from(edges)

    return graph


def np_mask_to_graph(mask):
    array = np.zeros(mask.shape)
    for i in range(0, array.shape[0]):
        for j in range(0, array.shape[1]):
            if not mask[i, j]:
                array[i, j] = np.nan
    return np_array_to_graph(array)

def np_array_to_subgraphs(array):
    g = np_array_to_graph(array)
    return [g.subgraph(sub) for sub in nx.connected_components(g)]

def np_mask_to_subgraphs(mask):
    g = np_mask_to_graph(mask)
    return [g.subgraph(sub) for sub in nx.connected_components(g)]



def rolling_batcher(ds, nlats = 5, nlons = 5, halo_size=1):

    latlen = len(ds_training['nlat'])
    lonlen = len(ds_training['nlon'])
    nlon_range = range(nlons,lonlen,nlons - 2*halo_size)
    nlat_range = range(nlats,latlen,nlats - 2*halo_size)

    batch = (
        ds_training
        .rolling({"nlat": nlats, "nlon": nlons})
        .construct({"nlat": "nlat_input", "nlon": "nlon_input"})[{'nlat':nlat_range, 'nlon':nlon_range}]
        .stack({"input_batch": ("nlat", "nlon")}, create_index=False)
        .rename_dims({'nlat_input':'nlat', 'nlon_input':'nlon'})
        .transpose('input_batch',...)
        .dropna('input_batch')
    ).compute()

    rnds = list(range(len(batch['input_batch'])))
    np.random.shuffle(rnds)
    batch = batch[{'input_batch':(rnds)}]
    return batch


batch = rolling_batcher(ds_training, 15, 15)
batch2 = batch[{'input_batch':0}]


def graph_builder(mask: np.ndarray, vars: xr.Dataset, names: list[str]) -> nx.Graph:
    imax = mask.shape[0] - 1
    jmax = mask.shape[1] - 1

    vars_graph = nx.Graph()
    # TODO: can we use fancy indexing to avoid these loops
    for i in range(0, imax + 1):
        for j in range(0, jmax + 1):
            if mask[i, j]:
                edges = point_to_graph(mask, i, j, imax, jmax)
                # vars_ij = {x:vars[x][i,j] for x in names} #vars[x][i,j].to_numpy()
                # TODO: double check c vs. fortran indexing conventions
                vars_sub = vars[names].isel(nlon=i, nlat=j)
                # vars_sub is an xarray dataset; it has the names in it already
                node_data = {vname: vars_sub[vname].values.item() for vname in vars_sub}
                vars_graph.add_node((i, j), **node_data)
                vars_graph.add_weighted_edges_from(edges, )

                # add self-loops
                # vars_graph.add_edge((i,j), (i,j), weight=1.)

    return vars_graph

from preconvolve import *
kernel = xr.DataArray([[0,  1, 0],
                       [-1, 0, 1],
                       [0, -1, 0]],
                       dims=["nlat", "nlon"])

def xr_ds_to_graph(batch, sc, kernel):
    batch.load()
    mask = batch['mask'].values

    # convolve and contract here
    # conv_f   = convolve(batch[sc.conv_var],  mask, kernel)
    # features = contract(batch[sc.input_var], mask, 1)
    # targets  = contract(batch[sc.target],    mask, 1)
    # mask     = contract(batch[['mask']],     mask, 1)['mask'] # silly, but it works

    # if we don't want an initial convolution...
    # conv_f   = {v:batch[v].to_numpy() for v in sc.conv_var}
    # features = {v:batch[v].to_numpy() for v in sc.input_var}
    # targets  = {v:batch[v].to_numpy() for v in sc.target}

    # features = features | conv_f

    features_graph = graph_builder(mask, batch, sc.conv_var + sc.input_var)
    targets_graph = graph_builder(mask, batch, sc.target)

    return features_graph, targets_graph

def xr_ds_to_subgraphs(batch, sc, kernel):
    f, t = xr_ds_to_graph(batch, sc, kernel)
    fsub = [f.subgraph(sub) for sub in nx.connected_components(f)]
    tsub = [t.subgraph(sub) for sub in nx.connected_components(t)]
    return fsub, tsub


'''
This function is just a placeholder function for various graph convolution operators, which can be passed to ggen_subgs.
'''
def graph_conv(graph):
    return nx.laplacian_matrix(graph) * graph

def ggen_subgs(batch_set, kernel, gconv=graph_conv):
    for i in range(len(batch_set['input_batch'])):
        batch = batch_set[{'input_batch':i}]
        fsub, tsub = xr_ds_to_subgraphs(batch, sc5, kernel)

        for j in range(len(fsub)):
            fpy = from_networkx(fsub[j], group_node_attrs = sc5.conv_var + sc5.input_var)
            tpy = from_networkx(tsub[j], group_node_attrs = sc5.target)
            yield (fpy, tpy)


def batch_generator(bgen, batch_size):
    b = (batch for batch in bgen)
    n = 0
    feats = list()
    targs = list()
    while n < 10:
        batch = [next(b) for i in range(batch_size)]
        feats = [batch[i][0] for i in range(batch_size)]
        targs = [batch[i][1] for i in range(batch_size)]

        yield feats, targs
        n += 1


ggen = ggen_subgs(batch, kernel)
bgen = batch_generator(ggen_subgs(batch, kernel), 1024)

# import snakeviz, cProfile
# %load_ext snakeviz

# pr = cProfile.Profile()
# pr.enable()
f, t = next(bgen)
# pr.disable()
# pr.dump_stats("C:/Users/cdupu/Downloads/stats.prof")


class MPD_in(MessagePassing):
    def __init__(self, in_channels, out_channels, message_channels):
        super().__init__(aggr='add')
        self.lin_1 = Linear(in_channels, out_channels)
        self.lin_2 = Linear(in_channels, out_channels)
        self.mlp = Seq(Linear(2 * in_channels + 2, message_channels),
                       ReLU(),
                       Linear(message_channels, in_channels))

    def forward(self, x, edge_index, edge_attr): #edge_attr
        out = self.propagate(x=x, edge_index=edge_index, edge_attr=edge_attr) #edge_attr=edge_attr
        out += self.lin_1(x) # Using += here is like a skip-connection, as opposed to = (according to Alex)
        return out

    def message(self, x_i, x_j, edge_attr): # edge_attr
        tmp = torch.cat([x_i,x_j],1) #edge_attr
        return self.lin_2(self.mlp(tmp)*(x_i-x_j))

class MsgModelDiff(torch.nn.Module):

    def __init__(self, num_in = 1, num_out = 1, num_hidden = 30, num_message = 200):
        super().__init__()
        self.layer_diff = MPD_in(num_in,num_hidden,num_message)
        self.layer_h = MPD_in(num_hidden,num_hidden,num_message)
        self.layer_out = MPD_in(num_hidden,num_out,num_message)
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



import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
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



def train(model, num_epochs=1, batch_size=32):
    # Set up the loss and the optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(num_epochs):
        for f, t in batch_generator(ggen_subgs(batch, kernel), batch_size):
            for features, targets in zip(f, t):

                optimizer.zero_grad()
                outs = model(features.x.float(), features.edge_index, features.weight)
                loss = loss_fn(outs, targets.x)
                loss.backward()
                optimizer.step()

        print(f'[Loss: {loss}')


model = MsgModelDiff(num_in=2, num_hidden=40, num_message=100)
model2 = GCN(5, 2, 2) # BUG CHECK: Does this work for the right reason?

train(model2, num_epochs=20, batch_size=64)