import numpy as np
import xarray as xr
import xbatcher as xb

import torch
import torch.nn as nn
from torch_geometric.utils.convert import to_networkx, from_networkx

from surface_currents_prep import *
from scenario              import Scenario
from models                import GCN, MsgModelDiff
from xr_to_networkx        import xr_to_graphs
from preconvolve           import *

sc1 = Scenario(['SSH'],             ['TAUX', 'TAUY'], ['U', 'V'], name = "derp")
sc5 = Scenario(['SSH', 'SST'], ['X', 'TAUX', 'TAUY'], ['U', 'V'], name = "herp")

###############################################

ds_training = load_training_data(sc5)
ds_training = just_the_data(ds_training)
ds_training = select_from(ds_training)


def rolling_batcher(ds, nlats = 5, nlons = 5, halo_size=1):

    latlen = len(ds['nlat'])
    lonlen = len(ds['nlon'])
    nlon_range = range(nlons,lonlen,nlons - 2*halo_size)
    nlat_range = range(nlats,latlen,nlats - 2*halo_size)

    batch = (
        ds
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


batch = rolling_batcher(ds_training, 9, 9)
batch2 = batch[{'input_batch':0}]

###############################################

kernel = xr.DataArray([[0,  1, 0],
                       [-1, 0, 1],
                       [0, -1, 0]],
                       dims=["nlat", "nlon"])

def ggen_subgs(batch_set, kernel):
    for i in range(len(batch_set['input_batch'])):
        batch = batch_set[{'input_batch':i}]
        fsub, tsub = xr_to_graphs(batch, sc5, kernel)

        for j in range(len(fsub)):
            fpy = from_networkx(fsub[j], group_node_attrs = sc5.conv_var + sc5.input_var)
            tpy = from_networkx(tsub[j], group_node_attrs = sc5.target)
            yield (fpy, tpy)


def batch_generator(batch, kernel, batch_size):
    bgen = ggen_subgs(batch, kernel)
    b = (batch for batch in bgen)
    n = 0
    feats = list()
    targs = list()
    while n < 30:
        batch = [next(b) for i in range(batch_size)]
        feats = [batch[i][0] for i in range(batch_size)]
        targs = [batch[i][1] for i in range(batch_size)]

        yield feats, targs
        n += 1

ggen = ggen_subgs(batch, kernel)
bgen = batch_generator(batch, kernel, 1024)

# import snakeviz, cProfile
# %load_ext snakeviz

# pr = cProfile.Profile()
# pr.enable()
# f, t = next(bgen)
# pr.disable()
# pr.dump_stats("C:/Users/cdupu/Downloads/stats.prof")


def train(model, num_epochs=1, batch_size=32):
    # Set up the loss and the optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(num_epochs):
        for f, t in batch_generator(batch, kernel, batch_size):
            for features, targets in zip(f, t):

                optimizer.zero_grad()
                outs = model(features.x.float(), features.edge_index, features.weight)
                loss = loss_fn(outs, targets.x)
                loss.backward()
                optimizer.step()
            # print(f'[Batch Loss: {loss}')

        print(f'[\tEpoch Loss: {loss}')

###############################################

if __name__ == '__main__':
    model = MsgModelDiff(num_in=5, num_out=2, num_message=100)
    model2 = GCN(5, 2)

    # train(model, num_epochs=20, batch_size=64)

    # import snakeviz, cProfile
    #
    # pr = cProfile.Profile()
    # pr.enable()
    train(model, num_epochs=2, batch_size=64)
    # pr.disable()
    # pr.dump_stats("C:/Users/cdupu/Downloads/stats.prof")