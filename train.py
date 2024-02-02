import numpy as np
import xarray as xr
import xbatcher as xb

import torch
import torch.nn as nn
from torch_geometric.utils.convert import to_networkx, from_networkx

import matplotlib.pyplot as plt

from surface_currents_prep import *
from scenario              import Scenario
from models                import GCN, MsgModelDiff
from xr_to_networkx        import xr_to_graphs, graphs_to_xr
from preconvolve           import *

sc1 = Scenario(['SSH'],             ['TAUX', 'TAUY'], ['U', 'V'], name = "derp")
sc5 = Scenario(['SSH', 'SST'], ['X', 'TAUX', 'TAUY'], ['U', 'V'], name = "herp")

###############################################

ds_training = load_training_data(sc5)
ds_training = just_the_data(ds_training)
ds_training = select_from(ds_training)

ds_testing = load_test_data(sc5)
ds_testing = just_the_data(ds_testing)
ds_testing = select_from(ds_testing)

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


training_batch = rolling_batcher(ds_training, 9, 9)
testing_batch  = rolling_batcher(ds_testing,  9, 9)

###############################################

kernel = xr.DataArray([[0,  1, 0],
                       [-1, 0, 1],
                       [0, -1, 0]],
                       dims=["nlat", "nlon"])

def ggen_subgs(batch_set, kernel):
    for i in range(len(batch_set['input_batch'])):
        batch = batch_set[{'input_batch':i}]
        csub, fsub, tsub = xr_to_graphs(batch, sc5, kernel)

        for j in range(len(fsub)):
            cpy = from_networkx(csub[j], group_node_attrs = sc5.conv_var)
            fpy = from_networkx(fsub[j], group_node_attrs = sc5.input_var)
            tpy = from_networkx(tsub[j], group_node_attrs = sc5.target)
            yield (cpy, fpy, tpy)


def batch_generator(batch, kernel, batch_size):
    bgen = ggen_subgs(batch, kernel)
    b = (batch for batch in bgen)
    n = 0
    convs = list()
    feats = list()
    targs = list()
    while n < 58:
        batch = [next(b) for i in range(batch_size)]
        convs = [batch[i][0] for i in range(batch_size)]
        feats = [batch[i][1] for i in range(batch_size)]
        targs = [batch[i][2] for i in range(batch_size)]

        yield convs, feats, targs
        n += 1

# import snakeviz, cProfile
# %load_ext snakeviz

# pr = cProfile.Profile()
# pr.enable()
# f, t = next(bgen)
# pr.disable()
# pr.dump_stats("C:/Users/cdupu/Downloads/stats.prof")


def train(model, num_epochs=1, batch_size=32, plot_loss=False):
    # Set up the loss and the optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    testing_loss = []
    for epoch in range(num_epochs):
        for c, f, t in batch_generator(training_batch, kernel, batch_size):
            for convs, features, targets in zip(c, f, t):

                optimizer.zero_grad()
                outs = model(convs.x.float(), features.x.float(), features.edge_index, features.weight)
                loss = loss_fn(outs, targets.x)
                loss.backward()
                optimizer.step()

        for c, f, t in batch_generator(testing_batch, kernel, batch_size):
            for convs, features, targets in zip(c, f, t):

                outs = model(convs.x.float(),features.x.float(), features.edge_index, features.weight)
                loss_temp = loss_fn(outs, targets.x)

        print(f'[\tEpoch Loss:\n {loss_temp}')
        testing_loss.append(loss_temp.item())

    if(plot_loss):
        plt.figure(figsize=(18, 5))
        plt.plot(range(num_epochs), testing_loss, color='#ff6347', label="loss")
        plt.plot(epoch, testing_loss[-1], marker = 'o', markersize=10, color='#ff6347')
        plt.legend()
        plt.xlabel(r'Epoch')
        plt.ylabel('Loss')
        plt.ylim([0, (1.3 * testing_loss[0])])
        plt.savefig('C:/Users/cdupu/Documents/gnn_training_loss.png')


###############################################

if __name__ == '__main__':
    model = MsgModelDiff(5, [40,20,10,5], 2, num_conv=2, num_conv_channels=80, num_message=100)

    train(model, num_epochs=30, batch_size=32, plot_loss=True)

    # import snakeviz, cProfile
    #
    # pr = cProfile.Profile()
    # pr.enable()
    # train(model, num_epochs=2, batch_size=64, plot_loss=True)
    # pr.disable()
    # pr.dump_stats("C:/Users/cdupu/Downloads/stats.prof")

    save_path = "C:/Users/cdupu/Documents/gnn_model.pt"
    torch.save(model.state_dict(), save_path)