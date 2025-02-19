import numpy as np
import xarray as xr
import xbatcher as xb

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from batching import rolling_batcher, batch_generator
from surface_currents_prep import *
from scenario import Scenario, sc5
from models import MsgModelDiff, ModelLikeAnirbans, get_halo_mask, remove_halo

def ez_plot(var, min, max, filename):
    plt.figure(figsize=(16, 10))
    xr.plot.contourf(var, levels = 100, vmin=min, vmax=max)
    plt.savefig(filename)


def predict(model, ds_predict, nbatches=58, batch_size=32):

    predict_batch = rolling_batcher(ds_predict, 7, 7)

    U_pred = np.full(ds_predict['U'].shape, np.nan)
    V_pred = np.full(ds_predict['V'].shape, np.nan)

    loss_fn = nn.MSELoss()
    for c, f, t, co in batch_generator(predict_batch, batch_size, nbatches):
        for convs, features, targets, coords in zip(c, f, t, co):

            halo = get_halo_mask(coords)
            features, targets, edges, weights = remove_halo(halo, features, targets)

            predictions_graph = model(convs.x.float(), features.float(), edges, weights, halo)
            batch_loss = loss_fn(predictions_graph, targets)

            for ct, node in enumerate(predictions_graph):
                nlat, nlon = coords[ct]

                U_pred[nlat, nlon] = node[0]
                V_pred[nlat, nlon] = node[1]

        print(f'[Batch Loss: {batch_loss}')

    U_pred = xr.DataArray(U_pred, dims=['nlat', 'nlon'])
    V_pred = xr.DataArray(V_pred, dims=['nlat', 'nlon'])
    U_diff = U_pred - ds_predict['U']
    V_diff = V_pred - ds_predict['V']

    ez_plot(ds_predict['U'], -20, 20, 'C:/Users/cdupu/Documents/model_A1_U.png')
    ez_plot(ds_predict['V'], -20, 20, 'C:/Users/cdupu/Documents/model_A1_V.png')
    ez_plot(U_pred, -20, 20, 'C:/Users/cdupu/Documents/model_A1_U_pred.png')
    ez_plot(V_pred, -20, 20, 'C:/Users/cdupu/Documents/model_A1_V_pred.png')
    ez_plot(U_diff, -100, 100, 'C:/Users/cdupu/Documents/model_A1_U_diff.png')
    ez_plot(V_diff, -100, 100, 'C:/Users/cdupu/Documents/model_A1_V_diff.png')


if __name__ == '__main__':

    load_path = "C:/Users/cdupu/Documents/gnn_model_A1.pt"
    model = ModelLikeAnirbans(5, [40,20,10], 2, num_conv=2, num_conv_channels=40, message_multiplier=2)
    model.load_state_dict(torch.load(load_path))

    ds_predict = load_predict_data(sc5)
    ds_predict = just_the_data(ds_predict)
    #ds_predict = select_from(ds_predict)

    #predict(model, ds_predict, batch_size=64, nbatches=464)
    predict(model, ds_predict, batch_size=512, nbatches=120)







