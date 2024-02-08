import numpy as np
import xarray as xr
import xbatcher as xb

import torch
import torch.nn as nn
from torch_geometric.utils.convert import to_networkx, from_networkx

import matplotlib.pyplot as plt


from batching              import rolling_batcher, batch_generator
from surface_currents_prep import *
from scenario              import Scenario, sc5
from models                import MsgModelDiff
from xr_to_networkx        import xr_to_graphs, graphs_to_xr


def predict(model, ds_predict, num_epochs=1, batch_size=32, plot_loss=False):

    predict_batch = rolling_batcher(ds_predict, 9, 9)

    loss_fn = nn.MSELoss()

    for c, f, t in batch_generator(predict_batch, batch_size):
        for convs, features, targets in zip(c, f, t):

            predictions = model(convs.x.float(), features.x.float(), features.edge_index, features.weight)
            batch_loss = loss_fn(predictions, targets.x)

        print(f'[Batch Loss: {batch_loss}')


if __name__ == '__main__':

    load_path = "C:/Users/cdupu/Documents/gnn_model2.pt"
    model = MsgModelDiff(5, [40,20,10], 2, num_conv=2, num_conv_channels=40, message_multiplier=2)
    model.load_state_dict(torch.load(load_path))
    print(model)

    ds_predict = load_predict_data(sc5)
    ds_predict = just_the_data(ds_predict)
    ds_predict = select_from(ds_predict)

    predict(model, ds_predict)







