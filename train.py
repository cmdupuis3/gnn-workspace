import numpy as np
import xarray as xr
import xbatcher as xb

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from surface_currents_prep import *
from scenario import Scenario, sc5
from models import MsgModelDiff, ModelLikeAnirbans, StencilBounds, \
                   get_halo_mask # , get_halo_edge_mask

from batching import rolling_batcher, batch_generator


def train(model, ds_training, ds_testing,
          num_epochs=1, nbatches=58, batch_size=32, plot_loss=False):

    training_batch = rolling_batcher(ds_training, 7, 7)
    testing_batch  = rolling_batcher(ds_testing,  7, 7)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    testing_loss = []
    for epoch in range(num_epochs):
        for c, f, t, co in batch_generator(training_batch, batch_size, nbatches):
            for convs, features, targets, coords in zip(c, f, t, co):

                optimizer.zero_grad()

                lat_indices, lon_indices = [], []
                [(lat_indices.append(lat), lon_indices.append(lon)) for lat, lon in coords]
                Bounds = StencilBounds(np.min(lat_indices),
                                       np.max(lat_indices),
                                       np.min(lon_indices),
                                       np.max(lon_indices))

                halo = get_halo_mask(lat_indices, lon_indices, Bounds)
                featuresX = [x for i, x in enumerate(features.x) if halo[i]]
                targetsX  = [x for i, x in enumerate(targets.x)  if halo[i]]
                featuresX = torch.stack(featuresX)
                targetsX  = torch.stack(targetsX)

                # We have to find the edge halos also, because they are different

                halo_nodes = [i for i, e in enumerate(halo) if not e]
                fet = torch.transpose(features.edge_index, 0, 1)
                edge_halo = [(0 == halo_nodes.count(x)) & (0 == halo_nodes.count(y)) for x, y in fet]

                edges   = [x for i, x in enumerate(fet) if edge_halo[i]]
                weights = [x for i, x in enumerate(features.weight) if edge_halo[i]]
                edges   = torch.stack(edges, 0)
                weights = torch.stack(weights, 0)
                edges   = torch.transpose(edges, 0, 1)


                outs = model(convs.x.float(), featuresX.float(), edges, weights, halo)
                loss = loss_fn(outs, targetsX)
                loss.backward()
                optimizer.step()

        num_batches = 0
        epoch_loss = 0.0
        for c, f, t, co in batch_generator(testing_batch, batch_size):
            for convs, features, targets, coords in zip(c, f, t, co):

                outs = model(convs.x.float(), features.x.float(), features.edge_index, features.weight, coords)
                batch_loss = loss_fn(outs, targets.x)

            num_batches = num_batches + 1
            epoch_loss = epoch_loss + batch_loss
            # print(f'[Batch Loss: {batch_loss}')

        epoch_loss = epoch_loss / num_batches
        print(f'[\tEpoch Loss: {epoch_loss}')
        testing_loss.append(epoch_loss.item())

    if(plot_loss):
        plt.figure(figsize=(18, 5))
        plt.plot(range(num_epochs), testing_loss, color='#ff6347', label="loss")
        plt.plot(epoch, testing_loss[-1], marker = 'o', markersize=10, color='#ff6347')
        plt.legend()
        plt.xlabel(r'Epoch')
        plt.ylabel('Loss')
        plt.ylim([10., (1.3 * testing_loss[0])])
        plt.yscale("log")
        plt.savefig('C:/Users/cdupu/Documents/gnn_training_loss.png')


if __name__ == '__main__':

    ds_training = load_training_data(sc5)
    ds_training = just_the_data(ds_training)
    ds_training = select_from(ds_training)

    ds_testing = load_test_data(sc5)
    ds_testing = just_the_data(ds_testing)
    ds_testing = select_from(ds_testing)

    model = ModelLikeAnirbans(5, [40,20,10], 2, num_conv=2, num_conv_channels=40, message_multiplier=2)
    train(model, ds_training, ds_testing, num_epochs=30, batch_size=64, plot_loss=True)

    save_path = "C:/Users/cdupu/Documents/gnn_model4.pt"
    torch.save(model.state_dict(), save_path)