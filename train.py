import numpy as np
import xarray as xr
import xbatcher as xb

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from surface_currents_prep import *
from scenario              import Scenario
from models                import GCN, MsgModelDiff
from batching              import rolling_batcher, batch_generator

sc1 = Scenario(['SSH'],             ['TAUX', 'TAUY'], ['U', 'V'], name = "derp")
sc5 = Scenario(['SSH', 'SST'], ['X', 'TAUX', 'TAUY'], ['U', 'V'], name = "herp")

###############################################

ds_training = load_training_data(sc5)
ds_training = just_the_data(ds_training)
ds_training = select_from(ds_training)

ds_testing = load_test_data(sc5)
ds_testing = just_the_data(ds_testing)
ds_testing = select_from(ds_testing)


def train(model, num_epochs=1, batch_size=32, plot_loss=False):
    training_batch = rolling_batcher(ds_training, 9, 9)
    testing_batch = rolling_batcher(ds_testing, 9, 9)

    # Set up the loss and the optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    testing_loss = []
    for epoch in range(num_epochs):
        for c, f, t in batch_generator(training_batch, batch_size):
            for convs, features, targets in zip(c, f, t):

                optimizer.zero_grad()
                outs = model(convs.x.float(), features.x.float(), features.edge_index, features.weight)
                loss = loss_fn(outs, targets.x)
                loss.backward()
                optimizer.step()

        num_batches = 0
        epoch_loss = 0.0
        for c, f, t in batch_generator(testing_batch, batch_size):
            for convs, features, targets in zip(c, f, t):

                outs = model(convs.x.float(),features.x.float(), features.edge_index, features.weight)
                batch_loss = loss_fn(outs, targets.x)

            num_batches = num_batches + 1
            epoch_loss = epoch_loss + batch_loss
            # print(f'[Batch Loss: {batch_loss}')

        epoch_loss = epoch_loss / num_batches
        print(f'[\tEpoch Loss:\n {epoch_loss}')
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

    model = MsgModelDiff(5, [40,20,10,5], 2, num_conv=2, num_conv_channels=40, message_multiplier=2)

    train(model, num_epochs=30, batch_size=64, plot_loss=True)

    save_path = "C:/Users/cdupu/Documents/gnn_model2.pt"
    torch.save(model.state_dict(), save_path)