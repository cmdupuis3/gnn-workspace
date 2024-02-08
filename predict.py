import numpy as np
import xarray as xr
import xbatcher as xb

import torch
import torch.nn as nn
from torch_geometric.utils.convert import to_networkx, from_networkx

import matplotlib.pyplot as plt



from surface_currents_prep import *
from scenario              import Scenario
from xr_to_networkx        import xr_to_graphs, graphs_to_xr

def predict():


if __name__ == '__main__':

    load_path = "C:/Users/cdupu/Documents/gnn_model2.pt"
    model = torch.load(load_path)

    sc5 = Scenario(['SSH', 'SST'], ['X', 'TAUX', 'TAUY'], ['U', 'V'], name="herp")
    ds_predict = load_predict_data(sc5)
    ds_predict = just_the_data(ds_predict)
    ds_predict = select_from(ds_predict)







