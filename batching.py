import numpy as np
import xarray as xr

from torch_geometric.utils.convert import to_networkx, from_networkx


from scenario              import Scenario
from xr_to_networkx        import xr_to_graphs, graphs_to_xr


sc5 = Scenario(['SSH', 'SST'], ['X', 'TAUX', 'TAUY'], ['U', 'V'], name = "herp")

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


def ggen_subgs(batch_set):
    for i in range(len(batch_set['input_batch'])):
        batch = batch_set[{'input_batch':i}]
        csub, fsub, tsub = xr_to_graphs(batch, sc5)

        for j in range(len(fsub)):
            cpy = from_networkx(csub[j], group_node_attrs = sc5.conv_var)
            fpy = from_networkx(fsub[j], group_node_attrs = sc5.input_var)
            tpy = from_networkx(tsub[j], group_node_attrs = sc5.target)
            yield (cpy, fpy, tpy)


def batch_generator(batch, batch_size):
    bgen = ggen_subgs(batch)
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