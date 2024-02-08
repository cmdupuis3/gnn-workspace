
import numpy as np
import networkx as nx
import xarray as xr

from preconvolve import *
from scenario import Scenario


sc5 = Scenario(['SSH', 'SST'], ['X', 'TAUX', 'TAUY'], ['U', 'V'], name = "herp")

def _point_to_graph(mask: np.ndarray, i: int, j: int, imax: int, jmax: int, weight: float =1.0):
    edges = []

    if not (j == jmax):
        if mask[i, j+1]:
            edges.append([(i,j), (i,j+1), weight])

    if not (i == imax):
        if mask[i+1,j]:
            edges.append([(i,j), (i+1,j), weight])

    return edges


def _np_array_to_graph(array):
    graph = nx.Graph()
    mask = ~np.isnan(array)

    imax = array.shape[0] - 1
    jmax = array.shape[1] - 1

    for i in range(0, imax + 1):
        for j in range(0, jmax + 1):
            if mask[i, j]:
                graph.add_node((i, j))
                edges = _point_to_graph(mask, i, j, imax, jmax)
                graph.add_weighted_edges_from(edges)

    return graph


def _np_mask_to_graph(mask):
    array = np.zeros(mask.shape)
    for i in range(0, array.shape[0]):
        for j in range(0, array.shape[1]):
            if not mask[i, j]:
                array[i, j] = np.nan
    return _np_array_to_graph(array)

def np_array_to_graphs(array):
    g = _np_array_to_graph(array)
    return [g.subgraph(sub) for sub in nx.connected_components(g)]

def np_mask_to_graphs(mask):
    g = _np_mask_to_graph(mask)
    return [g.subgraph(sub) for sub in nx.connected_components(g)]


def _graph_builder(mask: np.ndarray, vars: xr.Dataset, names: list[str]) -> nx.Graph:
    imax = mask.shape[0] - 1
    jmax = mask.shape[1] - 1

    vars_graph = nx.Graph()

    for j in range(0, jmax + 1):
        for i in range(0, imax + 1):
            if mask[i, j]:
                edges = _point_to_graph(mask, i, j, imax, jmax)
                # TODO: double check c vs. fortran indexing conventions
                vars_sub = vars[names].isel(nlon=i, nlat=j)
                node_data = {vname: vars_sub[vname].values.item() for vname in vars_sub}
                vars_graph.add_node((i, j), **node_data)
                vars_graph.add_weighted_edges_from(edges, )

                # add self-loops
                # vars_graph.add_edge((i,j), (i,j), weight=1.)

    return vars_graph


def _xr_ds_to_graph(batch, sc):
    batch.load()
    mask = batch['mask']

    conv_f_graph   = _graph_builder(mask.values, batch, sc.conv_var)
    features_graph = _graph_builder(mask.values, batch, sc.input_var)
    targets_graph  = _graph_builder(mask.values, batch, sc.target)

    return conv_f_graph, features_graph, targets_graph

def xr_to_graphs(batch, sc):
    c, f, t = _xr_ds_to_graph(batch, sc)
    csub = [c.subgraph(sub) for sub in nx.connected_components(c)]
    fsub = [f.subgraph(sub) for sub in nx.connected_components(f)]
    tsub = [t.subgraph(sub) for sub in nx.connected_components(t)]
    return csub, fsub, tsub

def _graphs_to_array(ds: xr.Dataset, var_name: str, graphs):
    '''
    Converts an iterable of networkx graphs into a single xarray DataArray
    '''

    new_array = xr.full_like(ds[var_name], np.nan)

    for graph in graphs:
        for k, v in list(graph.nodes(data=True)):
            # k is the "name", i.e. the original xarray coordinates
            print(k)





def graphs_to_xr(ds: xr.Dataset, sc: Scenario, graphs):

    var_names = sc.conv_var | sc.features | sc.targets

    ds_out = xr.Dataset(
        data_vars = {k: _graphs_to_array(ds, k, graphs) for k in var_names},
        coords = ds.coords,
        # attrs = # Probably don't want to copy paste
    )

    return ds_out