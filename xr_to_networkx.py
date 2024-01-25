
import numpy as np
import networkx as nx
import xarray as xr

def _point_to_graph(mask, i, j, imax, jmax, weight=1.0):
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
    # TODO: can we use fancy indexing to avoid these loops
    for i in range(0, imax + 1):
        for j in range(0, jmax + 1):
            if mask[i, j]:
                edges = _point_to_graph(mask, i, j, imax, jmax)
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


def _xr_ds_to_graph(batch, sc, kernel):
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

    features_graph = _graph_builder(mask, batch, sc.conv_var + sc.input_var)
    targets_graph = _graph_builder(mask, batch, sc.target)

    return features_graph, targets_graph

def xr_to_graphs(batch, sc, kernel):
    f, t = _xr_ds_to_graph(batch, sc, kernel)
    fsub = [f.subgraph(sub) for sub in nx.connected_components(f)]
    tsub = [t.subgraph(sub) for sub in nx.connected_components(t)]
    return fsub, tsub
