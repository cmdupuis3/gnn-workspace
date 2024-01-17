import numpy as np
import xarray as xr
from intake import open_catalog

def add_grid(ds):
    X = lambda lat: np.sin(np.radians(lat))
    Y = lambda lat, lon: np.cos(np.radians(lat)) * np.sin(np.radians(lon))
    Z = lambda lat, lon: -np.cos(np.radians(lat)) * np.cos(np.radians(lon))
    delta = lambda dx, dxMean, dxStd: (dx - dxMean) / dxStd

    lats = ds.YU.data
    lons = ds.XU.data
    DX = ds.DXT.data
    DY = ds.DYT.data

    x = X(lats)
    y = Y(lats, lons)
    z = Z(lats, lons)
    dX = delta(DX, np.mean(DX), np.std(DX))
    dY = delta(DY, np.mean(DY), np.std(DY))

    ds['X'] = ds.XU.dims, x
    ds['Y'] = ds.XU.dims, y
    ds['Z'] = ds.XU.dims, z
    ds['dx'] = ds.XU.dims, dX
    ds['dy'] = ds.XU.dims, dY

    return (ds)




def prepare_data(sc, training_time, test_time, predict_time, mask_time=11):
    cat = open_catalog("https://raw.githubusercontent.com/pangeo-data/pangeo-datastore/master/intake-catalogs/ocean/CESM_POP.yaml")
    ds = cat["CESM_POP_hires_control"].to_dask()
    ds = ds.rename({'U1_1': 'U', 'V1_1': 'V', 'TAUX_2': 'TAUX', 'TAUY_2': 'TAUY', 'SSH_2': 'SSH', 'ULONG': 'XU', 'ULAT': 'YU'})
    ds = add_grid(ds)

    def get_mask_from(ds, x):
        return ~np.isnan(ds[x])

    mask1 = get_mask_from(ds, 'SSH')[{'time': mask_time}]
    mask2 = get_mask_from(ds, 'TAUY')[{'time': mask_time}]
    mask3 = get_mask_from(ds, 'U')[{'time': mask_time}]
    mask = mask1 & mask2 & mask3
    mask = mask.compute()

    varList = sc.conv_var + sc.input_var + sc.target
    ds = ds[varList]
    ds['mask'] = mask

    ds_training = ds.isel(time=training_time)
    ds_training.to_zarr('scenarios/', group='training_' + sc.name + '.zarr', mode='w')
    del ds_training

    ds_test = ds.isel(time=test_time)
    ds_test.to_zarr('scenarios/', group='test_' + sc.name + '.zarr', mode='w')
    del ds_test

    ds_predict = ds.isel(time=predict_time)
    ds_predict.to_zarr('scenarios/', group='predict_' + sc.name + '.zarr', mode='w')
    del ds_predict

    pass



def loader(sc, name):
    # ds = xr.open_dataset(path, engine='zarr')
    ds = xr.open_zarr('C:/Users/cdupu/Downloads/scenarios/scenarios/',
                       group = name + '_' + sc.name + '.zarr',
                       decode_times=False)
    return ds

def load_training_data(sc):
    return loader(sc, 'training')

def load_test_data(sc):
    return loader(sc, 'test')

def load_predict_data(sc):
    return loader(sc, 'predict')




def just_the_data(ds):
    ds = ds.drop_vars(list(ds.coords.keys()))

    for attr in list(ds.attrs.keys()):
        del ds.attrs[attr]

    for var in list(ds.variables.keys()):
        for attr in list(ds[var].attrs.keys()):
            del ds[var].attrs[attr]
    return ds

def select_from(ds):
    return ds.isel(nlon=list(range(1850,2200)), nlat=list(range(500,1050)))