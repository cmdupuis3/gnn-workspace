
import numpy as np
import xarray as xr
from itertools import product

def contract(vars: xr.Dataset, mask: xr.Dataset, halo: int) -> xr.Dataset:
    latmax = mask.sizes['nlat']
    lonmax = mask.sizes['nlon']

    latrange = range(halo, latmax - halo)
    lonrange = range(halo, lonmax - halo)

    stencil = {'nlat': latrange, 'nlon': lonrange}

    sub = vars[stencil]
    # stencilled = {k: v.to_numpy() for k, v in sub.items()}
    # return stencilled
    return sub


def convolve(vars: xr.Dataset, mask: xr.Dataset, kernel: xr.DataArray) -> xr.Dataset:
    '''
    Assumes kernel is a NxN matrix where N is odd
    '''

    latmax = mask.sizes['nlat']
    lonmax = mask.sizes['nlon']

    halo_lat = int(xr.apply_ufunc(np.floor, kernel.sizes['nlat'] / 2.))
    halo_lon = int(np.floor(kernel.sizes['nlon'] / 2.))

    latrange = range(halo_lat, latmax - halo_lat)
    lonrange = range(halo_lon, lonmax - halo_lon)

    conved = dict()
    for k, v in vars.items():

        cvar = xr.zeros_like(v[latrange, lonrange])
        for i, j in product(latrange, lonrange):
            stencil = {"nlat": slice(i - halo_lat, i + halo_lat + 1), "nlon": slice(j - halo_lon, j + halo_lon + 1)}
            msub = mask[stencil]

            denominator = xr.apply_ufunc(np.abs, kernel * msub).sum(dim=['nlat', 'nlon'])
            if denominator == 0:  # Should these be NaNs?
                continue

            vsub = v[stencil]
            numerator = (kernel * msub * vsub).sum(dim=['nlat', 'nlon'])
            if np.isclose(numerator, 0.):  # check tolerance vs. variable values
                cvar[i - halo_lat, j - halo_lon] = 0.
            else:
                cvar[i - halo_lat, j - halo_lon] = numerator / denominator

        conved.update({k: cvar})

    return conved