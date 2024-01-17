
import numpy as np
from itertools import product

def contract(vars, mask, halo):
    latmax = mask.sizes['nlat']
    lonmax = mask.sizes['nlon']

    latrange = range(halo, latmax - halo)
    lonrange = range(halo, lonmax - halo)

    stencil = {'nlat': latrange, 'nlon': lonrange}

    sub = vars[stencil]
    stencilled = {k: v.to_numpy() for k, v in sub.items()}
    return stencilled


def convolve(vars, mask, kernel):
    '''
    Assumes kernel is a NxN matrix where N is odd
    '''

    latmax = mask.sizes['nlat']
    lonmax = mask.sizes['nlon']

    halo_lat = int(np.floor(kernel.sizes['nlat'] / 2.))
    halo_lon = int(np.floor(kernel.sizes['nlon'] / 2.))

    latrange = range(halo_lat, latmax - halo_lat)
    lonrange = range(halo_lon, lonmax - halo_lon)

    conved = dict()
    for k, v in vars.items():

        cvar = np.zeros_like(v[latrange, lonrange])
        for i, j in product(latrange, lonrange):
            stencil = {"nlat": slice(i - halo_lat, i + halo_lat + 1), "nlon": slice(j - halo_lon, j + halo_lon + 1)}
            msub = mask[stencil]

            denominator = np.abs(kernel * msub).sum(dim=['nlat', 'nlon'])
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