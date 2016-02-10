# -*- coding: utf-8 -*-
'''
Pair Correlation Functions
============================
'''
from scipy.integrate import cumtrapz
from exa import _np as np
from exa import _pd as pd


def compute_radial_pair_correlation(universe, A, B, dr=0.05, start=None, stop=None):
    '''
    '''
    start = universe.two['distance'].min() - 0.1
    stop = universe.two['distance'].max() + 0.1
    distances = universe.two.ix[(universe.two['symbols'] == A + B) | (universe.two['symbols'] == B + A), 'distance']
    bins = np.arange(start, stop, dr)
    bins = np.append(bins, bins[-1] + dr)
    hist, bins = np.histogram(distances, bins)
    n = len(distances)
    m = len(universe)
    f = universe._framelist[0]
    vol = universe.frame.ix[f, 'cell_volume']
    rho = n / vol
    r3 = bins[1:]**3 - bins[:-1]**3
    g = hist / (4 / 3 * np.pi * r3 * rho)
    r = (bins[1:] + bins[:-1]) / 2
    count = cumtrapz(g, x=r, dx=dr)
    count *= 4/ 3 * np.pi * r3
    g_name = ''.join(('$g_{', A + B, '}'))
    return pd.DataFrame.from_dict({g_name: g, '$r$': r, '$Pair Count$': count})
