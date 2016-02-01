# -*- coding: utf-8 -*-
'''
Pair Correlation Functions
============================
'''


def compute_radial_pair_correlation(universe, A, B, dr=0.05, start=None, stop=None):
    '''
    '''
    start = u.two['distance'].min() - 0.1
    stop = u.two['distance'].max() + 0.1
    distances = u.two.ix[(u.two['symbols'] == A + B) | (u.two['symbols'] == B + A), 'distance']
    bins = np.arange(start, stop, dr)
    bins = np.append(bins, bins[-1] + dr)
    hist, bins = np.histogram(distances, bins)
    n = len(distances)
    m = len(u)
    vol = u.frame.ix[0, 'cell_volume']
    rho = n / vol
    r3 = bins[1:]**3 - bins[:-1]**3
    g = hist / (4 / 3 * np.pi * r3 * rho)
    r = (bins[1:] + bins[:-1]) / 2
    i = np.cumsum(hist) / m
    return pd.DataFrame.from_dict({'g': g, 'r': r, 'i': i})
