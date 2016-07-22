# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Two Body Properties Computations
#####################################
Fast computations required for generating :class:`~exatomci.two.FreeTwo` and
:class:`~exatomic.two.PeriodicTwo` objects.

Warning:
    Without `numba`_, performance of this module is not guarenteed.

.. _numba: http://numba.pydata.org/
"""
import numpy as np
from exa._config import config
from exa.math.misc.repeat import repeat_counts_f8_2d
from exa.math.vector.cartesian import magnitude_xyz


def minimal_image(xyz, rxyz, oxyz):
    """
    """
    return np.mod(xyz, rxyz) + oxyz


def minimal_image_counts(xyz, rxyz, oxyz, counts):
    """
    """
    rxyz = repeat_counts_f8_2d(rxyz, counts)
    oxyz = repeat_counts_f8_2d(oxyz, counts)
    return minimal_image(xyz, rxyz, oxyz)


def periodic_two_frame(ux, uy, uz, rx, ry, rz, idx):
    """
    There is only one distance between two atoms.
    """
    m = [-1, 0, 1]
    n = len(ux)
    nn = 27*n*(n - 1)//2
    dxs = np.empty((nn, ), dtype=np.float64)    # Two body distance component x
    dys = np.empty((nn, ), dtype=np.float64)    # within corresponding periodic
    dzs = np.empty((nn, ), dtype=np.float64)    # unit cell
    pxs = np.empty((nn, ), dtype=np.float64)    # Projected j coordinate x
    pys = np.empty((nn, ), dtype=np.float64)    # Projected j coordinate y
    pzs = np.empty((nn, ), dtype=np.float64)    # Projected j coordinate z
    idx0s = np.empty((nn, ), dtype=np.int64)    # index of i
    idx1s = np.empty((nn, ), dtype=np.int64)    # index of j
    h = 0
    for i in range(n):
        xi = ux[i]
        yi = uy[i]
        zi = uz[i]
        for j in range(i + 1, n):
            xj = ux[j]
            yj = uy[j]
            zj = uz[j]
            for ii in m:
                for jj in m:
                    for kk in m:
                        xpj = xj + ii*rx
                        ypj = yj + jj*ry
                        zpj = zj + kk*rz
                        dxs[h] = xi - xpj
                        dys[h] = yi - ypj
                        dzs[h] = zi - zpj
                        pxs[h] = xpj
                        pys[h] = ypj
                        pzs[h] = zpj
                        idx0s[h] = idx[i]
                        idx1s[h] = idx[j]
                        h += 1
    ds = magnitude_xyz(dxs, dys, dzs)
    return dxs, dys, dzs, ds, idx0s, idx1s, pxs, pys, pzs


if config['dynamic']['numba'] == 'true':
    from numba import jit, vectorize
    from exa.math.vector.cartesian import magnitude_xyz
    types3 = ['int32(int32, int32, int32)', 'int64(int64, int64, int64)',
             'float32(float32, float32, float32)', 'float64(float64, float64, float64)']
    minimal_image_counts = jit(nopython=True, cache=True, nogil=True)(minimal_image_counts)
    minimal_image = vectorize(types3, nopython=True)(minimal_image)
    periodic_two_frame = jit(nopython=True, cache=True, nogil=True)(periodic_two_frame)
