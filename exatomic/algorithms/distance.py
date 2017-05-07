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
#from exa.math.misc.repeat import repeat_counts_f8_2d
#from exa.math.vector.cartesian import magnitude_xyz


def minimal_image(xyz, rxyz, oxyz):
    """
    """
    return np.mod(xyz, rxyz) + oxyz


#def minimal_image_counts(xyz, rxyz, oxyz, counts):
#    """
#    """
#    rxyz = repeat_counts_f8_2d(rxyz, counts)
#    oxyz = repeat_counts_f8_2d(oxyz, counts)
#    return minimal_image(xyz, rxyz, oxyz)


def periodic_pdist_euc_dxyz_idx(ux, uy, uz, rx, ry, rz, idxs, tol=10**-8):
    """
    Pairwise Euclidean distance computation for periodic systems returning
    distance vectors, pair indices, and projected coordinates.
    """
    m = [-1, 0, 1]
    n = len(ux)
    nn = n*(n - 1)//2
    dx = np.empty((nn, ), dtype=np.float64)    # Two body distance component x
    dy = np.empty((nn, ), dtype=np.float64)    # within corresponding periodic
    dz = np.empty((nn, ), dtype=np.float64)    # unit cell
    dr = np.empty((nn, ), dtype=np.float64)
    px = np.empty((nn, ), dtype=np.float64)    # Projected j coordinate x
    py = np.empty((nn, ), dtype=np.float64)    # Projected j coordinate y
    pz = np.empty((nn, ), dtype=np.float64)    # Projected j coordinate z
    pxj = np.empty((27, ), dtype=np.float64)
    pyj = np.empty((27, ), dtype=np.float64)
    pzj = np.empty((27, ), dtype=np.float64)
    prj = np.empty((27, ), dtype=np.float64)
    idxi = np.empty((nn, ), dtype=np.int64)    # index of i
    idxj = np.empty((nn, ), dtype=np.int64)    # index of j
    h = 0
    for i in range(n):
        xi = ux[i]
        yi = uy[i]
        zi = uz[i]
        indexi = idxs[i]
        for j in range(i + 1, n):
            xj = ux[j]
            yj = uy[j]
            zj = uz[j]
            hh = 0
            for ii in m:
                for jj in m:
                    for kk in m:
                        pxjj = xj + ii*rx
                        pxj[hh] = pxjj
                        pyjj = yj + jj*ry
                        pyj[hh] = pyjj
                        pzjj = zj + kk*rz
                        pzj[hh] = pzjj
                        prj[hh] = (xi - pxjj)**2 + (yi - pyjj)**2 + (zi - pzjj)**2
                        hh += 1
            hh = np.argmin(prj)
            dx[h] = pxj[hh]
            dy[h] = pyj[hh]
            dz[h] = pzj[hh]
            dr[h] = np.sqrt(prj[hh])
            pxjj = pxj[hh]
            pyjj = pyj[hh]
            pzjj = pzj[hh]
            if np.abs(xj - pxjj) <= tol:
                px[h] = np.nan
            else:
                px[h] = pxjj
            if np.abs(yj - pyjj) <= tol:
                py[h] = np.nan
            else:
                py[h] = pyjj
            if np.abs(zj - pzjj) <= tol:
                pz[h] = np.nan
            else:
                pz[h] = pzjj
            idxi[h] = indexi
            idxj[h] = idxs[j]
            h += 1
    return dx, dy, dz, dr, idxi, idxj, px, py, pz


def _compute(cx, cy, cz, rx, ry, rz, ox, oy, oz):
    """
    """
    l = [-1, 0, 1]
    m = len(cx)
    dx = np.empty((m, ), dtype=np.float64)
    dy = np.empty((m, ), dtype=np.float64)
    dz = np.empty((m, ), dtype=np.float64)
    px = np.empty((27, ), dtype=np.float64)
    py = np.empty((27, ), dtype=np.float64)
    pz = np.empty((27, ), dtype=np.float64)
    pr = np.empty((27, ), dtype=np.float64)
    h = 0
    for i in range(m):
        cxi = cx[i]
        cyi = cy[i]
        czi = cz[i]
        hh = 0
        for ii in l:
            for jj in l:
                for kk in l:
                    sx = ii*rx
                    sy = jj*ry
                    sz = kk*rz
                    xx = cxi + sx
                    yy = cyi + sy
                    zz = czi + sz
                    pr[hh] = (ox - xx)**2 + (oy - yy)**2 + (oz - zz)**2
                    px[hh] = sx
                    py[hh] = sy
                    pz[hh] = sz
                    hh += 1
        hh = np.argmin(pr)
        dx[h] = px[hh]
        dy[h] = py[hh]
        dz[h] = pz[hh]
        h += 1
    return dx, dy, dz


if config['dynamic']['numba'] == 'true':
    from numba import jit, vectorize
    #from exa.math.vector.cartesian import magnitude_xyz
    types3 = ['int32(int32, int32, int32)', 'int64(int64, int64, int64)',
             'float32(float32, float32, float32)', 'float64(float64, float64, float64)']
    #minimal_image_counts = jit(nopython=True, cache=True, nogil=True)(minimal_image_counts)
    minimal_image = vectorize(types3, nopython=True)(minimal_image)
    periodic_pdist_euc_dxyz_idx = jit(nopython=True, cache=True, nogil=True)(periodic_pdist_euc_dxyz_idx)
    _compute = jit(nopython=True, cache=True, nogil=True)(_compute)
