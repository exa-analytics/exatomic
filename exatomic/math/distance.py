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


def free_two_frame(x, y, z, idx, frame):
    """
    Compute the minimum data required for :class:`~exatomic.two.FreeTwo` for a
    given frame.

    Args:
        x (array): Array of x coordinates
        y (array): Array of y coordinates
        z (array): Array of z coordinates
        idx (array): Array of atom indices
        frame (int): Frame integer

    Returns:
        dx (array): Differences in x components (i - j)
        dy (array): Differences in y components (i - j)
        dz (array): Differences in z components (i - j)
        idx0 (array): First index (i)
        idx1 (array): Second index (j)
        frame (array): Frame array
        d (array): Distance
    """
    n = len(idx)
    nn = n*(n - 1)//2
    dx = np.empty((nn, ), dtype=np.float64)
    dy = np.empty((nn, ), dtype=np.float64)
    dz = np.empty((nn, ), dtype=np.float64)
    idx0 = np.empty((nn, ), dtype=np.int64)
    idx1 = np.empty((nn, ), dtype=np.int64)
    fdx = np.empty((nn, ), dtype=np.int64)
    h = 0
    for i in range(n):
        for j in range(i+1, n):
            dx[h] = x[i] - x[j]
            dy[h] = y[i] - y[j]
            dz[h] = z[i] - z[j]
            idx0[h] = idx[i]
            idx1[h] = idx[j]
            fdx[h] = frame
            h += 1
    return dx, dy, dz, idx0, idx1, fdx, magnitude_xyz(dx, dy, dz)


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


def periodic_two_frame(ux, uy, uz, rx, ry, rz, idx, frame):
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
    fdxs = np.empty((nn, ), dtype=np.int64)     # frame index
    h = 0
    for ii in m:
        for jj in m:
            for kk in m:
                for i in range(n):
                    xi = ux[i]
                    yi = uy[i]
                    zi = uz[i]
                    for j in range(i+1, n):
                        xj = ux[j]
                        yj = uy[j]
                        zj = uz[j]
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
                        fdxs[h] = frame
                        h += 1
    ds = magnitude_xyz(dxs, dys, dzs)
    return dxs, dys, dzs, ds, idx0s, idx1s, pxs, pys, pzs, fdxs


#def periodic_two_frame(ux, uy, uz, rx, ry, rz, idx, frame, tol=10**-8):
#    """
#    There is only one distance between two atoms.
#    """
#    m = [-1, 0, 1]
#    n = len(ux)
#    nn = n*(n - 1)//2
#    dxs = np.empty((nn, ), dtype=np.float64)    # Two body distance component x
#    dys = np.empty((nn, ), dtype=np.float64)    # within corresponding periodic
#    dzs = np.empty((nn, ), dtype=np.float64)    # unit cell
#    ds = np.empty((nn, ), dtype=np.float64)
#    pxs = np.empty((nn, ), dtype=np.float64)    # Projected j coordinate x
#    pys = np.empty((nn, ), dtype=np.float64)    # Projected j coordinate y
#    pzs = np.empty((nn, ), dtype=np.float64)    # Projected j coordinate z
#    idx0s = np.empty((nn, ), dtype=np.int64)    # index of i
#    idx1s = np.empty((nn, ), dtype=np.int64)    # index of j
#    fdxs = np.empty((nn, ), dtype=np.int64)     # frame index
#    iquad = np.empty((nn, ), dtype=np.float64)
#    jquad = np.empty((nn, ), dtype=np.float64)
#    kquad = np.empty((nn, ), dtype=np.float64)
#    quad = np.empty((nn, ), dtype=np.float64)
#    h = 0
#    for i in range(n):
#        for j in range(i+1, n):
#            xi = ux[i]
#            yi = uy[i]
#            zi = uz[i]
#            xj = ux[j]
#            yj = uy[j]
#            zj = uz[j]
#            pxj = np.empty((27, ), dtype=np.float64)    # projected coords
#            pyj = np.empty((27, ), dtype=np.float64)
#            pzj = np.empty((27, ), dtype=np.float64)
#            pdx = np.empty((27, ), dtype=np.float64)    # projected distances
#            pdy = np.empty((27, ), dtype=np.float64)
#            pdz = np.empty((27, ), dtype=np.float64)
#            iijjkk = np.empty((27, 3), dtype=np.float64)
#            hh = 0
#            for ii in m:
#                for jj in m:
#                    for kk in m:
#                        xpj = xj + ii*rx
#                        ypj = yj + jj*ry
#                        zpj = zj + kk*rz
#                        pdx[hh] = xi - xpj
#                        pdy[hh] = yi - ypj
#                        pdz[hh] = zi - zpj
#                        pxj[hh] = xpj
#                        pyj[hh] = ypj
#                        pzj[hh] = zpj
#                        iijjkk[hh] = [ii, jj, kk]
#                        hh += 1
#            pds = magnitude_xyz(pdx, pdy, pdz)
#            hh = np.argmin(pds)
#            #if pds[hh] < dmax:
#            if np.abs(pxj[hh] - xj) <= tol:
#                pxs[h] = np.nan
##                iquad[h] = np.nan
#            else:
#                pxs[h] = pxj[hh]
##                iquad[h] = iijjkk[hh, 0]
#            if np.abs(pyj[hh] - yj) <= tol:
#                pys[h] = np.nan
##                jquad[h] = np.nan
#            else:
#                pys[h] = pyj[hh]
##                jquad[h] = iijjkk[hh, 1]
#            if np.abs(pzj[hh] - zj) <= tol:
#                pzs[h] = np.nan
##                kquad[h] = np.nan
#            else:
#                pzs[h] = pzj[hh]
##                kquad[h] = iijjkk[hh, 2]
#            iquad[h] = iijjkk[hh, 0]
#            jquad[h] = iijjkk[hh, 1]
#            kquad[h] = iijjkk[hh, 2]
#            quad[h] = hh
#            dxs[h] = pdx[hh]
#            dys[h] = pdy[hh]
#            dzs[h] = pdz[hh]
#            fdxs[h] = frame
#            ds[h] = pds[hh]
#            idx0s[h] = idx[i]
#            idx1s[h] = idx[j]
#            h += 1
#    return dxs, dys, dzs, ds, idx0s, idx1s, fdxs, pxs, pys, pzs, iquad, jquad, kquad, quad


if config['dynamic']['numba'] == 'true':
    from numba import jit, vectorize
    from exa.math.vector.cartesian import magnitude_xyz
    types3 = ['int32(int32, int32, int32)', 'int64(int64, int64, int64)',
             'float32(float32, float32, float32)', 'float64(float64, float64, float64)']
    free_two_frame = jit(nopython=True, cache=True, nogil=True)(free_two_frame)
    minimal_image_counts = jit(nopython=True, cache=True, nogil=True)(minimal_image_counts)
    minimal_image = vectorize(types3, nopython=True)(minimal_image)
    periodic_two_frame = jit(nopython=True, cache=True, nogil=True)(periodic_two_frame)
