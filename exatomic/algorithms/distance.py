# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Two Body Properties Computations
#####################################
"""
import numpy as np
import numba as nb
from exatomic.base import nbtgt, nbpll


@nb.vectorize(["float64(float64, float64, float64)"], nopython=True, target=nbtgt)
def cartmag(x, y, z):
    """
    Vectorized operation to compute the magnitude of a three component array.
    """
    return np.sqrt(x**2 + y**2 + z**2)


@nb.vectorize(["float64(float64, float64)"], nopython=True, target=nbtgt)
def modv(x, y):
    """
    Vectorized modulo operation.

    Args:
        x (array): 1D array
        y (array, float): Scalar or 1D array

    Returns:
        z (array): x modulo y
    """
    return np.mod(x, y)


@nb.jit(nopython=True, nogil=True, parallel=nbpll)
def pdist_ortho(ux, uy, uz, a, b, c, index, dmax=8.0):
    """
    Pairwise two body calculation for bodies in an orthorhombic periodic cell.

    Does return distance vectors.

    An orthorhombic cell is defined by orthogonal vectors of length a and b
    (which define the base) and height vector of length c. All three vectors
    intersect at 90° angles. If a = b = c the cell is a simple cubic cell.
    This function assumes the unit cell is constant with respect to an external
    frame of reference and that the origin of the cell is at (0, 0, 0).

    Args:
        ux (array): In unit cell x array
        uy (array): In unit cell y array
        uz (array): In unit cell z array
        a (float): Unit cell dimension a
        b (float): Unit cell dimension b
        c (float): Unit cell dimension c index (array): Atom indexes
        dmax (float): Maximum distance of interest
    """
    m = [-1, 0, 1]
    dmax2 = dmax**2
    n = len(ux)
    nn = n*(n - 1)//2
    dx = np.empty((nn, ), dtype=np.float64)
    dy = dx.copy()
    dz = dx.copy()
    dr = dx.copy()
    ii = np.empty((nn, ), dtype=np.int64)
    jj = ii.copy()
    projection = ii.copy()
    k = 0
    # For each atom i
    for i in range(n):
        xi = ux[i]
        yi = uy[i]
        zi = uz[i]
        # For each atom j
        for j in range(i+1, n):
            xj = ux[j]
            yj = uy[j]
            zj = uz[j]
            dpr = dmax2
            inck = False
            prj = 0
            # Check all projections of atom i
            # Note that i, j are in the unit cell so we make a 3x3x3 'supercell'
            # of i around j
            # The index of the projections of i go from 0 to 26 (27 projections)
            # The 13th projection is the unit cell itself.
            for aa in m:
                for bb in m:
                    for cc in m:
                        pxi = xi + aa*a
                        pyi = yi + bb*b
                        pzi = zi + cc*c
                        dpx_ = pxi - xj
                        dpy_ = pyi - yj
                        dpz_ = pzi - zj
                        dpr_ = dpx_**2 + dpy_**2 + dpz_**2
                        # The second criteria here enforces that prefer the projection
                        # with the largest value (i.e. 0 = [-1, -1, -1] < 13 = [0, 0, 0]
                        # < 26 = [1, 1, 1])
                        # The system sets a fixed preference for the projected positions rather
                        # than having a random choice.
                        if dpr_ < dpr:
                            dx[k] = dpx_
                            dy[k] = dpy_
                            dz[k] = dpz_
                            dr[k] = np.sqrt(dpr_)
                            ii[k] = index[i]
                            jj[k] = index[j]
                            projection[k] = prj
                            dpr = dpr_
                            inck = True
                        prj += 1
            if inck:
                k += 1
    dx = dx[:k]
    dy = dy[:k]
    dz = dz[:k]
    dr = dr[:k]
    ii = ii[:k]
    jj = jj[:k]
    projection = projection[:k]
    return dx, dy, dz, dr, ii, jj, projection


@nb.jit(nopython=True, nogil=True, parallel=nbpll)
def pdist_ortho_nv(ux, uy, uz, a, b, c, index, dmax=8.0):
    """
    Pairwise two body calculation for bodies in an orthorhombic periodic cell.

    Does not return distance vectors.

    An orthorhombic cell is defined by orthogonal vectors of length a and b
    (which define the base) and height vector of length c. All three vectors
    intersect at 90° angles. If a = b = c the cell is a simple cubic cell.
    This function assumes the unit cell is constant with respect to an external
    frame of reference and that the origin of the cell is at (0, 0, 0).

    Args:
        ux (array): In unit cell x array
        uy (array): In unit cell y array
        uz (array): In unit cell z array
        a (float): Unit cell dimension a
        b (float): Unit cell dimension b
        c (float): Unit cell dimension c index (array): Atom indexes
        dmax (float): Maximum distance of interest
    """
    m = [-1, 0, 1]
    dmax2 = dmax**2
    n = len(ux)
    nn = n*(n - 1)//2
    dr = np.empty((nn, ), dtype=np.float64)
    ii = np.empty((nn, ), dtype=np.int64)
    jj = ii.copy()
    projection = ii.copy()
    k = 0
    # For each atom i
    for i in range(n):
        xi = ux[i]
        yi = uy[i]
        zi = uz[i]
        # For each atom j
        for j in range(i+1, n):
            xj = ux[j]
            yj = uy[j]
            zj = uz[j]
            dpr = dmax2
            inck = False
            prj = 0
            # Check all projections of atom i
            # Note that i, j are in the unit cell so we make a 3x3x3 'supercell'
            # of i around j
            # The index of the projections of i go from 0 to 26 (27 projections)
            # The 13th projection is the unit cell itself.
            for aa in m:
                for bb in m:
                    for cc in m:
                        pxi = xi + aa*a
                        pyi = yi + bb*b
                        pzi = zi + cc*c
                        dpx_ = pxi - xj
                        dpy_ = pyi - yj
                        dpz_ = pzi - zj
                        dpr_ = dpx_**2 + dpy_**2 + dpz_**2
                        # The second criteria here enforces that prefer the projection
                        # with the largest value (i.e. 0 = [-1, -1, -1] < 13 = [0, 0, 0]
                        # < 26 = [1, 1, 1])
                        # The system sets a fixed preference for the projected positions rather
                        # than having a random choice.
                        if dpr_ < dpr:
                            dr[k] = np.sqrt(dpr_)
                            ii[k] = index[i]
                            jj[k] = index[j]
                            projection[k] = prj
                            dpr = dpr_
                            inck = True
                        prj += 1
            if inck:
                k += 1
    dr = dr[:k]
    ii = ii[:k]
    jj = jj[:k]
    projection = projection[:k]
    return dr, ii, jj, projection


@nb.jit(nopython=True, nogil=True, parallel=nbpll)
def pdist(x, y, z, index, dmax=8.0):
    """
    Pairwise distance computation for points in cartesian space.

    Does return distance vectors.
    """
    dmax2 = dmax**2
    m = len(x)
    n = m*(m - 1)//2
    dx = np.empty((n, ), dtype=np.float64)
    dy = dx.copy()
    dz = dx.copy()
    dr = dx.copy()
    atom0 = np.empty((n, ), dtype=np.int64)
    atom1 = atom0.copy()
    k = 0
    for i in range(m):
        xi = x[i]
        yi = y[i]
        zi = z[i]
        for j in range(i + 1, m):
            dx_ = xi - x[j]
            dy_ = yi - y[j]
            dz_ = zi - z[j]
            dr2_ = dx_**2 + dy_**2 + dz_**2
            if dr2_ < dmax2:
                dx[k] = dx_
                dy[k] = dy_
                dz[k] = dz_
                dr[k] = np.sqrt(dr2_)
                atom0[k] = index[i]
                atom1[k] = index[j]
                k += 1
    dx = dx[:k]
    dy = dy[:k]
    dz = dz[:k]
    dr = dr[:k]
    atom0 = atom0[:k]
    atom1 = atom1[:k]
    return dx, dy, dz, dr, atom0, atom1


@nb.jit(nopython=True, nogil=True, parallel=nbpll)
def pdist_nv(x, y, z, index, dmax=8.0):
    """
    Pairwise distance computation for points in cartesian space.

    Does not return distance vectors.
    """
    dmax2 = dmax**2
    m = len(x)
    n = m*(m - 1)//2
    dr = np.empty((n, ), dtype=np.float64)
    atom0 = np.empty((n, ), dtype=np.int64)
    atom1 = atom0.copy()
    k = 0
    for i in range(m):
        xi = x[i]
        yi = y[i]
        zi = z[i]
        for j in range(i + 1, m):
            dr_ = (xi - x[j])**2 + (yi - y[j])**2 + (zi - z[j])**2
            if dr_ < dmax2:
                dr[k] = np.sqrt(dr_)
                atom0[k] = index[i]
                atom1[k] = index[j]
                k += 1
    dr = dr[:k]
    atom0 = atom0[:k]
    atom1 = atom1[:k]
    return dr, atom0, atom1
