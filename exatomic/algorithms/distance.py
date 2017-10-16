# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Two Body Properties Computations
#####################################
"""
import numpy as np
import numba as nb


@nb.vectorize(["float64(float64, float64, float64)"], nopython=True, target="parallel")
def cartmag(x, y, z):
    """
    Vectorized operation to compute the magnitude of a three component array.
    """
    return np.sqrt(x**2 + y**2 + z**2)


@nb.vectorize(["float64(float64, float64)"], nopython=True, target="parallel")
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


@nb.jit(nopython=True, nogil=True, parallel=True)
def pdist_pbc_ortho(ux, uy, uz, a, b, c, index, dmax=8.0):
    """
    Pairwise two body calculation for bodies in an orthorhombic periodic cell.

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
    for i in nb.prange(n):
        xi = ux[i]
        yi = uy[i]
        zi = uz[i]
        # For each atom j
        for j in range(i+1, n):
            xj = ux[j]
            yj = uy[j]
            zj = uz[j]
            dpx = np.nan
            dpy = np.nan
            dpz = np.nan
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


@nb.jit(nopython=True, nogil=True, parallel=True)
def pdist(x, y, z, index, dmax=8.0):
    """
    3D free boundary condition
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
    for i in nb.prange(m):
        for j in range(i + 1, m):
            dx_ = x[i] - x[j]
            dy_ = y[i] - y[j]
            dz_ = z[i] - z[j]
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


#def compute_two(atom, dmax=8.0):
#    dxs = []
#    dys = []
#    dzs = []
#    drs = []
#    atom0s = []
#    atom1s = []
#    for fdx, group in atom.groupby("frame"):
#        dx, dy, dz, dr, atom0, atom1 = nb_pdist(group['x'].values, group['y'].values,
#                                                group['z'].values, group.index.values, dmax)
#        dxs.append(dx)
#        dys.append(dy)
#        dzs.append(dz)
#        drs.append(dr)
#        atom0s.append(atom0)
#        atom1s.append(atom1)
#    dxs = np.concatenate(dxs)
#    dys = np.concatenate(dys)
#    dzs = np.concatenate(dzs)
#    drs = np.concatenate(drs)
#    atom0s = np.concatenate(atom0s).astype(int)
#    atom1s = np.concatenate(atom1s).astype(int)
#    return Two.from_dict({'dr': drs, 'dx': dxs, 'dy': dys, 'dz': dzs, 'atom0': atom0s, 'atom1': atom1s})
#
#@nb.jit(nopython=True, nogil=True, cache=True)
#def prjpdist(x, y, z, index):
#    """Origin is 0, 0, 0"""
#    n = len(x)
#    m = n*(n + 1)//2
#    d = np.empty((m, 4), dtype=np.int64)
#    adj = np.empty((m, ), dtype=np.int64)
#    index0 = np.empty((m, ), dtype=np.int64)
#    index1 = np.empty((m, ), dtype=np.int64)
#    k = 0
#    for i in range(n):
#        xi = x[i]
#        yi = y[i]
#        zi = z[i]
#        for j in range(i, n):
#            index0[k] = i
#            index1[k] = j
#            dx = (xi - x[j])**2
#            dy = (yi - y[j])**2
#            dz = (zi - z[j])**2
#            dr = dx + dy + dz
#            d[k, 0] = dx
#            d[k, 1] = dy
#            d[k, 2] = dz
#            d[k, 3] = dr
#            if dr > 2.0:
#                adj[k] = 0
#            else:
#                adj[k] = 1
#            k += 1
#    return d, adj, index0, index1
#
#d2, adj, index0, index1 = prjpdist(projections['a'].values, projections['b'].values, projections['c'].values,
#                         projections.index.values)
#
#prjdf = pd.DataFrame(d2, columns=("dx", "dy", "dz", "dr"))
#prjdf['adj'] = adj
#prjdf['index0'] = index0
#prjdf['index1'] = index1
#prjhash = prjdf[['dx', 'dy', 'dz', 'dr']].values
#prjhash
#
#
#
#
#from exa.math.misc.repeat import repeat_counts_f8_2d
#from exa.math.vector.cartesian import magnitude_xyz
#
#
#def minimal_image(xyz, rxyz, oxyz):
#    """
#    """
#    return np.mod(xyz, rxyz) + oxyz
#
#
##def minimal_image_counts(xyz, rxyz, oxyz, counts):
##    """
##    """
##    rxyz = repeat_counts_f8_2d(rxyz, counts)
##    oxyz = repeat_counts_f8_2d(oxyz, counts)
##    return minimal_image(xyz, rxyz, oxyz)
#
#
#def periodic_pdist_euc_dxyz_idx(ux, uy, uz, rx, ry, rz, idxs, tol=10**-8):
#    """
#    Pairwise Euclidean distance computation for periodic systems returning
#    distance vectors, pair indices, and projected coordinates.
#    """
#    m = [-1, 0, 1]
#    n = len(ux)
#    nn = n*(n - 1)//2
#    dx = np.empty((nn, ), dtype=np.float64)    # Two body distance component x
#    dy = np.empty((nn, ), dtype=np.float64)    # within corresponding periodic
#    dz = np.empty((nn, ), dtype=np.float64)    # unit cell
#    dr = np.empty((nn, ), dtype=np.float64)
#    px = np.empty((nn, ), dtype=np.float64)    # Projected j coordinate x
#    py = np.empty((nn, ), dtype=np.float64)    # Projected j coordinate y
#    pz = np.empty((nn, ), dtype=np.float64)    # Projected j coordinate z
#    pxj = np.empty((27, ), dtype=np.float64)
#    pyj = np.empty((27, ), dtype=np.float64)
#    pzj = np.empty((27, ), dtype=np.float64)
#    prj = np.empty((27, ), dtype=np.float64)
#    idxi = np.empty((nn, ), dtype=np.int64)    # index of i
#    idxj = np.empty((nn, ), dtype=np.int64)    # index of j
#    h = 0
#    for i in range(n):
#        xi = ux[i]
#        yi = uy[i]
#        zi = uz[i]
#        indexi = idxs[i]
#        for j in range(i + 1, n):
#            xj = ux[j]
#            yj = uy[j]
#            zj = uz[j]
#            hh = 0
#            for ii in m:
#                for jj in m:
#                    for kk in m:
#                        pxjj = xj + ii*rx
#                        pxj[hh] = pxjj
#                        pyjj = yj + jj*ry
#                        pyj[hh] = pyjj
#                        pzjj = zj + kk*rz
#                        pzj[hh] = pzjj
#                        prj[hh] = (xi - pxjj)**2 + (yi - pyjj)**2 + (zi - pzjj)**2
#                        hh += 1
#            hh = np.argmin(prj)
#            dx[h] = pxj[hh]
#            dy[h] = pyj[hh]
#            dz[h] = pzj[hh]
#            dr[h] = np.sqrt(prj[hh])
#            pxjj = pxj[hh]
#            pyjj = pyj[hh]
#            pzjj = pzj[hh]
#            if np.abs(xj - pxjj) <= tol:
#                px[h] = np.nan
#            else:
#                px[h] = pxjj
#            if np.abs(yj - pyjj) <= tol:
#                py[h] = np.nan
#            else:
#                py[h] = pyjj
#            if np.abs(zj - pzjj) <= tol:
#                pz[h] = np.nan
#            else:
#                pz[h] = pzjj
#            idxi[h] = indexi
#            idxj[h] = idxs[j]
#            h += 1
#    return dx, dy, dz, dr, idxi, idxj, px, py, pz
#
#
#def _compute(cx, cy, cz, rx, ry, rz, ox, oy, oz):
#    """
#    """
#    l = [-1, 0, 1]
#    m = len(cx)
#    dx = np.empty((m, ), dtype=np.float64)
#    dy = np.empty((m, ), dtype=np.float64)
#    dz = np.empty((m, ), dtype=np.float64)
#    px = np.empty((27, ), dtype=np.float64)
#    py = np.empty((27, ), dtype=np.float64)
#    pz = np.empty((27, ), dtype=np.float64)
#    pr = np.empty((27, ), dtype=np.float64)
#    h = 0
#    for i in range(m):
#        cxi = cx[i]
#        cyi = cy[i]
#        czi = cz[i]
#        hh = 0
#        for ii in l:
#            for jj in l:
#                for kk in l:
#                    sx = ii*rx
#                    sy = jj*ry
#                    sz = kk*rz
#                    xx = cxi + sx
#                    yy = cyi + sy
#                    zz = czi + sz
#                    pr[hh] = (ox - xx)**2 + (oy - yy)**2 + (oz - zz)**2
#                    px[hh] = sx
#                    py[hh] = sy
#                    pz[hh] = sz
#                    hh += 1
#        hh = np.argmin(pr)
#        dx[h] = px[hh]
#        dy[h] = py[hh]
#        dz[h] = pz[hh]
#        h += 1
#    return dx, dy, dz
