# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Two Body Properties Computations
#####################################
"""
import numpy as np
import numba as nb
import pandas as pd
from IPython.display import display
from ipywidgets import FloatProgress
from exatomic.base import nbpll


@nb.jit(nopython=True, nogil=True, parallel=nbpll)
def angles(dx, dy, dz, dr, atom0, atom1):
    n = len(dx)
    m = n*(n-1)//2
    rad = np.empty((m, ), dtype=np.float64)
    adx = np.empty((m, 3), dtype=np.int64)
    k = 0
    for i in range(n):
        for j in range(i+1, n):
            dot = dx[i]*dx[j]
            dot += dy[i]*dy[j]
            dot += dz[i]*dz[j]
            rad[k] = np.arccos(dot/(dr[i]*dr[j]))
            adx[k, 0] = atom0
            adx[k, 1] = atom1[i]
            adx[k, 2] = atom1[j]
            k += 1
    return rad, adx


# Angles
def compute_angles_out_of_core(hdfname, uni, bond=True):
    """
    Given an HDF of atom two body properties, compute angles.

    Atomic two body data is expected to have been computed (see
    :func:`~exatomic.core.two.compute_atom_two_out_of_core`)

    Args:
        hdfname (str): Path to HDF file containing two body data
        uni (:class:`~exatomic.core.universe.Universe`): Universe
        bond (bool): Restrict to bond angles (default True)

    Warning:
        If bond is set to False, this process may take a very long time.
    """
    store = pd.HDFStore(hdfname, mode="a")
    f = u.atom['frame'].unique()
    n = len(f)
    fp = FloatProgress(description="Computing:")
    display(fp)
    for i, fdx in enumerate(f):
        tdf = store.get("frame_"+str(fdx) + "/atom_two")
        indexes = []
        radians = []
        for atom0, group in tdf[tdf['bond'] == True].groupby("atom0"):
            dx = group['dx'].values.astype(float)
            dy = group['dy'].values.astype(float)
            dz = group['dz'].values.astype(float)
            dr = group['dr'].values.astype(float)
            atom1 = group['atom1'].values.astype(int)
            rad, adx = angles(dx, dy, dz, dr, atom0, atom1)
            indexes.append(adx)
            radians.append(rad)
        indexes = np.concatenate(indexes)
        radians = np.concatenate(radians)
        adf = pd.DataFrame(indexes, columns=("atom0", "atom1", "atom2"))
        adf['angle'] = radians
        store.put("frame_"+str(fdx) + "/atom_angle", adf)
        fp.value = i/n*100
    store.close()
    fp.close()
