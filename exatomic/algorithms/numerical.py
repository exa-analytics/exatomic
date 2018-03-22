# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Numerical methods and classes
###############################
Everything in this module is implemented in numba.
"""
import numpy as np
import pandas as pd
from numba import (jit, njit, jitclass, vectorize, prange,
                   deferred_type, optional, int64, float64, boolean)

#################
# Miscellaneous #
#################

@jit(nopython=True, cache=True)
def _fac(n,v): return _fac(n-1, n*v) if n else v
@jit(nopython=True, cache=True)
def _fac2(n,v): return _fac2(n-2, n*v) if n > 0 else v

@jit(nopython=True, cache=True)
def fac(n): return _fac(n, 1) if n > -1 else 0
@jit(nopython=True, cache=True)
def fac2(n): return _fac2(n, 1) if n > 1 else 1 if n > -2 else 0
@jit(nopython=True, cache=True)
def dfac21(n): return fac2(2 * n - 1)
@jit(nopython=True, cache=True)
def choose(n, k): return fac(n) // (fac(k) * fac(n - k))
@jit(nopython=True, cache=True)
def sdist(ax, ay, az, bx, by, bz):
    return (ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2

@vectorize(['int64(int64)'])
def _vec_fac(n): return fac(n)
@vectorize(['int64(int64)'])
def _vec_fac2(n): return fac2(n)
@vectorize(['int64(int64)'])
def _vec_dfac21(n): return dfac21(n)


################################
# Matrix packing and reshaping #
################################

@jit(nopython=True, cache=True)
def _tri_indices(vals):
    nel = vals.shape[0]
    nbas = int(np.round(np.roots(np.array([1, 1, -2 * vals.shape[0]]))[1]))
    chi0 = np.empty(nel, dtype=np.int64)
    chi1 = np.empty(nel, dtype=np.int64)
    cnt = 0
    for i in prange(nbas):
        for j in prange(i + 1):
            chi0[cnt] = i
            chi1[cnt] = j
            cnt += 1
    return chi0, chi1

@jit(nopython=True, cache=True)
def _triangle(vals):
    nbas = vals.shape[0]
    ndim = nbas * (nbas + 1) // 2
    tri = np.empty(ndim, dtype=np.float64)
    cnt = 0
    for i in prange(nbas):
        for j in prange(i + 1):
            tri[cnt] = vals[i, j]
            cnt += 1
    return tri

@jit(nopython=True, cache=True)
def _square(vals):
    nbas = int(np.round(np.roots(np.array([1, 1, -2 * vals.shape[0]]))[1]))
    square = np.empty((nbas, nbas), dtype=np.float64)
    cnt = 0
    for i in prange(nbas):
        for j in prange(i + 1):
            square[i, j] = vals[cnt]
            square[j, i] = vals[cnt]
            cnt += 1
    return square

@jit(nopython=True, cache=True)
def _flat_square_to_triangle(flat):
    cnt, ndim = 0, np.int64(np.sqrt(flat.shape[0]))
    tri = np.empty(ndim * (ndim + 1) // 2)
    for i in prange(ndim):
        for j in prange(i + 1):
            tri[cnt] = flat[i * ndim + j]
            cnt += 1
    return tri


#######################
# Basis set expansion #
#######################

@njit
def _enum_cartesian(L):
    # Gen1Int ordering
    # for z in range(L + 1):
    #     for y in range(L + 1 - z):
    #         yield (L - y - z, y, z)
    # Naive combinations with replacements
    # for i in range(L, -1, -1):
    #     for j in range(L, -1, -1):
    #         for k in range(L, -1, -1):
    #             if i + j + k == L:
    #                 yield (i, j, k)
    # Double loop CWR
    for x in range(L, -1, -1):
        for z in range(L + 1 - x):
            yield (x, L - x - z, z)

@njit
def _enum_spherical(L, increasing=True):
    if increasing:
        for m in range(-L, L + 1):
            yield L, m
    else:
        for m in range(L + 1):
            if not m:
                yield L, m
            else:
                for i in (m, -m):
                    yield L, i



#####################
# Basis set classes #
#####################

shell_type = deferred_type()

@jitclass([('L', int64), ('nprim', int64), ('ncont', int64),
           ('spherical', boolean), ('gaussian', boolean),
           ('alphas', float64[:]), ('_coef', float64[:]),
           ('rs', optional(int64[:])), ('ns', optional(int64[:]))])
class Shell(object):

    def dims(self):
        return self.nprim, self.ncont

    def contract(self):
        x = 0
        rect = np.empty((self.nprim, self.ncont))
        for i in range(self.nprim):
            for j in range(self.ncont):
                rect[i, j] = self._coef[x]
                x += 1
        return rect

    def norm_contract(self):
        if not self.gaussian:
            return self._sto_norm_contract()
        if self.spherical:
            return self._sphr_norm_contract()
        return self._cart_norm_contract()

    def enum_cartesian(self):
        return _enum_cartesian(self.L)

    def enum_spherical(self):
        return _enum_spherical(self.L)

    def _cart_norm_contract(self):
        # float is (2 * np.pi) ** -0.75
        return self._norm_cont_kernel(0.251979435538381)

    def _sphr_norm_contract(self):
        # float is (2 / np.pi) ** 0.25
        prefact = 0.893243841738002 / np.sqrt(dfac21(self.L))
        return self._norm_cont_kernel(prefact)

    def _norm_cont_kernel(self, pre):
        coef = self.contract()
        ltot = self.L + 1.5
        lhaf = ltot / 2
        prim, cont = coef.shape
        for c in range(cont):
            norm = 0.
            for pi in range(prim):
                for pj in range(prim):
                    ovl = (2 * (np.sqrt(self.alphas[pi] * self.alphas[pj])
                                     / (self.alphas[pi] + self.alphas[pj]))) ** ltot
                    norm += coef[pi, c] * coef[pj, c] * ovl
            norm = pre / np.sqrt(norm)
            for p in range(prim):
                coef[p, c] *= norm * (4.0 * self.alphas[p]) ** lhaf
        return coef

    def _sto_norm_contract(self):
        # Assumes no contractions
        return np.array(self._sto_norm()) * self.contract()

    def _sto_norm(self):
        return [((2 * a) ** n * ((2 * a) / fac(2 * n)) ** 0.5,)
                for a, n in zip(self.alphas, self.ns)]

    def __init__(self, coef, alphas, nprim, ncont, L,
                 spherical, gaussian, rs=None, ns=None):
        self.spherical = spherical
        self.gaussian = gaussian
        self.alphas = alphas
        self._coef = coef
        self.nprim = nprim
        self.ncont = ncont
        self.L = L
        self.rs = rs
        self.ns = ns


shell_type.define(Shell.class_type.instance_type)
