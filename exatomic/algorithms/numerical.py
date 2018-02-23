# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Numerical methods and classes
###############################
Everything in this module is implemented in numba.
"""
import numpy as np
from numba import (jit, njit, jitclass, vectorize, prange,
                   deferred_type, optional, int64, float64)

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

@jit(nopython=True, cache=True)
def _find(arr, target, start=0, stop=0):
    while arr[start] != target: start += 1
    stop = start
    while arr[stop] == target: stop += 1
    return start, stop


#############################
# Analytical normalizations #
#############################

@jit(nopython=True, cache=True)
def normalize(alpha, L):
    prefac = (2 / np.pi) ** (0.75)
    numer = 2 ** L * alpha ** ((L + 1.5) / 2)
    denom = dfac21(L) ** 0.5
    return prefac * numer / denom

@jit(nopython=True, cache=True)
def new_normalize(alpha, L):
    prefac = 2 * (2 * alpha) ** 0.75 / np.pi ** 0.25
    sqrt = np.sqrt(2 ** L / fac2(2 * L + 1))
    last = np.sqrt(2 * alpha) ** L
    return prefac * sqrt * last

@jit(nopython=True, cache=True)
def _new_prim_sphr_norm(alphas, L):
    Ns = np.empty(len(alphas), dtype=np.float64)
    for i, a in enumerate(alphas):
        Ns[i] = new_normalize(a, L)
    return Ns

@jit(nopython=True, cache=True)
def _prim_sphr_norm(alphas, L):
    Ns = np.empty(len(alphas), dtype=np.float64)
    for i, a in enumerate(alphas):
        Ns[i] = normalize(a, L)
    return Ns

@vectorize('float64(float64,int64)')
def _vec_sphr_norm(alpha, L):
    return normalize(alpha, L)
@vectorize('float64(float64,int64)')
def _new_vec_sphr_norm(alpha, L):
    return new_normalize(alpha, L)

@jit(nopython=True, cache=True)
def prim_normalize(alpha, l, m, n):
    numer = dfac21(l) * dfac21(m) * dfac21(n)
    denom = alpha ** (l + m + n)
    prefa = (np.pi / (2 * alpha)) ** 1.5
    return 1 / np.sqrt(prefa * numer / denom)
@jit(nopython=True, cache=True)
def _prim_cart_norm(alphas, l, m, n):
    Ns = np.empty(len(alphas), dtype=np.float64)
    for i, a in enumerate(alphas):
        Ns[i] = prim_normalize(a, l, m, n)
    return Ns
@vectorize('float64(float64,int64,int64,int64)')
def _vec_sto_norm(alpha, l, m, n):
    return prim_normalize(alpha, l, m, n)

@jit(nopython=True, cache=True)
def sto_normalize(alpha, n):
    return (2 * alpha) ** n * ((2 * alpha) / fac(2 * n)) ** 0.5
@jit(nopython=True, cache=True)
def _prim_sto_norm(alphas, n):
    Ns = np.empty(len(alphas), dtype=np.float64)
    for i, a in enumerate(alphas):
        Ns[i] = sto_normalize(a, n)
    return Ns
@vectorize('float64(float64,int64)')
def _vec_sto_norm(alpha, n):
    return sto_normalize(alpha, n)
@njit
def _norm_sto(alphas, ns):
    Ns = np.empty(len(alphas), dtype=np.float64)
    for i, (a, n) in enumerate(zip(alphas, ns)):
        Ns[i] = sto_normalize(a, n)
    return Ns

@jit(nopython=True, cache=True)
def _cont_norm(ds, alphas, Ns, l, m, n):
    tot = 0.
    for di, ai, ni in zip(ds, alphas, Ns):
        for dj, aj, nj in zip(ds, alphas, Ns):
            t = 1 / (ai + aj)
            S00 = np.pi ** 1.5 * t ** 1.5 * ni * nj
            t = 0.5 * t
            t1 = dfac21(l) * t ** l
            t2 = dfac21(m) * t ** m
            t3 = dfac21(n) * t ** n
            tot += di * dj * S00 * t1 * t2 * t3
    return 1 / np.sqrt(tot)

@jit(nopython=True, cache=True)
def _new_cont_norm(alphas, l, m, n):
    a = alphas.size
    a = np.empty((a, a), dtype=np.float64)
    p = dfac21(l) * dfac21(m) * dfac21(n)
    for i, ai in enumerate(alphas):
        for j, aj in enumerate(alphas):
            ak = 1 / (ai + aj)
            pr = (np.pi * ak) ** 1.5
            a[i, j] = p * pr * (ak / 2) ** (l + m + n)
    return a

#@jit(nopython=True, cache=True)
@njit
def _norm_cont_sphr(L, alphas, coef):
    prefac = 0.893243841738002 # (2 / np.pi) ** 0.25
    pdim, cdim = coef.shape
    ltot = L + 1.5
    ptot = 0.5 * ltot
    angfac = 1.0
    for ang in range(3, 2 * L + 1, 2):
        angfac *= ang
    angfac = prefac / np.sqrt(angfac)
    for c in range(cdim):
        norm = 0.
        for i in range(pdim):
            for j in range(pdim):
                ovl = (2 * (np.sqrt(alphas[i] * alphas[j]) /
                                   (alphas[i] + alphas[j]))) ** ltot
                norm += coef[i, c] * coef[j, c] * ovl
        norm = angfac / np.sqrt(norm)
        coef[:, c] *= norm * (4. * alphas) ** ptot
    return coef


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


#######################
# Basis set expansion #
#######################

@njit
def _enum_cartesian(L):
    for i in range(L, -1, -1):
        for j in range(L, -1, -1):
            for k in range(L, -1, -1):
                if i + j + k == L:
                    yield (i, j, k)

@njit
def _enum_spherical(L, increasing=True):
    if increasing:
        for m in range(-L, L + 1):
            yield m
    else:
        for m in range(L + 1):
            if not m:
                yield m
            else:
                for i in (m, -m):
                    yield i


#####################
# Basis set classes #
#####################

shell_type = deferred_type()

@jitclass([('alphas', float64[:]), ('shells', int64[:]), ('L', int64),
           ('_data', float64[:]), ('_pdim', int64), ('_cdim', int64),
           ('rs', optional(int64[:])), ('ns', optional(int64[:]))])
class Shell(object):

    def dims(self):
        return self._pdim, self._cdim

    def _prim_cart_norm(self, l, m, n):
        return _prim_cart_norm(self.alphas, l, m, n)

    def _prim_sphr_norm(self):
    #     return _norm_cont_sphr(self.L, self.alphas, self.contract())
        return _prim_sphr_norm(self.alphas, self.L)
        # return _new_prim_sphr_norm(self.alphas, self.L)

    def _cont_cart_norm(self, l, m, n):
        N = np.empty(self._cdim, dtype=np.float64)
        Ns = self._prim_cart_norm(l, m, n)
        Ns = np.outer(Ns, Ns)
        ak = _new_cont_norm(self.alphas, l, m, n)
        con = self.contract()
        for i in range(self._cdim):
            ds = np.outer(con[:, i], con[:, i])
            N[i] = 1 / np.sqrt((ak * Ns * ds).sum())
        return N

    def expand_cartesian(self):
        return _enum_cartesian(self.L)

    def expand_spherical(self):
        return _enum_spherical(self.L)

    def sto_norm_contract(self):
        return _norm_sto(self.alphas, self.ns) * self.contract()

    def cart_norm_contract(self, l, m, n):
        P = self._prim_cart_norm(l, m, n)
        N = self._cont_cart_norm(l, m, n)
        return np.outer(P, N) * self.contract()

    def new_cart_norm_contract(self, l, m, n):
        P = self._prim_cart_norm(l, m, n)
        N = self._cont_cart_norm(l, m, n)
        n = np.sqrt(dfac21(l + m + n) / (dfac21(l) * dfac21(m) * dfac21(n)))
        return n * np.outer(P, N) * self.contract()

    def sphr_norm_contract(self):
        #return _norm_cont_sphr(self.L, self.alphas, self.contract())
        P = self._prim_sphr_norm()
        N = np.ones(1, dtype=np.float64)
        return np.outer(P, N) * self.contract()

    def contracT(self):
        rect = np.empty((self._cdim, self._pdim),
                        dtype=np.float64)
        for j in range(self._pdim):
            for i in range(self._cdim):
                rect[i, j] = self._data[j * self._cdim + i]
        return rect

    def contract(self):
        x = 0
        rect = np.empty((self._pdim, self._cdim),
                        dtype=np.float64)
        for i in range(self._pdim):
            for j in range(self._cdim):
                rect[i, j] = self._data[x]
                x += 1
        return rect

    def __init__(self, data, alphas, shells,
                 pdim, cdim, L, rs=None, ns=None):
        self.alphas = alphas
        self.shells = shells
        self._data = data
        self._pdim = pdim
        self._cdim = cdim
        self.L = L
        self.rs = rs
        self.ns = ns

shell_type.define(Shell.class_type.instance_type)


########################
# Basis set algorithms #
########################

@njit
def _iter_atoms_shells(atoms, xyzs, sets, *shells):
    for center, seht in enumerate(atoms):
        x, y, z = xyzs[center]
        seti, setf = _find(sets, seht)
        for Li in range(seti, setf):
            shell = shells[Li]
            yield center, x, y, z, shell

@njit
def _iter_atoms_shells_cart(atoms, xyzs, sets, *shells):
    for center, x, y, z, shell in _iter_atoms_shells(atoms, xyzs, sets,
                                                     *shells):
        ii, jj = shell.dims()
        for l, m, n in shell.expand_cartesian():
            yield center, x, y, z, shell, ii, jj, l, m, n

@njit
def _iter_atoms_shells_sphr(atoms, xyzs, sets, *shells):
    for center, x, y, z, shell in _iter_atoms_shells(atoms, xyzs, sets,
                                                     *shells):
        ii, jj = shell.dims()
        for m in shell.expand_spherical():
            yield center, x, y, z, shell, ii, jj, shell.L, m

@njit
def _gen_p2c(atoms, xyzs, sets, npc, ncc, *shells):
    p2c = np.zeros((npc, ncc), dtype=np.float64)
    i, j = 0, 0
    for center, x, y, z, shell, ii, jj, l, m, n in \
        _iter_atoms_shells_cart(atoms, xyzs, sets, *shells):
        p2c[i : i + ii, j : j + jj] = shell.contract()
        i += ii
        j += jj
    return p2c


#@njit
def _gen_prims(atoms, xyzs, sets, npc, ncc, *shells):
    p2c = np.zeros((npc, ncc), dtype=np.float64)
    ra = np.empty(npc, dtype=np.float64)
    rx = np.empty(npc, dtype=np.float64)
    ry = np.empty(npc, dtype=np.float64)
    rz = np.empty(npc, dtype=np.float64)
    rc = np.empty(npc, dtype=np.int64)
    rl = np.empty(npc, dtype=np.int64)
    rm = np.empty(npc, dtype=np.int64)
    rn = np.empty(npc, dtype=np.int64)
    i, j = 0, 0
    for center, x, y, z, shell, ii, jj, l, m, n in \
        _iter_atoms_shells_cart(atoms, xyzs, sets, *shells):
        p2c[i : i + ii, j : j + jj] = shell.cart_norm_contract(l, m, n)
        ra[i : i + ii] = shell.alphas
        for k in range(ii):
            rc[i + k] = center
            rl[i + k] = l
            rm[i + k] = m
            rn[i + k] = n
            rx[i + k] = x
            ry[i + k] = y
            rz[i + k] = z
        i += ii
        j += jj
    return p2c, rc, rx, ry, rz, rl, rm, rn, ra


#@njit
import pandas as pd
def _new_gen_prims(atoms, xyzs, sets, npc, ncc, *shells):
    p2c = np.zeros((npc, ncc), dtype=np.float64)
    ra = np.empty(npc, dtype=np.float64)
    rx = np.empty(npc, dtype=np.float64)
    ry = np.empty(npc, dtype=np.float64)
    rz = np.empty(npc, dtype=np.float64)
    rc = np.empty(npc, dtype=np.int64)
    rl = np.empty(npc, dtype=np.int64)
    rm = np.empty(npc, dtype=np.int64)
    rn = np.empty(npc, dtype=np.int64)
    i, j = 0, 0
    for center, x, y, z, shell, ii, jj, l, m, n in \
        _iter_atoms_shells_cart(atoms, xyzs, sets, *shells):
        p2c[i : i + ii, j : j + jj] = shell.new_cart_norm_contract(l, m, n)
        print(i, j, l, m, n, pd.DataFrame(p2c[i : i + ii, j : j + jj]))
        ra[i : i + ii] = shell.alphas
        for k in range(ii):
            rc[i + k] = center
            rl[i + k] = l
            rm[i + k] = m
            rn[i + k] = n
            rx[i + k] = x
            ry[i + k] = y
            rz[i + k] = z
        i += ii
        j += jj
    return p2c, rc, rx, ry, rz, rl, rm, rn, ra





# _linked_cartesian_type = deferred_type()
# _linked_spherical_type = deferred_type()
#
# _typs = [int64, float64, float64[:]]
# _ints = ['l', 'm', 'n', 'c', 'ps']
# _flts = ['xa', 'ya', 'za', 'N']
# _arrs = ['alphas', 'ds', 'Ns', 'cs']
#
# _cspec = [(attr, typ) for typ, attrs in
#           zip(_typs, [_ints, _flts, _arrs]) for attr in attrs]
# _cspec.append(('next', optional(_linked_cartesian_type)))
#
# @jitclass(_cspec)
# class _LinkedCartesian(object):
#
#     def contraction(self):
#         return self.Ns * self.ds * self.N
#
#     def __init__(self, cen, xa, ya, za, ds, alphas, l, m, n, nxt):
#         self.c = cen
#         self.Ns = _prim_cart_norm(alphas, l, m, n)
#         self.N = _cont_norm(ds, alphas, self.Ns, l, m, n)
#         # self.cs = self.N * self.Ns * ds
#         self.ps = len(self.Ns)
#         self.alphas = alphas
#         self.ds = ds
#         self.xa = xa
#         self.ya = ya
#         self.za = za
#         self.l = l
#         self.m = m
#         self.n = n
#         self.next = nxt
#
#
# _ints = ['L', 'ml', 'c', 'ps']
#
# _sspec = [(attr, typ) for typ, attrs in
#           zip(_typs, [_ints, _flts, _arrs]) for attr in attrs]
#
# _sspec.append(('next', optional(_linked_spherical_type)))
# @jitclass(_sspec)
# class _LinkedSpherical(object):
#
#     def contraction(self):
#         return self.Ns * self.ds
#
#     def __init__(self, cen, xa, ya, za, ds, alphas, L, ml, nxt):
#         self.c = cen
#         self.Ns = _prim_sphr_norm(alphas, L)
#         self.ps = len(self.Ns)
#         self.alphas = alphas
#         self.ds = ds
#         self.xa = xa
#         self.ya = ya
#         self.za = za
#         self.ml = ml
#         self.N = 1
#         self.L = L
#         self.next = nxt
#
# def _reverse_linked(f):
#     r = None
#     while f is not None:
#         c = f
#         f = f.next
#         c.next = r
#         r = c
#     return r
#
# _linked_cartesian_type.define(_LinkedCartesian.class_type.instance_type)
# _linked_spherical_type.define(_LinkedSpherical.class_type.instance_type)
#
#
# @njit
# def _numba_primitive_arrays(f, npc, ncc):
#     """Generate 1D arrays for computing integrals. With singly-linked
#     list numba types, it is easier to populate the arrays backwards,
#     or reverse the link references first."""
#     p2c = np.zeros((npc, ncc), dtype=np.float64)
#     c = np.empty(npc, dtype=np.int64)
#     l = np.empty(npc, dtype=np.int64)
#     m = np.empty(npc, dtype=np.int64)
#     n = np.empty(npc, dtype=np.int64)
#     a = np.empty(npc, dtype=np.float64)
#     x = np.empty(npc, dtype=np.float64)
#     y = np.empty(npc, dtype=np.float64)
#     z = np.empty(npc, dtype=np.float64)
#     i, j = 0, 0
#     p = None
#     while f is not None:
#         print(i, j)
#         if (p is None or f.c != p.c or f.l != p.l or
#             f.m != p.m or f.n != p.n):
#             if p is not None:
#                 i += p.ps
#             a[i : i + f.ps] = f.alphas
#             for k in range(f.ps):
#                 c[i + k] = f.c
#                 l[i + k] = f.l
#                 m[i + k] = f.m
#                 n[i + k] = f.n
#                 x[i + k] = f.xa
#                 y[i + k] = f.ya
#                 z[i + k] = f.za
#         p2c[i : i + f.ps, j] = f.contraction()
#         j += 1
#         p = f
#         f = f.next
#     return p2c, c, x, y, z, l, m, n, a
