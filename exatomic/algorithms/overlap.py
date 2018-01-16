# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Overlap computation
######################
Utilities for computing the overlap between gaussian type functions.
"""

import numpy as np
from numba import jit

from ..core.basis import Overlap
from .basis import (gen_bfns, new_solid_harmonics,
                    new_car2sph, new_enum_cartesian,
                    fac, fac2, cart_lml_count)



def _overlap(bfns, p2c, c, x, y, z, l, m, n, a):
    """Build arrays for numbafied overlap."""
    ix, pf = 0, bfns[0]
    for j, f in enumerate(bfns):
        ii = f.ps
        if pf.l != f.l or pf.m != f.m or pf.n != f.n or not j:
            if j: ix += pf.ps
            c[ix : ix + ii] = f.center
            x[ix : ix + ii] = f.xa
            y[ix : ix + ii] = f.ya
            z[ix : ix + ii] = f.za
            l[ix : ix + ii] = f.l
            m[ix : ix + ii] = f.m
            n[ix : ix + ii] = f.n
            a[ix : ix + ii] = f.alphas
        p2c[ix : ix + ii, j] = f.cs
        pf = f
    # return c, x, y, z, l, m, n, a
    chi0, chi1, ovl = _wrap_overlap(c, x, y, z, l, m, n, a)
    return p2c, Overlap.from_dict({'chi0': chi0, 'chi1': chi1,
                                   'coef': ovl, 'frame': 0})


_cartouche = new_car2sph(new_solid_harmonics(6),
                         new_enum_cartesian)

def _cart_to_sphr(shls, sets, ncc, ncs):
    """Generate cartesian to spherical transform matrix."""
    c2s = np.zeros((ncc, ncs), dtype=np.float64)
    cx, sx = 0, 0
    for seht in sets:
        shl = shls[seht]
        for L, n in shl.items():
            exp = np.kron(_cartouche[L], np.eye(n)) if L else np.eye(n)
            ci, si = exp.shape
            c2s[cx : cx + ci, sx : sx + si] = exp
            cx += ci
            sx += si
    return c2s

def cart_to_sphr(uni):
    """Compute the cartesian to spherical contraction matrix."""
    ncc = uni.atom.set.map(uni.basis_set.functions(False)
                           .groupby('set').sum()).sum()
    ncs = uni.atom.set.map(uni.basis_set.functions(True)
                           .groupby('set').sum()).sum()
    shls = uni.basis_set.functions_by_shell()
    return _cart_to_sphr(shls, uni.atom.set, ncc, ncs)

def prim_cart_overlap(uni):
    """Compute the primitive overlap and contraction matrices from a universe."""
    # Generate the basis functions
    if hasattr(uni, 'basis_functions'):
        bfns = uni.basis_functions[0]
    else:
        uni.basis_functions = {0: gen_bfns(uni, forcecart=True)}
        bfns = uni.basis_functions[0]
    # Number of functions per shell
    shls = uni.basis_set.functions_by_shell()
    # Number of primitive cartesians
    npc = uni.atom.set.map(uni.basis_set.primitives(False)
                           .groupby('set').sum()).sum()
    # Number of contracted cartesians
    ncc = uni.atom.set.map(uni.basis_set.functions(False)
                           .groupby('set').sum()).sum()
    # Number of contracted sphericals
    ncs = uni.atom.set.map(uni.basis_set.functions(True)
                           .groupby('set').sum()).sum()
    # Cartesian to spherical transformation matrix
    c2s = np.zeros((ncc, ncs), dtype=np.float64)
    c2s = _cart_to_sphr(shls, uni.atom.set, ncc, ncs)
    # Allocate arrays to feed to numba
    p2c = np.zeros((npc, ncc), dtype=np.float64)
    c = np.empty(npc, dtype=np.int64)
    x = np.empty(npc, dtype=np.float64)
    y = np.empty(npc, dtype=np.float64)
    z = np.empty(npc, dtype=np.float64)
    l = np.empty(npc, dtype=np.int64)
    m = np.empty(npc, dtype=np.int64)
    n = np.empty(npc, dtype=np.int64)
    a = np.empty(npc, dtype=np.float64)
    p2c, ovl = _overlap(bfns, p2c, c, x, y, z, l, m, n, a)
    return p2c, c2s, ovl



@jit(nopython=True, cache=True)
def _overlap_1c(alp1, alp2, l1, m1, n1, l2, m2, n2):
    """Compute overlap between gaussian functions on the same center."""
    ll = l1 + l2
    mm = m1 + m2
    nn = n1 + n2
    if ll % 2 or mm % 2 or nn % 2: return 0
    ltot = ll // 2 + mm // 2 + nn // 2
    numer = np.pi ** (1.5) * fac2(ll - 1) * fac2(mm - 1) * fac2(nn - 1)
    denom = (2 ** ltot) * (alp1 + alp2) ** (ltot + 1.5)
    return numer / denom


@jit(nopython=True, cache=True)
def _nin(o1, o2, po1, po2, gamma, pg12):
    """Helper function for gaussian overlap between 2 centers."""
    otot = o1 + o2
    if not otot: return pg12
    if otot % 2: otot -= 1
    oio = 0.
    for i in range(otot // 2 + 1):
        k = 2 * i
        prod = pg12 * fac2(k - 1) / ((2 * gamma) ** i)
        qlo = max(-k, (k - 2 * o2))
        qhi = min( k, (2 * o1 - k)) + 1
        fk = 0.
        for q in range(qlo, qhi, 2):
            xx = (k + q) // 2
            zz = (k - q) // 2
            newt1 = fac(o1) / fac(xx) / fac(o1 - xx)
            newt2 = fac(o2) / fac(zz) / fac(o2 - zz)
            fk += newt1 * newt2 * (po1 ** (o1 - xx)) * (po2 ** (o2 - zz))
        oio += prod * fk
    return oio


@jit(nopython=True, cache=True)
def _overlap_2c(ab2, pax, pay, paz, pbx, pby, pbz,
                gamma, alp1, alp2, l1, m1, n1, l2, m2, n2):
    """Compute the overlap between two gaussian functions on different centers."""
    pg12 = np.sqrt(np.pi / gamma)
    xix = _nin(l1, l2, pax, pbx, gamma, pg12)
    yiy = _nin(m1, m2, pay, pby, gamma, pg12)
    ziz = _nin(n1, n2, paz, pbz, gamma, pg12)
    exp = alp1 * alp2 * ab2 / gamma
    return np.exp(-exp) * xix * yiy * ziz


@jit(nopython=True, cache=True)
def _wrap_overlap(c, x, y, z, l, m, n, a):
    """Compute a triangular portion of the primitive overlap matrix."""
    nprim, cnt = len(x), 0
    arlen = nprim * (nprim + 1) // 2
    chi0 = np.empty(arlen, dtype=np.int64)
    chi1 = np.empty(arlen, dtype=np.int64)
    ovl = np.empty(arlen, dtype=np.float64)
    for i in range(nprim):
        for j in range(i + 1):
            chi0[cnt] = i
            chi1[cnt] = j
            if c[i] == c[j]:
                ovl[cnt] = _overlap_1c(
                    a[i], a[j], l[i], m[i], n[i], l[j], m[j], n[j])
            else:
                gamma = a[i] + a[j]
                xp = (a[i] * x[i] + a[j] * x[j]) / gamma
                yp = (a[i] * y[i] + a[j] * y[j]) / gamma
                zp = (a[i] * z[i] + a[j] * z[j]) / gamma
                pix = xp - x[i]
                piy = yp - y[i]
                piz = zp - z[i]
                pjx = xp - x[j]
                pjy = yp - y[j]
                pjz = zp - z[j]
                ovl[cnt] = _overlap_2c(
                    (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 + (z[i] - z[j]) ** 2,
                    pix, piy, piz, pjx, pjy, pjz, gamma,
                    a[i], a[j], l[i], m[i], n[i], l[j], m[j], n[j])
            cnt += 1
    return chi0, chi1, ovl
