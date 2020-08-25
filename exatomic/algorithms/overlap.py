# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Overlap computation
######################
Utilities for computing the overlap between gaussian type functions.
"""

import numpy as np
from numba import jit, prange
from .numerical import fac, fac2, dfac21, sdist, choose
from .car2sph import car2sph_scaled
from exatomic.base import nbche

#################################
# Primitive cartesian integrals #
#################################

@jit(nopython=True, nogil=True, cache=nbche)
def _fj(j, l, m, a, b):
    """From Handbook of Computational Quantum Chemistry by David B. Cook
    in chapter 7.7.1 -- Essentially a FOILing of the pre-exponential
    cartesian power dependence in one dimension."""
    tot, i, f = 0., max(0, j - m), min(j, l) + 1
    for k in prange(i, f):
        tot += (choose(l, k) *
                choose(m, int(j - k)) *
                a ** (l - k) *
                b ** (m + k - j))
    return tot

@jit(nopython=True,nogil=True, cache=nbche)
def _nin(l, m, pa, pb, p, N):
    """From Handbook of Computational Quantum Chemistry by David B. Cook
    in chapter 7.7.1 -- Sums the result of _fj over the total angular momentum
    in one dimension."""
    ltot = l + m
    if not ltot: return N
    tot = 0.
    for j in prange(int(ltot // 2 + 1)):
        tot += (_fj(2 * j, l, m, pa, pb) *
                dfac21(j) / (2 * p) ** j)
    return tot * N

@jit(nopython=True, nogil=True, cache=nbche)
def _gaussian_product(a, b, ax, ay, az, bx, by, bz):
    """
    From Molecular Electronic-Structure Theory by Trygve Helgaker et al.
    Computes a product gaussian following section 9.2.3; see equations
    9.2.10 through 9.2.15.
    """
    p = a + b
    mu = a * b / p
    px = (a * ax + b * bx) / p
    py = (a * ay + b * by) / p
    pz = (a * az + b * bz) / p
    ab2 = sdist(ax, ay, az, bx, by, bz)
    return (np.sqrt(np.pi / p), p, mu, ab2,
            px - ax, py - ay, pz - az,
            px - bx, py - by, pz - bz)

@jit(nopython=True, nogil=True, cache=nbche)
def _primitive_overlap_product(l1, m1, n1, l2, m2, n2,
                               N, p, mu, ab2, pax, pay, paz, pbx, pby, pbz):
    """Compute primitive cartesian overlap integral in terms of a gaussian product."""
    return (np.exp(-mu * ab2) * _nin(l1, l2, pax, pbx, p, N)
                              * _nin(m1, m2, pay, pby, p, N)
                              * _nin(n1, n2, paz, pbz, p, N))

@jit(nopython=True, nogil=True, cache=nbche)
def _primitive_overlap(a1, a2, ax, ay, az, bx, by, bz, l1, m1, n1, l2, m2, n2):
    """Compute a primitive cartesian overlap integral."""
    #N, p, mu, ab2, pax, pay, paz, pbx, pby, pbz = \
    p = _gaussian_product(a1, a2, ax, ay, az, bx, by, bz)
    return _primitive_overlap_product(l1, m1, n1, l2, m2, n2, *p)

@jit(nopython=True, nogil=True, cache=nbche)
def _primitive_kinetic(a1, a2, ax, ay, az, bx, by, bz, l1, m1, n1, l2, m2, n2):
    """Compute the kinetic energy as a linear combination of overlap terms."""
    #N, p, mu, ab2, pax, pay, paz, pbx, pby, pbz = \
    p = _gaussian_product(a1, a2, ax, ay, az, bx, by, bz)
    t =  4 * a1 * a2 * _primitive_overlap_product(l1 - 1, m1, n1, l2 - 1, m2, n2, *p)
    t += 4 * a1 * a2 * _primitive_overlap_product(l1, m1 - 1, n1, l2, m2 - 2, n2, *p)
    t += 4 * a1 * a2 * _primitive_overlap_product(l1, m1, n1 - 1, l2, m2, n2 - 1, *p)
    if l1 and l2:
        t += l1 * l2 * _primitive_overlap_product(l1 - 1, m1, n1, l2 - 1, m2, n2, *p)
    if m1 and m2:
        t += l1 * l2 * _primitive_overlap_product(l1, m1 - 1, n1, l2, m2 - 1, n2, *p)
    if n1 and n2:
        t += l1 * l2 * _primitive_overlap_product(l1, m1, n1 - 1, l2, m2, n2 - 1, *p)
    if l1: t -=  2 * a2 * l1 * _primitive_overlap_product(l1 - 1, m1, n1, l2 + 1, m2, n2, *p)
    if l2: t -=  2 * a1 * l2 * _primitive_overlap_product(l1 + 1, m1, n1, l2 - 1, m2, n2, *p)
    if m1: t -=  2 * a2 * m1 * _primitive_overlap_product(l1, m1 - 1, n1, l2, m2 + 1, n2, *p)
    if m2: t -=  2 * a1 * m2 * _primitive_overlap_product(l1, m1 + 1, n1, l2, m2 - 1, n2, *p)
    if n1: t -=  2 * a2 * n1 * _primitive_overlap_product(l1, m1, n1 - 1, l2, m2, n2 + 1, *p)
    if n2: t -=  2 * a1 * n2 * _primitive_overlap_product(l1, m1, n1 + 1, l2, m2, n2 - 1, *p)
    return t / 2

######################################
# Generators over shells/shell-pairs #
######################################

@jit(nopython=True, nogil=True, cache=False)
def _iter_atom_shells(ptrs, xyzs, *shls):
    """Generator yielding indices, atomic coordinates and basis set shells."""
    nshl = len(ptrs)
    for i in range(nshl):
        pa, pi = ptrs[i]
        yield (i, xyzs[pa][0], xyzs[pa][1], xyzs[pa][2], shls[pi])

@jit(nopython=True, nogil=True, cache=False)
def _iter_atom_shell_pairs(ptrs, xyzs, *shls):
    """Generator yielding indices, atomic coordinates and basis set
    shells in block-pair order."""
    nshl = len(ptrs)
    for i in range(nshl):
        for j in range(i + 1):
            pa, pi = ptrs[i]
            pb, pj = ptrs[j]
            yield (i, j, xyzs[pa][0], xyzs[pa][1], xyzs[pa][2],
                         xyzs[pb][0], xyzs[pb][1], xyzs[pb][2],
                         shls[pi], shls[pj])

############################################
# Integral processing for of Shell objects #
############################################

@jit(nopython=True, nogil=True, cache=nbche)
def _cartesian_overlap_shell(xa, ya, za, xb, yb, zb,
                             li, mi, ni, lj, mj, nj,
                             ialpha, jalpha):
    """Compute pairwise cartesian integrals exponents in a block-pair."""
    pints = np.empty((len(ialpha), len(jalpha)))
    for i, ia in enumerate(ialpha):
        for j, ja in enumerate(jalpha):
            pints[i, j] = _primitive_overlap(ia, ja,
                                             xa, ya, za, xb, yb, zb,
                                             li, mi, ni, lj, mj, nj)
    return pints

@jit(nopython=True, nogil=True, cache=nbche)
def _cartesian_shell_pair(ax, ay, az, bx, by, bz, ishl, jshl):
    """Compute fully contracted block-pair integrals including
    expansion of angular momentum dependence."""
    inrm = ishl.norm_contract()
    jnrm = jshl.norm_contract()
    ideg = (ishl.L + 1) * (ishl.L + 2) // 2
    jdeg = (jshl.L + 1) * (jshl.L + 2) // 2
    pint = np.empty((ideg * ishl.nprim, jdeg * jshl.nprim))
    for magi, (li, mi, ni) in enumerate(ishl.enum_cartesian()):
        for magj, (lj, mj, nj) in enumerate(jshl.enum_cartesian()):
            ianc = magi * ishl.nprim
            janc = magj * jshl.nprim
            pint[ianc : ianc + ishl.nprim,
                 janc : janc + jshl.nprim] = \
                    _cartesian_overlap_shell(ax, ay, az, bx, by, bz,
                                             li, mi, ni, lj, mj, nj,
                                             ishl.alphas, jshl.alphas)
    if ishl.L:
        inrm = np.kron(np.eye(ideg), inrm)
        if ishl.spherical:
            inrm = np.dot(inrm, np.kron(car2sph_scaled(ishl.L),
                                        np.eye(ishl.ncont)))
    if jshl.L:
        jnrm = np.kron(np.eye(jdeg), jnrm)
        if jshl.spherical:
            jnrm = np.dot(jnrm, np.kron(car2sph_scaled(jshl.L),
                                        np.eye(jshl.ncont)))
    return np.dot(inrm.T, np.dot(pint, jnrm))

@jit(nopython=True, nogil=True, cache=False)
def _cartesian_shell_pairs(ndim, ptrs, xyzs, *shls):
    """Construct a full square (overlap) integral matrix."""
    cart = np.zeros((ndim, ndim))
    ii = 0
    for i, j, ax, ay, az, bx, by, bz, ishl, jshl \
        in _iter_atom_shell_pairs(ptrs, xyzs, *shls):
        if not j: jj = 0
        cint = _cartesian_shell_pair(ax, ay, az, bx, by, bz, ishl, jshl)
        iblk, jblk = cint.shape
        cart[ii : ii + iblk, jj : jj + jblk] = cint
        if i != j: cart[jj : jj + jblk, ii : ii + iblk] = cint.T
        else: ii += iblk
        jj += jblk
    return cart

##################################
# Obara-Saika recursion relation #
##################################

@jit(nopython=True, nogil=True, cache=nbche)
def _obara_s_recurr(p, l, m, pa, pb, s):
    """There is a bug in this function. Do not use."""
    if not l + m: return s
    p2 = 1 / (2 * p)
    s0 = np.zeros((l + 1, m + 1))
    s0[0, 0] = s
    if l: s0[1, 0] = pa * s
    if m: s0[0, 1] = pb * s
    if l and m: s0[1, 1] = pb * s0[1, 0] + p2 * s
    for i in range(1, l):
        for j in range(1, m):
            mul = p2 * (i * s0[i - 1, j] + j * s0[i, j - 1])
            s0[i + 1, j] = pa * s0[i, j] + mul
            s0[i, j + 1] = pb * s0[i, j] + mul
            s0[i + 1, j + 1] = pa * s0[i, j + 1] + p2 * ((i + 1) * s0[i, j] + j * s0[i + 1, j])
    return s0[l, m]


@jit(nopython=True, nogil=True, cache=nbche)
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
