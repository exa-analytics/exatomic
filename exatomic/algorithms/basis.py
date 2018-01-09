# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Basis Function Manipulation
################################
Functions for managing and manipulating basis set data.
Many of the ordering schemes used in computational codes can be
generated programmatically with the right numerical function.
This is preferred to an explicit parsing and storage of a given
basis set ordering scheme.
"""
import re
from operator import mul
from functools import reduce
from abc import ABC, abstractmethod
from collections import OrderedDict, Counter
from itertools import combinations_with_replacement as cwr

import numpy as np
from numba import jit, vectorize, prange
from numexpr import evaluate

from symengine import var, exp, Add, Mul, Integer
x, y, z = var("x y z")
# _x, _y, _z = var("_x _y _z")

lorder = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'k', 'l', 'm']
_spher = [2 * i + 1 for i in range(len(lorder))]
_carts = [  1,   3,   6,  10,  15,  21,  28]

lmap = OrderedDict([(l, i) for i, l in enumerate(lorder)])
lmap.update([('px', 1), ('py', 1), ('pz', 1)])
rlmap = {value: key for key, value
         in lmap.items() if len(key) == 1}

spher_ml_count = OrderedDict([(l, d) for l, d in zip(lorder, _spher)])
spher_lml_count = OrderedDict([(i, d) for i, d in
                              enumerate(spher_ml_count.values())])
cart_ml_count = OrderedDict([(l, d) for l, d in zip(lorder, _carts)])
cart_lml_count = OrderedDict([(i, d) for i, d in
                             enumerate(cart_ml_count.values())])

enum_cartesian = {0: [[0, 0, 0]],
                  1: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                  2: [[2, 0, 0], [1, 1, 0], [1, 0, 1],
                      [0, 2, 0], [0, 1, 1], [0, 0, 2]],
                  3: [[3, 0, 0], [2, 1, 0], [2, 0, 1],
                      [1, 2, 0], [1, 1, 1], [1, 0, 2],
                      [0, 3, 0], [0, 2, 1], [0, 1, 2],
                      [0, 0, 3]],
                  4: [[4, 0, 0], [3, 1, 0], [3, 0, 1],
                      [2, 2, 0], [2, 1, 1], [2, 0, 2],
                      [1, 3, 0], [1, 2, 1], [1, 1, 2],
                      [1, 0, 3], [0, 4, 0], [0, 3, 1],
                      [0, 2, 2], [0, 1, 3], [0, 0, 4]],
                  5: [[5, 0, 0], [4, 1, 0], [4, 0, 1],
                      [3, 2, 0], [3, 1, 1], [3, 0, 2],
                      [2, 3, 0], [2, 2, 1], [2, 1, 2],
                      [2, 0, 3], [1, 4, 0], [1, 3, 1],
                      [1, 2, 2], [1, 1, 3], [1, 0, 4],
                      [0, 5, 0], [0, 4, 1], [0, 3, 2],
                      [0, 2, 3], [0, 1, 4], [0, 0, 5]]}
gaussian_cartesian = enum_cartesian.copy()
gaussian_cartesian[2] = [[2, 0, 0], [0, 2, 0], [0, 0, 2],
                         [1, 1, 0], [1, 0, 1], [0, 1, 1]]



def new_solid_harmonics(lmax):
    """Symbolic, recursive solid harmonics for the angular component
    of a wave function."""
    def _top_sh(lp, sp, sm):
        kr = int(not lp)
        return (np.sqrt(2 ** kr * (2 * lp + 1) / (2 * lp + 2)) *
                (x * sp - (1 - kr) * y * sm))
    def _mid_sh(lp, m, sm, smm):
        return (((2 * lp + 1) * z * sm - np.sqrt((lp + m) *
                (lp - m)) * (x*x + y*y + z*z) * smm) /
                (np.sqrt((lp + m + 1) * (lp - m + 1))))
    def _bot_sh(lp, sp, sm):
        kr = int(not lp)
        return (np.sqrt(2 ** kr * (2 * lp + 1) / (2 * lp + 2)) *
                (y * sp + (1 - kr) * x * sm))
    sh = OrderedDict([(l, OrderedDict([])) for l in range(lmax + 1)])
    sh[0][0] = Integer(1)
    for l in range(1, lmax + 1):
        lp = l - 1
        mls = list(range(-l, l + 1))
        sh[l][mls[0]] = _bot_sh(lp, sh[lp][lp], sh[lp][-lp])
        for ml in mls[1:-1]:
            try:
                rec = sh[lp - 1][ml]
            except KeyError:
                rec = sh[lp][ml]
            sh[l][ml] = _mid_sh(lp, ml, sh[lp][ml], rec)
        sh[l][mls[-1]] = _top_sh(lp, sh[lp][lp], sh[lp][-lp])
    return sh


def solid_harmonics(lmax):
    """Symbolic, recursive solid harmonics for the angular component
    of a wave function."""
    def _top_sh(lp, sp, sm):
        kr = int(not lp)
        return ((2 ** kr * (2 * lp + 1) / (2 * lp + 2)) ** 0.5 *
                (x * sp - (1 - kr) * y * sm))
    def _mid_sh(lp, m, sm, smm):
        return (((2 * lp + 1) * z * sm - ((lp + m) * (lp - m)) ** 0.5 *
                (x*x + y*y + z*z) * smm) /
                (((lp + m + 1) * (lp - m + 1)) ** 0.5))
    def _bot_sh(lp, sp, sm):
        kr = int(not lp)
        return ((2 ** kr * (2 * lp + 1) / (2 * lp + 2)) ** 0.5 *
                (y * sp + (1 - kr) * x * sm))
    sh = OrderedDict([((0, 0), Integer(1))])
    for l in range(1, lmax + 1):
        lp = l - 1
        mls = list(range(-l, l + 1))
        sh[(l, mls[0])] = _bot_sh(lp, sh[(lp, lp)], sh[(lp, -lp)])
        for ml in mls[1:-1]:
            try:
                rec = sh[(lp - 1, ml)]
            except KeyError:
                rec = sh[(lp, ml)]
            sh[(l, ml)] = _mid_sh(lp, ml, sh[(lp, ml)], rec)
        sh[(l, mls[-1])] = _top_sh(lp, sh[(lp, lp)], sh[(lp, -lp)])
    return sh

# def car2sph(sh, cart):
#     """Turns symbolic solid harmonic functions into a dictionary of
#     arrays containing cartesian to spherical transformation matrices.
#
#     Args
#         sh (OrderedDict): the result of solid_harmonics(l_tot)
#         cart (dict): dictionary of l, cartesian l, m, n ordering
#     """
#     conv, prevL, mlcnt = {}, 0, 0
#     for (L, ml), sym in sh.items():
#         if L > 5: continue
#         mlcnt = mlcnt if prevL == L else 0
#         conv.setdefault(L, np.zeros((cart_lml_count[L],
#                                      spher_lml_count[L]),
#                                     dtype=np.float64))
#         coefs = sym.expand().as_coefficients_dict()
#         for i, (l, m, n) in enumerate(cart[L]):
#             if L == 1:
#                 conv[L] = np.array(cart[L])
#                 break
#             key = x ** l * y ** m * z ** n
#             conv[L][i, mlcnt] = coefs[key]
#         prevL = L
#         mlcnt += 1
#     return conv


new_enum_cartesian = {0: [[0, 0, 0]]}
for L in range(1, 9):
    combs = [Counter(c) for c in cwr('xyz', L)]
    new_enum_cartesian[L] = [[c[i] for i in 'xyz'] for c in combs]


def new_solid_harmonics(lmax):
    """Symbolic, recursive solid harmonics for the angular component
    of a wave function."""
    def _top_sh(lp, sp, sm):
        kr = int(not lp)
        return ((2 ** kr * (2 * lp + 1) / (2 * lp + 2)) ** 0.5 *
                (x * sp - (1 - kr) * y * sm))
    def _mid_sh(lp, m, sm, smm):
        return (((2 * lp + 1) * z * sm - ((lp + m) * (lp - m)) ** 0.5 *
                (x*x + y*y + z*z) * smm) /
                (((lp + m + 1) * (lp - m + 1)) ** 0.5))
    def _bot_sh(lp, sp, sm):
        kr = int(not lp)
        return ((2 ** kr * (2 * lp + 1) / (2 * lp + 2)) ** 0.5 *
                (y * sp + (1 - kr) * x * sm))
    sh = OrderedDict([(l, OrderedDict([])) for l in range(lmax + 1)])
    sh[0][0] = Integer(1)
    for l in range(1, lmax + 1):
        lp = l - 1
        mls = list(range(-l, l + 1))
        sh[l][mls[0]] = _bot_sh(lp, sh[lp][lp], sh[lp][-lp])
        for ml in mls[1:-1]:
            try:
                rec = sh[lp - 1][ml]
            except KeyError:
                rec = sh[lp][ml]
            sh[l][ml] = _mid_sh(lp, ml, sh[lp][ml], rec)
        sh[l][mls[-1]] = _top_sh(lp, sh[lp][lp], sh[lp][-lp])
    return sh


def new_car2sph(sh, cart, orderedp=True):
    conv = {L: np.zeros((cart_lml_count[L],
                        spher_lml_count[L]))
            for L in range(max(sh.keys()) + 1)}
    for L, mls in sh.items():
        if not L or (L == 1 and orderedp):
            conv[L] = np.array(cart[L])
            continue
        enum = [Counter(c) for c in cwr('xyz', L)]
        lmn = [[en[i] for i in 'xyz'] for en in enum]
        cdxs = [reduce(mul, i) for i in cwr((x, y, z), L)]
        for ml, sym in mls.items():
            mli = ml + L
            coefs = sym.expand().as_coefficients_dict()
            for crt, coef in coefs.items():
                if isinstance(crt, Integer): continue
                idx = cdxs.index(crt)
                l, m, n = lmn[idx]
                dif = max(l,m,n) - min(l,m,n)
                # Hack job ??
                if dif == L:
                    fc = 1
                elif L == 3 and not dif:
                    fc = np.sqrt(15)
                elif L == 4 and dif == 2:
                    fc = np.sqrt(105) / 3
                elif L == 4 and dif == 1:
                    fc = np.sqrt(35)
                else:
                    fc = np.sqrt(2 * L - 1)
                conv[L][idx, mli] = 1 / fc * coefs[cdxs[idx]]
    return conv


class Basis(object):

    def evaluate_diff(self, xs, ys, zs, cart='x', order=1):
        flds = np.empty((self.nbas, len(xs)), dtype=np.float64)
        for i, f in enumerate(self.functions):
            flds[i] = evaluate(str(f.diff(cart=cart, order=order)))
        return flds

    def evaluate(self, xs, ys, zs):
        flds = np.empty((self.nbas, len(xs)), dtype=np.float64)
        for i, f in enumerate(self.functions):
            flds[i] = evaluate(str(f))
        return flds

    def __repr__(self):
        return 'Basis({})'.format(self.nbas)

    def __len__(self):
        return len(self.functions)

    def __getitem__(self, key):
        return self.functions.__getitem__(key)

    def __init__(self, functions):
        self.functions = functions
        self.nbas = len(self.functions)


class _Function(ABC):

    _x = property(lambda self: var("_x"))
    _y = property(lambda self: var("_y"))
    _z = property(lambda self: var("_z"))
    _r = property(lambda self: (_x**2 + _y**2 + _z**2)**0.5)
    _r2 = property(lambda self: _x**2 + _y**2 + _z**2)

    @classmethod
    def diff(cls, cart='x', order=1, expr=None):
        """Compute the nth order derivative with respect to cart.

        Args
            cart (str): 'x', 'y', or 'z'
            order (int): order of differentiation

        Returns
            expr (symbolic): The symbolic derivative
        """
        if not isinstance(order, int) or order < 0:
            raise Exception("order must be non-negative int")
        expr = self._expr
        for i in range(order):
            expr = expr.diff('_'+cart+'s')
        return cls(expr)

    def evaluate(self, xs, ys, zs):
        return evaluate(self._expr.subs({'_x': 'xs',
                                         '_y': 'ys',
                                         '_z': 'zs'}))

    def __init__(self, expr):
        self._expr = expr


class BasisFunction(ABC):
    """Abstract base class for simple container of basis set information.

    Attributes
        x, y, z (symbolic): symbolic cartesian values
        r, r2 (symbolic): symbolic radial values (in terms of x,y,z)
        center (int): arbitrary atom center index
        xa, ya, za (float): atomic positions
        expnt (symbolic): exponential radial dependence
        alphas (np.array): primitive exponents
        ds (np.array): primitive contraction coefficients
        Ns (np.array): primitive normalization constants
        cs (np.array): overall contraction coefficients
        ps (int): length of the above arrays
    """
    # Class level properties as they can be shared between instances
    x = property(lambda self: x)
    y = property(lambda self: y)
    z = property(lambda self: z)
    r = property(lambda self: (x**2 + y**2 + z**2)**0.5)
    r2 = property(lambda self: x**2 + y**2 + z**2)

    # Abstract methods must be implemented in all sub-classes
    @abstractmethod
    def _angular(self): pass
    @abstractmethod
    def _prim_norm(self): pass
    @abstractmethod
    def _cont_norm(self): pass

    # Symbolic derivative of the basis function
    def diff(self, cart='x', order=1, expr=None):
        """Compute the nth order derivative with respect to cart.

        Args
            cart (str): 'x', 'y', or 'z'
            order (int): order of differentiation

        Returns
            expr (symbolic): The symbolic derivative
        """
        if not isinstance(order, int) or order < 0:
            raise Exception("order must be non-negative int")
        expr = self._angular() * self._radial()
        for i in range(order):
            expr = expr.diff(cart+'s')
        return expr

    # Radial dependence should be the same regardless of angular terms
    def _radial(self):
        return sum((c * exp(-a * self.expnt)
                    for c, a in zip(self.cs, self.alphas)))

    # Substitute atomic position into a symbolic expression
    def _subs(self, expr):
        return expr.subs({'x': 'xs-{}'.format(self.xa),
                          'y': 'ys-{}'.format(self.ya),
                          'z': 'zs-{}'.format(self.za)})

    def __str__(self):
        return str(self._angular() * self._radial())

    def __init__(self, center, xa, ya, za, ds, alphas,
                 gaussian=True):
        self.center = center
        self.xa = xa
        self.ya = ya
        self.za = za
        ex = self.r2 if gaussian else self.r
        self.expnt = self._subs(ex)
        try:
            self.ds = ds.values
            self.alphas = alphas.values
        except AttributeError:
            self.ds = ds
            self.alphas = alphas
        self.ps = len(self.ds)
        self.Ns = self._prim_norm()
        self.cs = self._cont_norm()


class CartesianBasisFunction(BasisFunction):
    """A basis function with cartesian angular dependence.
    (See also :class:`~exatomic.algorithms.basis.BasisFunction`.)

    Attributes:
        l (int): cartesian power in x
        m (int): cartesian power in y
        n (int): cartesian power in z
        L (int): sum of l, m, and n
    """

    def _angular(self):
        addtl = 1
        if self.rpre: addtl *= self.rpre
        if self.rpow: addtl *= self.r ** rpow
        return self._subs(addtl * self.x ** self.l *
                                  self.y ** self.m *
                                  self.z ** self.n)

    def _prim_norm(self):
        if self.gaussian:
            return _prim_cart_norm(self.alphas, self.l, self.m, self.n)
        return _prim_sto_norm(self.alphas, self.rpow)

    def _cont_norm(self):
        return self.Ns * self.ds * _cont_norm(self.ds, self.alphas, self.Ns,
                                              self.l, self.m, self.n)

    def __repr__(self):
        return (f'CBF(x={self.xa:.2f},y={self.ya:.2f},z={self.za:.2f},'
                f'l={self.l},m={self.m},n={self.n},p={self.ps})')

    def __init__(self, center, xa, ya, za, ds, alphas, l, m, n,
                 gaussian=True, rpow=0, rpre=0):
        self.l = l
        self.m = m
        self.n = n
        self.L = l + m + n
        self.rpow = rpow
        self.rpre = rpre
        self.gaussian = gaussian
        super(CartesianBasisFunction, self).__init__(
            center, xa, ya, za, ds, alphas, gaussian=gaussian)


class SphericalBasisFunction(BasisFunction):
    """A basis function with spherical angular dependence.
    (See also :class:`~exatomic.algorithms.basis.BasisFunction`.)

    Attributes:
        L (int): orbital angular momentum
        ml (int): magnetic quantum number
    """

    sh = property(lambda self: new_solid_harmonics(6))

    def _angular(self):
        return self._subs(self.sh[self.L][self.ml])

    def _prim_norm(self):
        return _prim_sphr_norm(self.alphas, self.L)

    def _cont_norm(self):
        return self.Ns * self.ds

    def __repr__(self):
        return (f'SBF(x={self.xa:.2f},y={self.ya:.2f},z={self.za:.2f},'
                f'L={self.L},ml={self.ml},p={self.ps})')

    def __init__(self, center, xa, ya, za, ds, alphas, L, ml,
                 gaussian=True, l=0, m=0, n=0):
        self.L = L
        self.ml = ml
        self.l = l
        self.m = m
        self.n = n
        super(SphericalBasisFunction, self).__init__(
            center, xa, ya, za, ds, alphas, gaussian=gaussian)


def gen_bfns(uni, frame=0, forcecart=False):
    bso = uni.basis_set_order.groupby(['frame', 'center'])
    sets = uni.basis_set.groupby(['frame', 'set'])
    atom = uni.atom.groupby('frame').get_group(frame)
    atom = enumerate(zip(atom['set'], atom['x'],
                         atom['y'], atom['z']))
    bfns = []
    # for i, (seht, x, y, z) in atom:
    #     ordr = bso.get_group((frame, i))
    #     bas = sets.get_group((frame, seht)).groupby(('L', 'shell'))
    #     for L, ml, shl, l, m, n in zip(ordr['L'], ordr['ml'],
    #                                    ordr['shell'], ordr['l'],
    #                                    ordr['m'], ordr['n']):
    #         s = bas.get_group((L, shl))
    #         if L < 2:
    #             bfns.append(CartesianBasisFunction(
    #                 i, x, y, z, s['d'], s['alpha'], L, l, m, n))
    #         else:
    #             bfns.append(SphericalBasisFunction(
    #                 i, x, y, z, s['d'], s['alpha'], L, ml))
    # return Basis(bfns)
    if forcecart:
        for i, (seht, x, y, z) in atom:
            bas = sets.get_group((frame, seht)).groupby('L')
            for L, grp in bas:
                for l, m, n in new_enum_cartesian[L]:
                    for sh, sub in grp.groupby('shell'):
                        bfns.append(CartesianBasisFunction(
                            i, x, y, z, sub['d'], sub['alpha'], l, m, n))
    elif uni.basis_set.spherical:
        for i, (seht, x, y, z) in atom:
            ordr = bso.get_group((frame, i))
            bas = sets.get_group((frame, seht)).groupby(['L', 'shell'])
            for L, ml, shl in zip(ordr['L'], ordr['ml'],
                                  ordr['shell']):
                s = bas.get_group((L, shl))
                bfns.append(SphericalBasisFunction(
                    i, x, y, z, s['d'], s['alpha'], L, ml))
    elif all(i in uni.basis_set_order.columns for i in ('l', 'm', 'n')):
        for i, (seht, x, y, z) in atom:
            ordr = bso.get_group((frame, i))
            bas = sets.get_group((frame, seht)).groupby('L')
            for L, l, m, n in zip(ordr['L'], ordr['l'],
                                  ordr['m'], ordr['n']):
                s = bas.get_group((L, shl))
                bfns.append(CartesianBasisFunction(
                    i, x, y, z, s['d'], s['alpha'], l, m, n))
    else:
        print('basis not obtained')
    return Basis(bfns)


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

@vectorize(['int64(int64)'])
def _vec_fac(n): return fac(n)
@vectorize(['int64(int64)'])
def _vec_fac2(n): return fac2(n)
@vectorize(['int64(int64)'])
def _vec_dfac21(n): return dfac21(n)

@jit(nopython=True, cache=True)
def normalize(alpha, L):
    prefac = (2 / np.pi) ** (0.75)
    numer = 2 ** L * alpha ** ((L + 1.5) / 2)
    denom = dfac21(L) ** 0.5
    return prefac * numer / denom
@jit(nopython=True, cache=True)
def _prim_sphr_norm(alphas, L):
    Ns = np.empty(len(alphas), dtype=np.float64)
    for i, a in enumerate(alphas):
        Ns[i] = normalize(a, L)
    return Ns
@vectorize('float64(float64,int64)')
def _vec_sphr_norm(alpha, L):
    return normalize(alpha, L)

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
    #tot = 1 / np.sqrt(tot)
    return 1 / np.sqrt(tot)
    #return Ns * ds * tot


@jit(nopython=True, cache=True)
def _ovl_indices(vals):
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
