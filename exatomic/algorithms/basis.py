# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
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
from operator import mul
from functools import reduce
from collections import OrderedDict, Counter
from itertools import combinations_with_replacement as cwr
import numpy as np
import pandas as pd
from numexpr import evaluate
try:
    from symengine import var, exp, cos, sin, Mul, Integer
except ImportError:
    from sympy import symbols as var
    from sympy import exp, cos, sin, Mul, Integer
from exatomic.algorithms.overlap import _cartesian_shell_pairs, _iter_atom_shells
from exatomic.algorithms.numerical import fac, _tri_indices, _triangle


_x, _y, _z = var("_x _y _z")
_r = (_x ** 2 + _y ** 2 + _z ** 2) ** 0.5

lorder = ['s', 'p', 'd', 'f', 'g',
          'h', 'i', 'k', 'l', 'm']
lmap = OrderedDict()
rlmap = OrderedDict()
spher_ml_count = OrderedDict()
cart_ml_count = OrderedDict()
spher_lml_count = OrderedDict()
cart_lml_count = OrderedDict()
enum_cartesian = OrderedDict()
for i, L in enumerate(lorder):
    lmap[L] = i
    rlmap[i] = L
    spher_ml_count[L] = 2 * i + 1
    cart_ml_count[L] = (i + 1) * (i + 2) // 2
    spher_lml_count[i] = spher_ml_count[L]
    cart_lml_count[i] = cart_ml_count[L]
    enum_cartesian[i] = []
    cnts = [Counter(c) for c in cwr('xyz', i)]
    enum_cartesian[i] = np.array([[0, 0, 0]]) if not cnts \
                        else np.array([[c[x] for x in 'xyz'] for c in cnts])
lmap.update([('px', 1), ('py', 1), ('pz', 1)])
gaussian_cartesian = enum_cartesian.copy()
gaussian_cartesian[2] = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2],
                                  [1, 1, 0], [1, 0, 1], [0, 1, 1]])


def _hermite_gaussians(lmax):
    order = 2 * lmax + 1
    hgs = OrderedDict()
    der = exp(-_x ** 2)
    for t in range(order):
        if t: der = der.diff(_x)
        hgs[t] = (-1) ** t * der
    return hgs


def gen_enum_cartesian(lmax):
    return OrderedDict([(L, np.array([[l, L - l - n, n]
                        for l in range(L, -1, -1)
                        for n in range(L + 1 - l)]))
                        for L in range(lmax + 1)])


def spherical_harmonics(lmax):
    phase = {m: (-1) ** m for m in range(lmax + 1)}
    facts = {n: fac(n) for n in range(2 * lmax + 1)}
    sh = OrderedDict()
    for L in range(lmax + 1):
        sh[L] = OrderedDict()
        rac = ((2 * L + 1) / (4 * np.pi)) ** 0.5
        der = (_x ** 2 - 1) ** L
        den = 2 ** L * facts[L]
        for _ in range(L):
            der = der.diff(_x)
        for m in range(L + 1):
            pol = (1 - _x ** 2) ** (m/2)
            if m: der = der.diff(_x)
            leg = phase[m] / den * (pol * der).subs({_x: _z / _r})
            if not m:
                sh[L][m] = rac * leg
                continue
            N = 2 ** 0.5 * phase[m]
            facs = facts[L - m] / facts[L + m]
            norm = facs ** 0.5
            phi = (m * _x).subs({_x: 'arctan2(_y, _x)'})
            fun = cos(phi)
            sh[L][m] = N * rac * norm * leg * fun
            fun = sin(phi)
            sh[L][-m] = N * rac * norm * leg * fun
    return sh


def solid_harmonics(lmax):
    """Symbolic, recursive solid harmonics for the angular component
    of a wave function."""
    def _top_sh(lp, kr, sp, sm):
        return ((2 ** kr * (2 * lp + 1) / (2 * lp + 2)) ** 0.5 *
                (_x * sp - (1 - kr) * _y * sm))
    def _mid_sh(lp, m, sm, smm):
        return (((2 * lp + 1) * _z * sm - ((lp + m) * (lp - m)) ** 0.5 *
                (_x*_x + _y*_y + _z*_z) * smm) /
                (((lp + m + 1) * (lp - m + 1)) ** 0.5))
    def _bot_sh(lp, kr, sp, sm):
        return ((2 ** kr * (2 * lp + 1) / (2 * lp + 2)) ** 0.5 *
                (_y * sp + (1 - kr) * _x * sm))
    sh = OrderedDict([(l, OrderedDict([])) for l in range(lmax + 1)])
    sh[0][0] = Integer(1)
    for l in range(1, lmax + 1):
        lp = l - 1
        kr = int(not lp)
        mls = list(range(-l, l + 1))
        sh[l][mls[0]] = _bot_sh(lp, kr, sh[lp][lp], sh[lp][-lp])
        for ml in mls[1:-1]:
            try: rec = sh[lp - 1][ml]
            except KeyError: rec = sh[lp][ml]
            sh[l][ml] = _mid_sh(lp, ml, sh[lp][ml], rec)
        sh[l][mls[-1]] = _top_sh(lp, kr, sh[lp][lp], sh[lp][-lp])
    return sh


def car2sph(sh, cart, orderedp=True):
    c2s = OrderedDict([(L, np.zeros(((L + 1) * (L + 2) // 2, 2 * L + 1)))
                      for L in range(max(sh.keys()) + 1)])
    for L, mls in sh.items():
        if not L or (L == 1 and orderedp):
            c2s[L] = np.array(cart[L])
            continue
        cdxs = [reduce(mul, xyz) for xyz in cwr((_x, _y, _z), L)]
        for ml, sym in mls.items():
            mli = ml + L
            coefs = sym.expand().as_coefficients_dict()
            for crt, coef in coefs.items():
                if isinstance(crt, Integer): continue
                idx = cdxs.index(crt)
                c2s[L][idx, mli] = coefs[cdxs[idx]]
    return c2s


class Symbolic(object):
    @property
    def _constructor(self):
        return Symbolic

    def diff(self, cart='x', order=1):
        """Compute the nth order derivative symbolically with respect to cart.

        Args
            cart (str): 'x', 'y', or 'z'
            order (int): order of differentiation

        Returns
            expr (symbolic): The symbolic derivative
        """
        assert cart in ['x', 'y', 'z']
        assert isinstance(order, int) and order > 0
        expr = self._expr
        for _ in range(order):
            expr = expr.diff('_'+cart)
        return Symbolic(expr)

    def evaluate(self, xs, ys, zs, arr=None, alpha=None):
        subs = {_x: 'xs', _y: 'ys', _z: 'zs'}
        if arr is not None:
            return evaluate('arr * ({})'.format(str(self._expr.subs(subs))))
        if alpha is None:
            return evaluate(str(self._expr.subs(subs)))
        expnt = exp(-alpha * _r ** 2)
        return evaluate(str((self._expr * expnt).subs(subs)))

    def __repr__(self):
        return str(self._expr)

    def __init__(self, expr):
        self._expr = expr


class Basis(object):
    """Composition wrapper class that leverages symbolic expressions using
    symengine and numexpr, using values extracted from the numerical Shell
    jitclasses, to evaluate basis functions on a numerical grid.

    Args
        uni (exatomic.core.universe.Universe): a universe with basis set
        frame (int): frame corresponding to basis set (default=0)
        cartp (bool): forces p function ordering as (x, y, z) not (-1, 0, 1)
    """
    # Unscaled solid harmonics
    _sh = solid_harmonics(6)

    @property
    def _constructor(self):
        return Basis

    def integrals(self):
        """Compute the overlap matrix using primitive cartesian integrals."""
        from exatomic.core.basis import Overlap
        ovl = _cartesian_shell_pairs(len(self), self._ptrs.astype(np.int64),
                                     self._xyzs, *self._shells)
        ovl = _triangle(ovl)
        chi0, chi1 = _tri_indices(ovl)
        return Overlap.from_dict({'chi0': chi0, 'chi1': chi1,
                                  'frame': 0, 'coef': ovl})

    def enum_shell(self, shl):
        """Return a generator over angular momentum degrees of freedom."""
        if shl.spherical:
            return shl.enum_spherical()
        return shl.enum_cartesian()

    def evaluate(self, xs, ys, zs):
        """Evaluate basis functions on a numerical grid."""
        if not self._gaussian:
            return self._evaluate_sto(xs, ys, zs)
        return self._evaluate_gau(xs, ys, zs)

    def evaluate_diff(self, xs, ys, zs, cart='x'):
        """Evaluate basis function derivatives on a numerical grid."""
        if not self._gaussian:
            return self._evaluate_diff_sto(xs, ys, zs, cart)
        return self._evaluate_diff_gau(xs, ys, zs, cart)

    def _radial(self, x, y, z, alphas, cs, rs=None, pre=None):
        """Generates the symbolic radial portion of a basis function."""
        if not self._gaussian:
            return Symbolic(
                sum((pre * self._expnt ** r * c * exp(-a * self._expnt)
                    for c, a, r in zip(cs, alphas, rs))
                    ).subs({_x: _x - x, _y: _y - y, _z: _z - z}))
        return Symbolic(
            sum((c * exp(-a * self._expnt)
                for c, a in zip(cs, alphas))
                ).subs({_x: _x - x, _y: _y - y, _z: _z - z}))

    def _angular(self, shl, x, y, z, *ang):
        """Generates the symbolic angular portion of a basis function."""
        if len(ang) == 3:
            sym = _x ** ang[0] * _y ** ang[1] * _z ** ang[2]
        else:
            # Many codes order p functions (x, y, z), not (-1, 0, 1)
            if ang[0] == 1 and self._cartp:
                mp = {-1: 1, 0: -1, 1: 0}
                sym = Basis._sh[ang[0]][mp[ang[1]]]
            else:
                sym = Basis._sh[ang[0]][ang[1]]
            # Scaled solid harmonics as in the overlap code
            if shl.spherical and self._program == 'molcas':
                sym /= (2 * np.pi ** 0.5)
        return Symbolic(sym.subs({_x: _x - x, _y: _y - y, _z: _z - z}))

    def _evaluate_sto(self, xs, ys, zs):
        """Evaluates a full STO basis set and returns a numpy array."""
        cnt, flds = 0, np.empty((len(self), len(xs)))
        for i, ax, ay, az, ishl in \
            _iter_atom_shells(self._ptrs, self._xyzs, *self._shells):
            norm = ishl.norm_contract()
            for mag in self.enum_shell(ishl):
                a = self._angular(ishl, ax, ay, az, *mag).evaluate(xs, ys, zs)
                for c in range(ishl.ncont):
                    pre = 1 if np.isclose(self._pre[cnt], 0) else self._pre[cnt]
                    r = self._radial(ax, ay, az, ishl.alphas, norm[:, c],
                                     rs=ishl.rs, pre=pre)
                    flds[cnt] = r.evaluate(xs, ys, zs, arr=a)
                    cnt += 1
        return flds

    def _evaluate_diff_sto(self, xs, ys, zs, cart):
        raise NotImplementedError("Verify symbolic differentiation of STOs.")

    def _evaluate_gau(self, xs, ys, zs):
        """Evaluates a full Gaussian basis set and returns a numpy array."""
        cnt, flds = 0, np.empty((len(self), len(xs)))
        for _, ax, ay, az, ishl in _iter_atom_shells(self._ptrs, self._xyzs, *self._shells):
            norm = ishl.norm_contract()
            for mag in self.enum_shell(ishl):
                a = self._angular(ishl, ax, ay, az, *mag).evaluate(xs, ys, zs)
                for c in range(ishl.ncont):
                    r = self._radial(ax, ay, az, ishl.alphas, norm[:,c])
                    flds[cnt] = r.evaluate(xs, ys, zs, arr=a)
                    cnt += 1
        return flds

    def _evaluate_diff_gau(self, xs, ys, zs, cart):
        """Evaluates the derivatives of a full Gaussian basis
        set and returns a numpy array."""
        cnt, flds = 0, np.empty((len(self), len(xs)))
        for i, ax, ay, az, ishl in _iter_atom_shells(self._ptrs, self._xyzs, *self._shells):
            norm = ishl.norm_contract()
            for mag in self.enum_shell(ishl):
                a = self._angular(ishl, ax, ay, az, *mag)
                da = a.diff(cart=cart).evaluate(xs, ys, zs)
                a = a.evaluate(xs, ys, zs)
                for c in range(ishl.ncont):
                    r = self._radial(ax, ay, az, ishl.alphas, norm[:, c])
                    dr = r.diff(cart=cart).evaluate(xs, ys, zs)
                    r = r.evaluate(xs, ys, zs)
                    flds[cnt] = evaluate('da * r + a * dr')
                    cnt += 1
        return flds

    def __len__(self):
        return self._ncs if self._spherical else self._ncc

    def __repr__(self):
        chk = (i.spherical for i in self._shells)
        repr = 'Basis({},{{}})'.format(len(self)).format
        if all(chk): return repr('spherical')
        if not any(chk): return repr('cartesian')
        return repr('mixed')

    def __init__(self, uni, frame=0, cartp=True):
        self._program = uni.meta['program']
        ptrs, xyzs, shells = uni.enumerate_shells()
        self._ptrs = ptrs
        self._xyzs = xyzs
        self._shells = shells
        self._spherical = uni.basis_set.spherical
        self._gaussian = uni.basis_set.gaussian
        self._npc = uni.basis_dims['npc']
        self._ncc = uni.basis_dims['ncc']
        self._nps = uni.basis_dims['nps']
        self._ncs = uni.basis_dims['ncs']
        self._cartp = cartp
        self._expnt = _r ** 2
        if not uni.basis_set.gaussian:
            self._expnt = _r
            self._pre = uni.basis_set_order['prefac']
