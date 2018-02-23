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
import pandas as pd
from numexpr import evaluate

from symengine import var, exp, cos, sin, Add, Mul, Integer
from symengine.lib.symengine_wrapper import factorial as _factorial

from .overlap import _primitive_overlap, _primitive_kinetic
from .numerical import (fac, fac2, dfac21, _gen_prims, Shell, _gen_prims,
                        _iter_atoms_shells_cart, _iter_atoms_shells_sphr,
                        _tri_indices, _square, _triangle)

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
    cnts = [Counter(c) for c in cwr('xyz', i)]
    enum_cartesian[i] = np.array([[0, 0, 0]]) if not cnts \
                        else np.array([[c[x] for x in 'xyz'] for c in cnts])
lmap.update([('px', 1), ('py', 1), ('pz', 1)])
gaussian_cartesian = enum_cartesian.copy()
gaussian_cartesian[2] = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2],
                                  [1, 1, 0], [1, 0, 1], [0, 1, 1]])


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
    def _top_sh(lp, sp, sm):
        kr = int(not lp)
        return ((2 ** kr * (2 * lp + 1) / (2 * lp + 2)) ** 0.5 *
                (_x * sp - (1 - kr) * _y * sm))
    def _mid_sh(lp, m, sm, smm):
        return (((2 * lp + 1) * _z * sm - ((lp + m) * (lp - m)) ** 0.5 *
                (_x*_x + _y*_y + _z*_z) * smm) /
                (((lp + m + 1) * (lp - m + 1)) ** 0.5))
    def _bot_sh(lp, sp, sm):
        kr = int(not lp)
        return ((2 ** kr * (2 * lp + 1) / (2 * lp + 2)) ** 0.5 *
                (_y * sp + (1 - kr) * _x * sm))
    sh = OrderedDict([(l, OrderedDict([])) for l in range(lmax + 1)])
    sh[0][0] = Integer(1)
    for l in range(1, lmax + 1):
        lp = l - 1
        mls = list(range(-l, l + 1))
        sh[l][mls[0]] = _bot_sh(lp, sh[lp][lp], sh[lp][-lp])
        for ml in mls[1:-1]:
            try: rec = sh[lp - 1][ml]
            except KeyError: rec = sh[lp][ml]
            sh[l][ml] = _mid_sh(lp, ml, sh[lp][ml], rec)
        sh[l][mls[-1]] = _top_sh(lp, sh[lp][lp], sh[lp][-lp])
    return sh


def car2sph(sh, cart, orderedp=True):
    conv = OrderedDict([(L, np.zeros((cart_lml_count[L],
                                      spher_lml_count[L])))
                        for L in range(max(sh.keys()) + 1)])
    for L, mls in sh.items():
        if not L or (L == 1 and orderedp):
            conv[L] = np.array(cart[L])
            continue
        enum = [Counter(c) for c in cwr('xyz', L)]
        lmn = [[en[i] for i in 'xyz'] for en in enum]
        cdxs = [reduce(mul, i) for i in cwr((_x, _y, _z), L)]
        for ml, sym in mls.items():
            mli = ml + L
            coefs = sym.expand().as_coefficients_dict()
            for crt, coef in coefs.items():
                if isinstance(crt, Integer): continue
                idx = cdxs.index(crt)
                l, m, n = lmn[idx]
                fc = 1 / np.sqrt(dfac21(L) / (dfac21(l) * dfac21(m) * dfac21(n)))
                conv[L][idx, mli] = fc * coefs[cdxs[idx]]
    return conv


_sh = solid_harmonics(6)
_cartouche = car2sph(_sh, enum_cartesian)


class Symbolic(object):

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
        for i in range(order):
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

    def _atoms_shells(self, frame=0):
        atoms = self._uni.atom[self._uni.atom.frame == frame]
        shells = self._uni.basis_set.shells()
        return (atoms['set'].astype(np.int64).values,
                atoms[['x', 'y', 'z']].values,
                shells['set'].values,
                shells[0].values)

    def _primitives(self):
        atoms, xyzs, sets, shells = self._atoms_shells()
        return _gen_prims(atoms, xyzs, sets,
                          self._uni.basis_dims['npc'],
                          self._uni.basis_dims['ncc'],
                          *shells)

    def _cart_to_sphr(self):
        c2s = np.zeros((self._uni.basis_dims['ncc'],
                        self._uni.basis_dims['ncs']), dtype=np.float64)
        sets = self._uni.atom.set
        shls = self._uni.basis_set.functions_by_shell()
        cx, sx = 0, 0
        for seht in sets:
            shl = shls[seht]
            for L, n in shl.items():
                # The following depends on a code's basis function
                # ordering and is not correct in the general case.
                # This works for Molcas basis function ordering.
                exp = np.kron(_cartouche[L], np.eye(n)) if L else np.eye(n)
                ci, si = exp.shape
                c2s[cx : cx + ci, sx : sx + si] = exp
                cx += ci
                sx += si
        return c2s

    def _radial(self, x, y, z, alphas, cs, rs=None, pre=None):
        if not self._uni.basis_set.gaussian:
            return Symbolic(
                sum((pre * self._expnt ** r * c * exp(-a * self._expnt)
                    for c, a, r in zip(cs, alphas, rs))
                    ).subs({_x: _x - x, _y: _y - y, _z: _z - z}))
        return Symbolic(
            sum((c * exp(-a * self._expnt)
                for c, a in zip(cs, alphas))
                ).subs({_x: _x - x, _y: _y - y, _z: _z - z}))

    def _angular(self, x, y, z, *ang):
        if len(ang) == 3:
            sym = _x ** ang[0] * _y ** ang[1] * _z ** ang[2]
        elif ang[0] == 1 and self._cartp:
            mp = {-1: 1, 0: -1, 1: 0}
            sym = _sh[ang[0]][mp[ang[1]]]
        else:
            sym = _sh[ang[0]][ang[1]]
        return Symbolic(sym.subs({_x: _x - x, _y: _y - y, _z: _z - z}))

    def _prep_eval(self, d):
        flds = np.empty((len(self), d), dtype=np.float64)
        atoms, xyzs, sets, shells = self._atoms_shells()
        itr = self._itr(atoms, xyzs, sets, *shells)
        return flds, itr, 0

    def _evaluate_sto(self, xs, ys, zs):
        flds, itr, i = self._prep_eval(len(xs))
        pres = self._uni.basis_set_order.prefac.values
        for center, x, y, z, shell, ii, jj, *ang in itr:
            norm = self._nrm(shell)
            a = self._angular(x, y, z, *ang).evaluate(xs, ys, zs)
            for j in range(jj):
                pre = 1. if not pres[i] else pres[i]
                r = self._radial(x, y, z, shell.alphas, norm[:, j],
                                 rs=shell.rs, pre=pre)
                flds[i] = r.evaluate(xs, ys, zs, arr=a)
                i += 1
        return flds

    def _evaluate_cart(self, xs, ys, zs):
        flds, itr, i = self._prep_eval(len(xs))
        for center, x, y, z, shell, ii, jj, *ang in itr:
            norm = self._nrm(shell, *ang)
            a = self._angular(x, y, z, *ang).evaluate(xs, ys, zs)
            for j in range(jj):
                r = self._radial(x, y, z, shell.alphas, norm[:, j])
                flds[i] = r.evaluate(xs, ys, zs, arr=a)
                i += 1
        return flds

    def _evaluate_sphr(self, xs, ys, zs):
        flds, itr, i = self._prep_eval(len(xs))
        for center, x, y, z, shell, ii, jj, *ang in itr:
            norm = self._nrm(shell)
            a = self._angular(x, y, z, *ang).evaluate(xs, ys, zs)
            for j in range(jj):
                r = self._radial(x, y, z, shell.alphas, norm[:, j])
                flds[i] = r.evaluate(xs, ys, zs, arr=a)
                i += 1
        return flds

    def _evaluate_diff_sphr(self, xs, ys, zs, cart):
        flds, itr, i = self._prep_eval(len(xs))
        for center, x, y, z, shell, ii, jj, *ang in itr:
            norm = self._nrm(shell)
            a = self._angular(x, y, z, *ang)
            da = a.diff(cart=cart).evaluate(xs, ys, zs)
            a = a.evaluate(xs, ys, zs)
            for j in range(jj):
                r = self._radial(x, y, z, shell.alphas, norm[:, j])
                dr = r.diff(cart=cart).evaluate(xs, ys, zs)
                r = r.evaluate(xs, ys, zs)
                flds[i] = da * r + a * dr
                #flds[i] = evaluate('da * r + a * dr')
                i += 1
        return flds

    def _evaluate_diff_cart(self, xs, ys, zs, cart):
        flds, itr, i = self._prep_eval(len(xs))
        for center, x, y, z, shell, ii, jj, *ang in itr:
            norm = self._nrm(shell, *ang)
            a = self._angular(x, y, z, *ang)
            da = a.diff(cart=cart).evaluate(xs, ys, zs)
            a = a.evaluate(xs, ys, zs)
            for j in range(jj):
                r = self._radial(x, y, z, shell.alphas, norm[:, j])
                dr = r.diff(cart=cart).evaluate(xs, ys, zs)
                r = r.evaluate(xs, ys, zs)
                flds[i] = evaluate('da * r + a * dr')
                i += 1
        return flds

    def integrals(self):
        from exatomic.core.basis import Overlap
        if self._uni.meta['program'] != 'molcas':
            raise NotImplementedError('Must test for codes != molcas')
        p2c, *prims = self._primitives()
        povl = _square(_primitive_overlap(*prims))
        covl = np.dot(p2c.T, np.dot(povl, p2c))
        # pkin = _square(_primitive_kinetic(*prims))
        # ckin = np.dot(p2c.T, np.dot(pkin, p2c))
        if self._uni.basis_set.spherical:
            c2s = self._cart_to_sphr()
            covl = np.dot(c2s.T, np.dot(covl, c2s))
            # ckin = np.dot(c2s.T, np.dot(ckin, c2s))
        covl = _triangle(covl)
        # ckin = _triangle(ckin)
        chi0, chi1 = _tri_indices(covl)
        return Overlap.from_dict({'chi0': chi0, 'chi1': chi1,
                                  'frame': 0, 'coef': covl})

    def evaluate(self, xs, ys, zs):
        if not self._uni.basis_set.gaussian:
            return self._evaluate_sto(xs, ys, zs)
        if not self._uni.basis_set.spherical:
            return self._evaluate_cart(xs, ys, zs)
        return self._evaluate_sphr(xs, ys, zs)

    def evaluate_diff(self, xs, ys, zs, cart='x'):
        if not self._uni.basis_set.gaussian:
            raise NotImplementedError(
                "Must test symengine differentiation for STOs.")
        if not self._uni.basis_set.spherical:
            return self._evaluate_diff_cart(xs, ys, zs, cart=cart)
        return self._evaluate_diff_sphr(xs, ys, zs, cart=cart)

    def __len__(self):
        if self._uni.basis_set.spherical:
            return self._uni.basis_dims['ncs']
        return self._uni.basis_dims['ncc']

    def __repr__(self):
        if self._uni.basis_set.spherical:
            return 'Basis({},spherical)'.format(len(self))
        return 'Basis({},cartesian)'.format(len(self))

    def __init__(self, uni, frame=0, cartp=True):
        self._uni = uni
        self._cartp = cartp
        self._expnt = _r ** 2
        self._itr = _iter_atoms_shells_sphr
        self._nrm = Shell.sphr_norm_contract
        if not uni.basis_set.spherical:
            self._itr = _iter_atoms_shells_cart
            self._nrm = Shell.cart_norm_contract
        if not uni.basis_set.gaussian:
            self._expnt = _r
            self._nrm = Shell.sto_norm_contract
