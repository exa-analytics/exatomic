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
from collections import OrderedDict, Counter, defaultdict
from itertools import combinations_with_replacement as cwr
import numpy as np
import pandas as pd
from numexpr import evaluate
try:
    from symengine import var, exp, cos, sin, Mul, Integer, Float
except ImportError:
    from sympy import symbols as var
    from sympy import exp, cos, sin, Mul, Integer, Float
from exatomic.algorithms.overlap import _cartesian_shell_pairs, _iter_atom_shells
from exatomic.algorithms.numerical import fac, _tri_indices, _triangle, _enum_spherical


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
    """Symbolic hermite gaussians up to order lmax.

    Args:
        lmax (int): highest order angular momentum quantum number
    """
    order = 2 * lmax + 1
    hgs = OrderedDict()
    der = exp(-_x ** 2)
    for t in range(order):
        if t: der = der.diff(_x)
        hgs[t] = (-1) ** t * der
    return hgs


def gen_enum_cartesian(lmax):
    """Cartesian powers in the order expected for overlap
    computation up to order lmax.

    Args:
        lmax (int): highest order angular momentum quantum number
    """
    return OrderedDict([(L, np.array([[l, L - l - n, n]
                                      for l in range(L, -1, -1)
                                      for n in range(L + 1 - l)]))
                        for L in range(lmax + 1)])


def spherical_harmonics(lmax):
    """Symbolic real spherical harmonics up to order lmax.

    .. code-block:: python

        sh = spherical_harmonics(6)  # Inclusive, so computes up to l = 6
        sh[3][-3]                    # symbolic f-phi angular function

    Args:
        lmax (int): highest order angular momentum quantum number
    """
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


def solid_harmonics(lmax, scaled=False):
    """Symbolic real solid harmonics up to order lmax.

    .. code-block:: python

        sh = solid_harmonics(6)  # Inclusive, so computes up to l = 6
        sh[3][-3]                # symbolic f-phi angular function

    Args:
        lmax (int): highest order angular momentum quantum number
        scaled (bool): if scaled, includes factor of 1 / (2 * np.pi ** 0.5)
    """
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
    if scaled: sh[0][0] = Float(1 / (2 * np.pi ** 0.5))
    else: sh[0][0] = Integer(1)
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
        if scaled:
            for ml in mls:
                sh[l][ml] /= 2 * np.pi ** 0.5
    return sh


def car2sph(sh, cart, orderedp=True):
    """Cartesian to spherical transform matrices.

    .. code-block:: python

        sh = solid_harmonics(8)        # symbolic solid harmonics
        cart = gen_enum_cartesian(8)   # cartesian powers in a defined order
        c2s = car2sph(sh, cart)        # dictionary of {l: transform_matrix}

    Args:
        sh (OrderedDict): symbolic solid harmonics
        cart (OrderedDict): cartesian powers in a defined order
        orderedp (bool): order l=1 as ['x', 'y', 'z'], not [-1, 0, 1] (default True)

    Returns:
        c2s (OrderedDict): cartesian to spherical transform matrices
    """
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
            #for crt, coef in coefs.items():
            for crt, _ in coefs.items():
                if isinstance(crt, (Integer, Float)): continue
                idx = cdxs.index(crt)
                c2s[L][idx, mli] = coefs[cdxs[idx]]
    return c2s


class Symbolic(object):
    """
    A small wrapper around sympy or symengine symbolic expressions.
    Allows for evaluation of symbolic derivatives and numerical evaluation
    of expressions on a numerical grid.
    """


    def diff(self, cart='x', order=1):
        """Compute the nth order derivative symbolically with respect to cart.

        Args:
            cart (str): 'x', 'y', or 'z'
            order (int): order of differentiation

        Returns:
            expr (symbolic): The symbolic derivative
        """
        if cart not in ['x', 'y', 'z']:
            raise ValueError('cart must be in "xyz".')
        if not isinstance(order, int) or order < 0:
            raise ValueError('order must be non-negative integer.')
        expr = self._expr
        for _ in range(order):
            expr = expr.diff('_'+cart)
        return Symbolic(expr)


    def evaluate(self, xs, ys, zs, arr=None, alpha=None):
        """Evaluate symbolic expression on a numerical grid.

        Args:
            xs (np.ndarray): 1D-array of x values
            ys (np.ndarray): 1D-array of y values
            zs (np.ndarray): 1D-array of z values
            arr (np.ndarray): additional 1D-array to multiply expression by
            alpha (float): multiply expression by gaussian with exponent alpha

        Note:
            See :meth:`exatomic.algorithms.orbital_util.numerical_grid_from_field_params`
            for grid construction details.
        """
        subs = {_x: 'xs', _y: 'ys', _z: 'zs'}
        if arr is not None:
            return evaluate('arr * ({})'.format(str(self._expr.subs(subs))))
        if alpha is None:
            return evaluate(str(self._expr.subs(subs)))
        expnt = exp(-alpha * _r ** 2)
        return evaluate(str((self._expr * expnt).subs(subs)))

    def __mul__(self, other):
        return self.__class__(self._expr * other._expr)

    def __repr__(self):
        return str(self._expr)

    def __init__(self, expr):
        self._expr = expr


class BasisFunctions(object):
    """Composition wrapper class that leverages symbolic expressions using
    symengine and numexpr, using values extracted from the numerical Shell
    jitclasses, to evaluate basis functions on a numerical grid.

    Args:
        uni (:class:`exatomic.core.universe.Universe`): a universe with basis set
        frame (int): frame corresponding to basis set (default 0)
        cartp (bool): forces p function ordering as (x, y, z) not (-1, 0, 1)
    """


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
        """Return a generator over angular momentum degrees of freedom.

        Args:
            shl (:class:`exatomic.algorithms.numerical.Shell`): basis set shell
        """
        return shl.enum_spherical() if shl.spherical else shl.enum_cartesian()


    def evaluate(self, xs, ys, zs):
        """Evaluate basis functions on a numerical grid.

        Args:
            xs (np.ndarray): 1D-array of x values
            ys (np.ndarray): 1D-array of y values
            zs (np.ndarray): 1D-array of z values

        Note:
            See :meth:`exatomic.algorithms.orbital_util.numerical_grid_from_field_params`
            for grid construction details.
        """
        if self._meta['gaussian']:
            if self._meta['program'] in ['molcas']:
                func = self._evaluate_gau_mag
            elif self._meta['program'] in ['nwchem']:
                func = self._evaluate_gau_bso
        else: func = self._evaluate_sto
        return func(xs, ys, zs)

    def list(self):
        """Construct symbolic basis functions.

        Returns:
            bfns (list): symbolic basis functions
        """
        if self._meta['gaussian']:
            if self._meta['program'] in ['molcas']:
                func = self._evaluate_gau_mag
            elif self._meta['program'] in ['nwchem']:
                func = self._evaluate_gau_bso
        else:
            func = self._evaluate_sto
        return func(None, None, None, eval=False)


    def evaluate_diff(self, xs, ys, zs, cart='x'):
        """Evaluate basis function derivatives on a numerical grid.

        Args:
            xs (np.ndarray): 1D-array of x values
            ys (np.ndarray): 1D-array of y values
            zs (np.ndarray): 1D-array of z values
            cart (str): derivative with respect to cart (in ['x', 'y', 'z'])

        Note:
            See :meth:`exatomic.algorithms.orbital_util.numerical_grid_from_field_params`
            for grid construction details.
        """
        if self._meta['program'] in ['nwchem']:
            raise NotImplementedError("Code up _evaluate_diff_gau_bso.")
        if not self._meta['gaussian']:
            return self._evaluate_diff_sto(xs, ys, zs, cart)
        return self._evaluate_diff_gau(xs, ys, zs, cart)


    def _radial(self, x, y, z, alphas, cs, rs=None, pre=None):
        """Generates the symbolic radial portion of a basis function."""
        if pre is not None:
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
            l, m, n = ang
            sym = _x ** l * _y ** m * _z ** n
        else:
            L, ml = ang
            sym = self._sh[L][ml]
        return Symbolic(sym.subs({_x: _x - x, _y: _y - y, _z: _z - z}))


    def _evaluate_sto(self, xs, ys, zs, eval=True):
        """Evaluates a full STO basis set and returns a numpy array."""
        cnt = 0
        if eval: flds = np.empty((len(self), len(xs)))
        else: flds = [None for _ in range(len(self))]
        for _, ax, ay, az, ishl in _iter_atom_shells(self._ptrs, self._xyzs, *self._shells):
            norm = ishl.norm_contract()
            if self._meta['spherical']: ang = ishl.enum_spherical()
            else: ang = ishl.enum_cartesian()
            for mag in ang:
                a = self._angular(ishl, ax, ay, az, *mag)
                if eval: a = a.evaluate(xs, ys, zs)
                for c in range(ishl.ncont):
                    pre = 1 if self._meta['spherical'] else self._pre[cnt]
                    r = self._radial(ax, ay, az, ishl.alphas, norm[:, c],
                                     rs=ishl.rs, pre=pre)
                    if eval: flds[cnt] = r.evaluate(xs, ys, zs, arr=a)
                    else: flds[cnt] = a * r
                    cnt += 1
        return flds


    def _evaluate_diff_sto(self, xs, ys, zs, cart):
        raise NotImplementedError("Verify symbolic differentiation of STOs.")


    def _evaluate_gau_bso(self, xs, ys, zs, eval=True):
        """Evaluates a full Gaussian basis set and returns a numpy array."""
        cnt = 0
        if eval: flds = np.empty((len(self), len(xs)))
        else: flds = [None for _ in range(len(self))]
        cache = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        p = pd.DataFrame(self._ptrs, columns=('center', 'shelldx'))
        p['L'] = [self._shells[i].L for i in p['shelldx']]
        grps = p.groupby(['center', 'L'])
        for cen, L, ml in zip(self._bso['center'],
                              self._bso['L'], self._bso['ml']):
            ax, ay, az = self._xyzs[cen]
            ishl = self._shells[grps.get_group((cen, L))['shelldx'].values[0]]
            norm = ishl.norm_contract()
            a = self._angular(ishl, ax, ay, az, L, ml)
            r = self._radial(ax, ay, az, ishl.alphas, norm[:,cache[cen][L][ml]])
            if eval: flds[cnt] = r.evaluate(xs, ys, zs, arr=a.evaluate(xs, ys, zs))
            else: flds[cnt] = a * r
            cache[cen][L][ml] += 1
            cnt += 1
        return flds


    def _evaluate_gau_mag(self, xs, ys, zs, eval=True):
        """Evaluates a full Gaussian basis set and returns a numpy array."""
        cnt = 0
        if eval: flds = np.empty((len(self), len(xs)))
        else: flds = [None for _ in range(len(self))]
        for _, ax, ay, az, ishl in _iter_atom_shells(self._ptrs, self._xyzs, *self._shells):
            norm = ishl.norm_contract()
            for mag in self.enum_shell(ishl):
                a = self._angular(ishl, ax, ay, az, *mag)
                if eval: a = a.evaluate(xs, ys, zs)
                for c in range(ishl.ncont):
                    r = self._radial(ax, ay, az, ishl.alphas, norm[:,c])
                    if eval: flds[cnt] = r.evaluate(xs, ys, zs, arr=a)
                    else: flds[cnt] = a * r
                    cnt += 1
        return flds


    def _evaluate_diff_gau(self, xs, ys, zs, cart):
        """Evaluates the derivatives of a full Gaussian basis
        set and returns a numpy array."""
        cnt, flds = 0, np.empty((len(self), len(xs)))
        for _, ax, ay, az, ishl in _iter_atom_shells(self._ptrs, self._xyzs, *self._shells):
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
        return self._ncs if self._meta['spherical'] else self._ncc


    def __repr__(self):
        chk = (i.spherical for i in self._shells)
        _repr = 'BasisFunctions({},{{}})'.format(len(self)).format
        if all(chk): return _repr('spherical')
        if not any(chk): return _repr('cartesian')
        return _repr('mixed')


    def __init__(self, uni, frame=0, cartp=True):
        # Attach relevant uni attributes
        self._meta = uni.meta
        if self._meta['program'] in ['nwchem']:
            self._bso = uni.basis_set_order
        ptrs, xyzs, shells = uni.enumerate_shells()
        self._ptrs = ptrs
        self._xyzs = xyzs
        self._shells = shells
        self._ncc = uni.basis_dims['ncc']
        self._ncs = uni.basis_dims['ncs']
        # Scaled or unscaled solid harmonics
        lmax = uni.basis_set.lmax
        sh = solid_harmonics(lmax)
        scaled = self._meta['program'] in ['molcas']
        if scaled and lmax > 2:
            ssh = solid_harmonics(lmax, scaled=True)
            for L in range(2, lmax + 1):
                sh[L] = ssh[L]
        # Re-order p functions as 'x', 'y', 'z' rather than -1, 0, 1
        if cartp:
            ptmp = sh[1].copy()
            sh[1] = OrderedDict((ml, ptmp[ml]) for ml in (1, -1, 0))
        self._sh = sh
        # Exponential dependence
        self._expnt = _r ** 2
        if not self._meta['gaussian']:
            self._expnt = _r
            self._pre = uni.basis_set_order['prefac']
