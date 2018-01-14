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
import re
#import sympy
import numpy as np
#from sympy import Add, Mul
from collections import OrderedDict
from numba import jit, vectorize

import re
from operator import add
from numexpr import evaluate
from symengine import var, exp, Add, Mul, Integer

var("x y z")


lorder = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'k', 'l', 'm']
lmap = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6, 'k': 7, 'l': 8,
        'm': 9, 'px': 1, 'py': 1, 'pz': 1}
rlmap = {value: key for key, value in lmap.items() if len(key) == 1}
spher_ml_count = {'s': 1, 'p': 3, 'd': 5, 'f': 7, 'g': 9, 'h': 11, 'i': 13, 'k': 15,
                  'l': 17, 'm': 19}
spher_lml_count = {lorder.index(key): value for key, value in spher_ml_count.items()}
cart_ml_count = {'s': 1, 'p': 3, 'd': 6, 'f': 10, 'g': 15, 'h': 21, 'i': 28}
cart_lml_count = {lorder.index(key): value for key, value in cart_ml_count.items()}
enum_cartesian = {0: [[0, 0, 0]],
                  1: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                  2: [[2, 0, 0], [1, 1, 0], [1, 0, 1],
                      [0, 2, 0], [0, 1, 1], [0, 0, 2]],
                  3: [[3, 0, 0], [2, 1, 0], [2, 0, 1],
                      [1, 2, 0], [1, 1, 1], [1, 0, 2],
                      [0, 3, 0], [0, 2, 1], [0, 1, 2], [0, 0, 3]],
                  4: [[4, 0, 0], [3, 1, 0], [3, 0, 1], [2, 2, 0], [2, 1, 1],
                      [2, 0, 2], [1, 3, 0], [1, 2, 1], [1, 1, 2], [1, 0, 3],
                      [0, 4, 0], [0, 3, 1], [0, 2, 2], [0, 1, 3], [0, 0, 4]],
                  5: [[5, 0, 0], [4, 1, 0], [4, 0, 1], [3, 2, 0], [3, 1, 1],
                      [3, 0, 2], [2, 3, 0], [2, 2, 1], [2, 1, 2], [2, 0, 3],
                      [1, 4, 0], [1, 3, 1], [1, 2, 2], [1, 1, 3], [1, 0, 4],
                      [0, 5, 0], [0, 4, 1], [0, 3, 2], [0, 2, 3], [0, 1, 4],
                      [0, 0, 5]]}
gaussian_cartesian = enum_cartesian.copy()
gaussian_cartesian[2] = [[2, 0, 0], [0, 2, 0], [0, 0, 2],
                         [1, 1, 0], [1, 0, 1], [0, 1, 1]]

def solid_harmonics(l_max):

    def _top_sh(lcur, sp, sm):
        lpre = lcur - 1
        kr = 1 if lpre == 0 else 0
        return (np.sqrt(2 ** kr * (2 * lpre + 1) / (2 * lpre + 2)) *
                (x * sp - (1 - kr) * y * sm))

    def _mid_sh(lcur, m, sm, smm):
        lpre = lcur - 1
        return (((2 * lpre + 1) * z * sm - np.sqrt((lpre + m) *
                (lpre - m)) * (x*x + y*y + z*z) * smm) /
                (np.sqrt((lpre + m + 1) * (lpre - m + 1))))

    def _bot_sh(lcur, sp, sm):
        lpre = lcur - 1
        kr = 1 if lpre == 0 else 0
        return (np.sqrt(2 ** kr * (2 * lpre + 1) / (2 * lpre + 2)) *
                (y * sp + (1 - kr) * x * sm))

    sh = OrderedDict()
    sh[(0, 0)] = Integer(1)
    for l in range(1, l_max + 1):
        lpre = l - 1
        ml_all = list(range(-l, l + 1))
        sh[(l, ml_all[0])] = _bot_sh(l, sh[(lpre,lpre)], sh[(lpre,-(lpre))])
        for ml in ml_all[1:-1]:
            try:
                sh[(l, ml)] = _mid_sh(l, ml, sh[(lpre,ml)], sh[(lpre-1,ml)])
            except KeyError:
                sh[(l, ml)] = _mid_sh(l, ml, sh[(lpre,ml)], sh[(lpre,ml)])
        sh[(l, ml_all[-1])] = _top_sh(l, sh[(lpre,lpre)], sh[(lpre,-(lpre))])
    return sh


class PrimitiveFunction(object):

    cart = {'x': x, 'y': y, 'z': z}

    def gradient(self, cart='x', order=1, expr=None):
        if not isinstance(order, int) or order < 0:
            raise Exception("order must be non-negative int")
        expr = self.kernel
        for i in range(order):
            expr = expr.diff(self.cart[cart])
        return expr

    def __str__(self):
        return str(self.kernel)

    def __init__(self, xa, ya, za, pre, N, alpha,
                 gaussian=True):
        """Assumes Ns are actually Nds"""
        self.xA = xa
        self.yA = ya
        self.zA = za
        self.xa = x - xa
        self.ya = y - ya
        self.za = z - za
        self.r2 = self.xa ** 2 + self.ya ** 2 + self.za ** 2
        self.r = self.r2 ** (0.5)
        self.pre = pre.subs({'x': self.xa,
                             'y': self.ya,
                             'z': self.za})
        if gaussian: self.exp = N * exp(-alpha * self.r2)
        else:
            self.exp = N * exp(-alphas * self.r)
            print("ADF needs r-dependent pre-exponential")
        self.kernel = self.pre * self.exp





class BasisFunction(object):

    cart = {'x': x, 'y': y, 'z': z}

    def gradient(self, cart='x', order=1, expr=None):
        if not isinstance(order, int) or order < 0:
            raise Exception("order must be non-negative int")
        expr = self.kernel
        for i in range(order):
            expr = expr.diff(self.cart[cart])
        return expr

    def _add(self, Ns, alphas, expnt, expr=None):
        expr = Ns[0] * exp(-alphas[0] * expnt)
        for N, alpha in zip(Ns[1:], alphas[1:]):
            expr += N * exp(-alpha * expnt)
        return expr

    def __str__(self):
        return str(self.kernel)

    def __init__(self, xa, ya, za, pre, Ns, alphas,
                 gaussian=True):
        """Assumes Ns are actually Nds"""
        self.xA = xa
        self.yA = ya
        self.zA = za
        self.xa = x - xa
        self.ya = y - ya
        self.za = z - za
        self.r2 = self.xa ** 2 + self.ya ** 2 + self.za ** 2
        self.r = self.r2 ** (0.5)
        self.pre = pre.subs({'x': self.xa,
                             'y': self.ya,
                             'z': self.za})
        try:
            Ns = Ns.values
            alphas = alphas.values
        except AttributeError:
            pass
        self.prim = len(Ns)
        if gaussian:
            self.exp = self._add(Ns, alphas, self.r2)
        else:
            self.exp = self._add(Ns, alphas, self.r)
            print("ADF needs r-dependent pre-exponential")
        self.kernel = self.pre * self.exp


class CartesianBasisFunction(BasisFunction):

    def __repr__(self):
        return 'BsFn(x={:.2f},y={:.2f},z={:.2f},'\
               'prim={},l={},m={},n={})'.format(
                   self.xA, self.yA, self.zA,
                   self.prim, self.l, self.m, self.n)

    def __init__(self, xa, ya, za, Ns, alphas, l, m, n):
        pre = x ** l * y ** m * z ** n
        super(CartesianBasisFunction, self).__init__(
            xa, ya, za, pre, Ns, alphas)
        self.l = l
        self.m = m
        self.n = n


class SphericalBasisFunction(BasisFunction):

    sh = solid_harmonics(6)

    def __repr__(self):
        return 'BsFn(x={:.2f},y={:.2f},z={:.2f},'\
               'prim={},L={},ml={})'.format(
                   self.xA, self.yA, self.zA,
                   self.prim, self.L, self.ml)

    def __init__(self, xa, ya, za, Ns, alphas, L, ml):
        pre = self.sh[(L, ml)]
        super(SphericalBasisFunction, self).__init__(
            xa, ya, za, pre, Ns, alphas)
        self.L = L
        self.ml = ml


def clean_sh(sh):
    """Turns symbolic solid harmonic functions into string representations
    to be using in generating basis functions.

    Args
        sh (OrderedDict): Output from exatomic.algorithms.basis.solid_harmonics

    Returns
        clean (OrderedDict): cleaned strings
    """
    _replace = {'x': '{x}', 'y': '{y}', 'z': '{z}', ' - ': ' -'}
    _repatrn = re.compile('|'.join(_replace.keys()))
    clean = OrderedDict()
    for key, sym in sh.items():
        string = str(sym.expand()).replace(' + ', ' ')
        string = _repatrn.sub(lambda x: _replace[x.group(0)], string)
        clean[key] = [pre + '*' for pre in string.split()]
    return clean


def car2sph(sh, cart):
    """
    Turns symbolic solid harmonic functions into a dictionary of
    arrays containing cartesian to spherical transformation matrices.

    Args
        sh (OrderedDict): the result of solid_harmonics(l_tot)
        cart (dict): dictionary of l, cartesian l, m, n ordering
    """
    conv, prevL, mlcnt = {}, 0, 0
    for (L, ml), sym in sh.items():
        if L > 5: continue
        mlcnt = mlcnt if prevL == L else 0
        conv.setdefault(L, np.zeros((cart_lml_count[L],
                                     spher_lml_count[L]),
                                    dtype=np.float64))
        coefs = sym.expand().as_coefficients_dict()
        for i, (l, m, n) in enumerate(cart[L]):
            if L == 1:
                conv[L] = np.array(cart[L])
                break
            key = x ** l * y ** m * z ** n
            conv[L][i, mlcnt] = coefs[key]
        prevL = L
        mlcnt += 1
    return conv



@jit(nopython=True, cache=True)
def _fac(n,v): return _fac(n-1, n*v) if n else v

@jit(nopython=True, cache=True)
def fac(n): return _fac(n, 1)

@jit(nopython=True, cache=True)
def _fac2(n,v): return _fac2(n-2, n*v) if n > 0 else v

@jit(nopython=True, cache=True)
def fac2(n):
    if n < -1: return 0
    if n < 2: return 1
    return _fac2(n, 1)

@jit(nopython=True, cache=True)
def normalize(alpha, L):
    prefac = (2 / np.pi) ** (0.75)
    numer = 2 ** (L) * alpha ** ((L + 1.5) / 2)
    denom = (fac2(2 * L - 1)) ** (0.5)
    return prefac * numer / denom

@jit(nopython=True, cache=True)
def prim_normalize(alpha, l, m, n):
    L = l + m + n
    prefac = (2 / np.pi) ** (0.75)
    numer = 2 ** (L) * alpha ** ((L + 1.5) / 2)
    denom = (fac2(2*l - 1) * fac2(2*m - 1) * fac2(2*n - 1)) ** (0.5)
    return prefac * numer / denom

@jit(nopython=True, cache=True)
def cont_normalize(alphas, ds, l, m, n):
    L = l + m + n
    prefac = np.pi ** (1.5) / 2 ** L
    fc = fac2(2*l - 1) * fac2(2*m - 1) * fac2(2*n - 1)
    summ = 0.
    for ai, di in zip(alphas, ds):
        for aj, dj in zip(alphas, ds):
            summ += di * dj / (ai + aj) ** (L + 1.5)
    return (prefac * fc * summ) ** -0.5




@jit(nopython=True, cache=True)
def sto_normalize(alpha, n):
    return (2 * alpha) ** n * ((2 * alpha) / fac(2 * n)) ** 0.5

@vectorize(['int64(int64)'])
def _vec_fac(n):
    return fac(n)

@vectorize(['int64(int64)'])
def _vec_fac2(n):
    return fac2(n)

@vectorize(['float64(float64,int64)'])
def _vec_normalize(alpha, L):
    return normalize(alpha, L)

@vectorize(['float64(float64,int64,int64,int64)'])
def _vec_prim_normalize(alpha, l, m, n):
    return prim_normalize(alpha, l, m, n)

@vectorize(['float64(float64,int64)'])
def _vec_sto_normalize(alpha, n):
    return sto_normalize(alpha, n)

### Is this necessary?
@jit(nopython=True, cache=True)
def _ovl_indices(nbas, nel):
    chis = np.empty((nel, 2), dtype=np.int64)
    cnt = 0
    for i in range(nbas):
        for j in range(i + 1):
            chis[cnt, 0] = i
            chis[cnt, 1] = j
            cnt += 1
    return chis


@jit(nopython=True, cache=True)
def _nin(o1, o2, po1, po2, gamma, pg12):
    otot = o1 + o2
    if not otot:
        return pg12
    oio = 0.
    ii = ((otot - 1) // 2 + 1) if (otot % 2) else (otot // 2 + 1)
    for i in range(ii):
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
def _overlap_1c(alp1, alp2, l1, m1, n1, l2, m2, n2):
    """Compute overlap of primitive gaussians on the same center."""
    ll = l1 + l2
    mm = m1 + m2
    nn = n1 + n2
    if ll % 2 or mm % 2 or nn % 2:
        return 0
    ltot = ll // 2 + mm // 2 + nn // 2
    numer = np.pi ** (1.5) * fac2(ll - 1) * fac2(mm - 1) * fac2(nn - 1)
    denom = (2 ** ltot) * (alp1 + alp2) ** (ltot + 1.5)
    return numer / denom


@jit(nopython=True, cache=True)
def _overlap_2c(ab2, pax, pay, paz, pbx, pby, pbz,
                gamma, alp1, alp2, l1, m1, n1, l2, m2, n2):
    """Compute the overlap between two cartesian gaussians."""
    pg12 = np.sqrt(np.pi / gamma)
    xix = _nin(l1, l2, pax, pbx, gamma, pg12)
    yiy = _nin(m1, m2, pay, pby, gamma, pg12)
    ziz = _nin(n1, n2, paz, pbz, gamma, pg12)
    exp = alp1 * alp2 * ab2 / gamma
    return np.exp(-exp) * xix * yiy * ziz
    #return N1 * N2 * np.exp(-exp) * xix * yiy * ziz



@jit(nopython=True, cache=True)
#def _wrap_overlap(x, y, z, l, m, n, N, alpha):
def _wrap_overlap(x, y, z, l, m, n, N, alpha):
    nprim, cnt = len(x), 0
    arlen = nprim * (nprim + 1) // 2
    chi0 = np.empty(arlen, dtype=np.int64)
    chi1 = np.empty(arlen, dtype=np.int64)
    ovl = np.empty(arlen, dtype=np.float64)
    for i in range(nprim):
        for j in range(i + 1):
            chi0[cnt] = i
            chi1[cnt] = j
            xA = x[i]
            yA = y[i]
            zA = z[i]
            xB = x[j]
            yB = y[j]
            zB = z[j]
            l1 = l[i]
            m1 = m[i]
            n1 = n[i]
            l2 = l[j]
            m2 = m[j]
            n2 = n[j]
            # Unused?
            N1 = N[i]
            N2 = N[j]
            #
            alpha1 = alpha[i]
            alpha2 = alpha[j]
            abx = xA - xB
            aby = yA - yB
            abz = zA - zB
            ab2 = abx ** 2 + aby ** 2 + abz ** 2
            if ab2 < 1e-8:
                ovl[cnt] = _overlap_1c(alpha1, alpha2,
                                                l1, m1, n1,
                                                l2, m2, n2)
            else:
                gamma = alpha1 + alpha2
                xP = (alpha1 * xA + alpha2 * xB) / gamma
                yP = (alpha1 * yA + alpha2 * yB) / gamma
                zP = (alpha1 * zA + alpha2 * zB) / gamma
                pax = xP - xA
                pay = yP - yA
                paz = zP - zA
                pbx = xP - xB
                pby = yP - yB
                pbz = zP - zB
                ovl[cnt] = _overlap_2c(ab2, pax, pay, paz,
                                               pbx, pby, pbz, gamma,
                                               alpha1, alpha2,
                                               l1, m1, n1,
                                               l2, m2, n2)
            cnt += 1
    return chi0, chi1, ovl
