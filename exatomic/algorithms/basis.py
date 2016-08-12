# -*- coding: utf-8 -*-
'''
Basis Function Manipulation
################################
Functions for managing and manipulating basis set data.
Many of the ordering schemes used in computational codes can be
generated programmatically with the right numerical function.
This is preferred to an explicit parsing and storage of a given
basis set ordering scheme.
'''
import sympy
import numpy as np
from exatomic._config import config
from collections import OrderedDict

x, y, z = sympy.symbols('x y z')

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
                      [1, 2, 0], [1, 0, 2], [1, 1, 1],
                      [0, 3, 0], [0, 2, 1], [0, 1, 2], [0, 0, 3]],
                  4: [[4, 0, 0], [3, 1, 0], [3, 0, 1], [2, 2, 0], [2, 1, 1],
                      [2, 0, 2], [1, 3, 0], [1, 2, 1], [1, 1, 2], [1, 0, 3],
                      [0, 4, 0], [0, 3, 1], [0, 2, 2], [0, 1, 3], [0, 0, 4]],
                  5: [[5, 0, 0], [4, 1, 0], [4, 0, 1], [3, 2, 0], [3, 1, 1],
                      [3, 0, 2], [2, 3, 0], [2, 2, 1], [2, 1, 2], [2, 0, 3],
                      [1, 4, 0], [1, 3, 1], [1, 2, 2], [1, 1, 3], [1, 0, 4],
                      [0, 5, 0], [0, 4, 1], [0, 3, 2], [0, 2, 3], [0, 1, 4],
                      [0, 0, 5]]}

#def cartesian_gtf_exponents(l):
#    '''
#    Generic generation of cartesian Gaussian type function exponents.
#
#    Generates the linearly dependent, :math:`i`, :math:`j`, :math:`k`, values for the Gaussian
#    type functions of the form:
#
#    .. math::
#
#        f(x, y, z) = x^{i}y^{j}z^{k}e^{-\alpha r^{2}}
#
#    Args:
#        l (int): Orbital angular momentum
#
#    Returns:
#        array: Array of i, j, k values for cartesian Gaussian type functions
#
#    Note:
#        This returns the linearly dependent indices (array) in arbitrary
#        order.
#    '''
#    m = l + 1
#    n = (m + 1) * m // 2
#    values = np.empty((n, 3), dtype=np.int64)
#    h = 0
#    for i in range(m):
#        for j in range(m):
#            for k in range(m):
#                if i + j + k == l:
#                    values[h] = [k, j, i]
#                    h += 1
#    return values


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
    sh[(0, 0)] = 1
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


def car2sph_transform_matrices(sh, l_tot):
    '''
    Generates cartesian to spherical transformation matrices as an ordered dict
    with key corresponding to l value.

    Args
        sh (OrderedDict): the result of solid_harmonics(l_tot)
    '''
    s = [1]
    p = [y, z, x]
    d = [x*x, x*y, x*z, y*y, y*z, z*z]
    f = [x*x*x, x*x*y, x*x*z, x*y*y, x*y*z, x*z*z, y*y*y, y*y*z, y*z*z, z*z*z]
    g = [x*x*x*x, x*x*x*y, x*x*x*z, x*x*y*y, x*x*y*z,
         x*x*z*z, x*y*y*y, x*y*y*z, x*y*z*z, x*z*z*z,
         y*y*y*y, y*y*y*z, y*y*z*z, y*z*z*z, z*z*z*z]
    h = [x*x*x*x*x, x*x*x*x*y, x*x*x*x*z, x*x*x*y*y, x*x*x*y*z, x*x*x*z*z, x*x*y*y*y,
         x*x*y*y*z, x*x*y*z*z, x*x*z*z*z, x*y*y*y*y, x*y*y*y*z, x*y*y*z*z, x*y*z*z*z,
         x*z*z*z*z, y*y*y*y*y, y*y*y*y*z, y*y*y*z*z, y*y*z*z*z, y*z*z*z*z, z*z*z*z*z]
    ltopow = {0: s, 1: p, 2: d, 3: f, 4: g, 5: h}
    transdims = {0: (1, 1), 1: (3, 3), 2: (5, 6),
                 3: (7, 10), 4: (9, 15), 5: (11, 21)}
    ndict = OrderedDict()
    for lcur in range(l_tot + 1):
        ndict[lcur] = np.zeros(transdims[lcur])
        for ml in range(-lcur, lcur + 1):
            moff = lcur + ml
            expr = sh[(lcur, ml)]
            powers = ltopow[lcur]
            try:
                nexpr = expr.as_coeff_Mul()
            except AttributeError:
                ndict[lcur][moff,0] = expr
                continue
            for i, power in enumerate(powers):
                if float(nexpr[0]).is_integer():
                    ndict[lcur][moff, i] = sympy.expand(nexpr[1]).coeff(power, 1)
                else:
                    if power == nexpr[1]:
                        ndict[lcur][moff, powers.index(power)] = nexpr[0]
    return ndict
#from numba import jit, vectorize, int64, float64

def fac(n):
    if n < 0: return 0
    if n == 0: return 1
    ns = np.empty((n), dtype=np.int64)
    for i in enumerate(ns):
        ns[i[0]] = n
        n -= 1
    return np.prod(ns)


def fac2(n):
    if n < -1: return 0
    if n < 2: return 1
    ns = np.empty((n//2,), dtype=np.int64)
    for i in enumerate(ns):
        ns[i[0]] = n
        n -= 2
    return np.prod(ns)

def normalize(alpha, l, m, n):
    prefac = (2 / np.pi) ** (0.75)
    numer = 2 ** (l + m + n) * alpha ** ((l + m + n + 1.5) / 2)
    denom = (fac2(2 * l - 1) *
             fac2(2 * m - 1) *
             fac2(2 * n - 1)) ** (0.5)
    return prefac * numer / denom

def sloppy_normalize(alpha, L):
    prefac = (2 / np.pi) ** (0.75)
    numer = 2 ** (L) * alpha ** ((L + 1.5) / 2)
    denom = (fac2(2 * L - 1)) ** (0.5)
    return prefac * numer / denom

def _vec_fac(n):
    return fac(n)

def _vec_fac2(n):
    return fac2(n)

def _vec_normalize(alpha, l, m, n):
    prefac = (2 / np.pi) ** (0.75)
    numer = 2 ** (l + m + n) * alpha ** ((l + m + n + 1.5) / 2)
    denom = (_vec_fac2(2 * l - 1) *
             _vec_fac2(2 * m - 1) *
             _vec_fac2(2 * n - 1)) ** (0.5)
    return prefac * numer / denom

def _vec_sloppy_normalize(alpha, L):
    prefac = (2 / np.pi) ** (0.75)
    numer = 2 ** (L) * alpha ** ((L + 1.5) / 2)
    denom = (_vec_fac2(2 * L - 1)) ** (0.5)
    return prefac * numer / denom

def _overlap(x1, x2, y1, y2, z1, z2, l1, l2, m1, m2, n1, n2, N1, N2, alpha1, alpha2):
    '''
    Pardon the Fortran style that follows. This was translated from the snafu
    electronic structure software package.
    '''
    s12 = 0.
    tol = 1e-8
    abx = x1 - x2
    aby = y1 - y2
    abz = z1 - z2
    ab2 = abx * abx + aby * aby + abz * abz
    if ab2 < tol:
        ll = l1 + l2
        mm = m1 + m2
        nn = n1 + n2
        if ll % 2 != 0 or mm % 2 != 0 or nn % 2 != 0:
            return s12
        ll2 = ll // 2
        mm2 = mm // 2
        nn2 = nn // 2
        ltot = ll2 + mm2 + nn2
        numer = np.pi ** (1.5) * fac2(ll - 1) * fac2(mm - 1) * fac2(nn - 1)
        denom = (2 ** ltot) * (alpha1 + alpha2) ** (ltot + 1.5)
        s12 = N1 * N2 * numer / denom
        return s12
    gamma = alpha1 + alpha2
    xp = (alpha1 * x1 + alpha2 * x2) / gamma
    yp = (alpha1 * y1 + alpha2 * y2) / gamma
    zp = (alpha1 * z1 + alpha2 * z2) / gamma
    px1 = xp - x1
    py1 = yp - y1
    pz1 = zp - z1
    px2 = xp - x2
    py2 = yp - y2
    pz2 = zp - z2
    pg12 = np.sqrt(np.pi / gamma)
    xix = 0
    yiy = 0
    ziz = 0
    ltot = l1 + l2
    mtot = m1 + m2
    ntot = n1 + n2
    if ltot == 0:
        xix = pg12
    else:
        iii = (ltot - 1) // 2 if ltot % 2 != 0 else ltot // 2
        for i in range(iii):
            k = 2 * i
            prod = pg12 * fac2(k - 1) / ((2 * gamma) ** i)
            qlow = max(-k, (k - 2 * l2))
            qhigh = min(k, (2 * l1 - k))
            fk = 0
            for q in range(qlow, qhigh, 2):
                j = (k + q) // 2
                k = (k - q) // 2
                newt1 = fac(l1) / fac(j) / fac(l1 - j)
                newt2 = fac(l2) / fac(k) / fac(l2 - k)
                fk += newt1 * newt2 * (px1 ** (l1 - j)) * (px2 ** (l2 - k))
            xix += prod * fk
    if mtot == 0:
        yiy = pg12
    else:
        iii = (mtot - 1) // 2 if mtot % 2 != 0 else mtot // 2
        for i in range(iii):
            k = 2 * i
            prod = pg12 * fac2(k - 1) / ((2 * gamma) ** i)
            qlow = max(-k, (k - 2 * m2))
            qhigh = min(k, (2 * m1 - k))
            fk = 0
            for q in range(qlow, qhigh, 2):
                j = (k + q) // 2
                k = (k - q) // 2
                newt1 = fac(m1) / fac(j) / fac(m1 - j)
                newt2 = fac(m2) / fac(k) / fac(m2 - k)
                fk += newt1 * newt2 * (py1 ** (m1 - j)) * (py2 ** (m2 - k))
            yiy += prod * fk
    if ntot == 0:
        ziz = pg12
    else:
        iii = (ntot - 1) // 2 if ntot % 2 != 0 else ntot // 2
        for i in range(iii):
            k = 2 * i
            prod = pg12 * fac2(k - 1) / ((2 * gamma) ** i)
            qlow = max(-k, (k - 2 * n2))
            qhigh = min(k, (2 * n1 - k))
            fk = 0
            for q in range(qlow, qhigh, 2):
                j = (k + q) // 2
                k = (k - q) // 2
                newt1 = fac(n1) / fac(j) / fac(n1 - j)
                newt2 = fac(n2) / fac(k) / fac(n2 - k)
                fk += newt1 * newt2 * (pz1 ** (n1 - j)) * (pz2 ** (n2 - k))
            ziz += prod * fk
    exponent = alpha1 * alpha2 * ab2 / gamma
    s12 = N1 * N2 * np.exp(-exponent) * xix * yiy * ziz
    return s12


def _wrap_overlap(x, y, z, l, m, n, N, alpha):
    nprim = len(x)
    arlen = nprim * (nprim + 1) // 2
    f1x = np.empty(arlen, dtype=np.float64)
    f1y = np.empty(arlen, dtype=np.float64)
    f1z = np.empty(arlen, dtype=np.float64)
    f1N = np.empty(arlen, dtype=np.float64)
    f1a = np.empty(arlen, dtype=np.float64)
    f1l = np.empty(arlen, dtype=np.int64)
    f1m = np.empty(arlen, dtype=np.int64)
    f1n = np.empty(arlen, dtype=np.int64)
    f2x = np.empty(arlen, dtype=np.float64)
    f2y = np.empty(arlen, dtype=np.float64)
    f2z = np.empty(arlen, dtype=np.float64)
    f2N = np.empty(arlen, dtype=np.float64)
    f2a = np.empty(arlen, dtype=np.float64)
    f2l = np.empty(arlen, dtype=np.int64)
    f2m = np.empty(arlen, dtype=np.int64)
    f2n = np.empty(arlen, dtype=np.int64)
    chi1 = np.empty(arlen, dtype=np.int64)
    chi2 = np.empty(arlen, dtype=np.int64)
    cnt = 0
    for i in range(nprim):
        for j in range(i + 1):
            f1x[cnt] = x[i]
            f2x[cnt] = x[j]
            f1y[cnt] = y[i]
            f2y[cnt] = y[j]
            f1z[cnt] = z[i]
            f2z[cnt] = z[j]
            f1N[cnt] = N[i]
            f2N[cnt] = N[j]
            f1a[cnt] = alpha[i]
            f2a[cnt] = alpha[j]
            f1l[cnt] = l[i]
            f2l[cnt] = l[j]
            f1m[cnt] = m[i]
            f2m[cnt] = m[j]
            f1n[cnt] = n[i]
            f2n[cnt] = n[j]
            chi1[cnt] = i
            chi2[cnt] = j
            cnt += 1
    overlap = _overlap(f1x, f2x, f1y, f2y, f1z, f2z, f1l, f2l,
                       f1m, f2m, f1n, f2n, f1N, f2N, f1a, f2a)
    return chi1, chi2, overlap


if config['dynamic']['numba'] == 'true':
    from numba import jit, vectorize
    fac = jit(nopython=True)(fac)
    fac2 = jit(nopython=True)(fac2)
    normalize = jit(nopython=True)(normalize)
    sloppy_normalize = jit(nopython=True)(sloppy_normalize)
    _vec_fac = vectorize(['int64(int64)'])(_vec_fac)
    _vec_fac2 = vectorize(['int64(int64)'])(_vec_fac2)
    _vec_normalize = vectorize(['float64(float64,int64,int64,int64)'])(_vec_normalize)
    _vec_sloppy_normalize = vectorize(['float64(float64,int64)'])(_vec_sloppy_normalize)
    _overlap = vectorize(['float64(float64,float64,float64,float64,float64,float64,int64, \
                          int64,int64,int64,int64,int64,float64,float64,float64,float64)'])(_overlap)
    _wrap_overlap = jit()(_wrap_overlap)
