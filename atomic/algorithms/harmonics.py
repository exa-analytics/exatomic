# -*- coding: utf-8 -*-
'''
Spherical and Solid Harmonics
=================================================
These functions generate and manipulate spherical and solid harmonics. For solid harmonics, this
module provides numerical approaches for dealing with them.
'''
import re
import numpy as np
import pandas as pd
import sympy as sy
from sympy.parsing.sympy_parser import parse_expr
from sympy.physics.secondquant import KroneckerDelta as kr
from exa import _conf


def solid_harmonics(l, return_all=False, vectorize=False, standard_symbols=True):
    '''
    Generate a set of spherical solid harmonic functions for a given angular
    momentum.

        >>> solid_harmonics(0)
        {(0, 0): 1}
        >>> solid_harmonics(1, True)
        {(0, 0): 1, (1, -1): y, (1, 0): z, (1, 1): x}

    Args:
        l (int): Orbital angular moment
        return_all (bool): If true, return all computed solid harmonics
        vectorize (bool): If true, return vectorized functions (for numerical methods)
        standard_symbols (bool): Convert to standard symbol notation (e.g. x*y => xy)

    Returns:
        functions (dict): Dictionary of (l, ml) keys and symbolic function values
    '''
    x, y, z = sy.symbols('x y z', imaginary=False)
    r2 = x**2 + y**2 + z**2
    desired_l = l
    s = {(0,0): 1}
    for l in range(1, desired_l + 1):
        lminus1 = l - 1 if l >= 1 else 0
        negl = -lminus1 if lminus1 != 0 else 0
        # top
        s[(l, l)] = sy.sqrt(2**kr(lminus1, 0) * (2 * lminus1 + 1) / (2 * lminus1 + 2)) * \
                    (x * s[(lminus1, lminus1)] - (1 - kr(lminus1, 0)) * y * s[(lminus1, negl)])
        # bottom
        s[(l, negl - 1)] = sy.sqrt(2**kr(lminus1, 0) * (2 * lminus1 + 1) / (2 * lminus1 + 2)) * \
                           (y * s[(lminus1, lminus1)] + (1 - kr(lminus1, 0)) * x * s[(lminus1, negl)])
        for m in range(-l, l + 1)[1:-1]:
            lminus2 = lminus1 - 1 if lminus1 - 1 >= 0 else 0
            s_lminus2_m = 0
            if (lminus2, m) in s:
                s_lminus2_m = s[(lminus2, m)]
            s[(l, m)] = ((2 * lminus1 + 1) * z * s[(lminus1, m)] - sy.sqrt((lminus1 + m) * (lminus1 - m)) * \
                         r2 * s_lminus2_m) / sy.sqrt((lminus1 + m + 1) * (lminus1 - m + 1))
    # If true, transform the symbolic notation of things like x*y (which represents dxy)
    # to simply xy (which is also a symbol and therefore possible to manipulate
    # with .subs({})).
    if standard_symbols:
        match1 = r'([xyz])\*\*(\d+)'
        for i in range(2, desired_l+1):
            match0 = r'\*'.join(['([xyz])'] * i)
            replace0 = r''.join(['\\' + str(j) for j in range(1, i+1)])
            for k, v in [(k, v) for k, v in s.items() if k[0] == i]:
                expr = str(v.expand())
                expr = re.sub(match0, replace0, expr)
                while len(re.findall(match1, expr)) > 0:
                    for arg, count in re.findall(match1, expr):
                        count = int(count)
                        f = r''.join([arg[0], r'\*\*', str(count)])
                        r = r''.join([arg[0]] * count)
                        expr = re.sub(f, r, expr)
                        expr = re.sub(match0, replace0, expr)
                for j in range(1, i + 1):
                    match0 = r'\*'.join(['([xyz])'] * j)
                    replace0 = r''.join(['\\' + str(k) for k in range(1, j+1)])
                    expr = re.sub(match0, replace0, expr)
                s[k] = parse_expr(expr)
    if vectorize:
        if _conf['pkg_numba']:
            from numba import float64, vectorize
            for key, func in ((key, func) for key, func in s.items() if key[0] > 1):
                args = [str(arg) for arg in func.free_symbols]
                j = vectorize(['float64({})'.format(','.join(['float64'] * len(args)))], nopython=True)
                lamfunc = sy.lambdify([str(arg) for arg in func.free_symbols], func, 'numpy')
                s[key] = j(lamfunc)
        else:
            for key, func in ((key, func) for key, func in s.items() if key[0] > 1):
                lamfunc = sy.lambdify([str(arg) for arg in func.free_symbols], func, 'numpy')
                s[key] = np.vectorize(lamfunc)
    if return_all:
        return s
    return {key: value for key, value in s.items() if key[0] == desired_l}
