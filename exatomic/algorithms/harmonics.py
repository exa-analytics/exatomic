# -*- coding: utf-8 -*-
"""
Spherical and Solid Harmonics
=================================================
These functions generate and manipulate spherical and solid harmonics. For solid harmonics, this
module provides numerical approaches for dealing with them.
"""
import re
import sympy as sy
import numba as nb
from sympy.parsing.sympy_parser import parse_expr
from sympy.physics.secondquant import KroneckerDelta as kr


class SolidHarmonics:
    """
    Store a collection of solid harmonic functions.
    """
    @property
    def _constructor(self):
        return SolidHarmonics


def solid_harmonics(l, return_all=False, vectorize=False, standard_symbols=True):
    """
    Generate a set of spherical solid harmonic functions for a given angular
    momentum.

        >>> solid_harmonics(0)
        {(0, 0): 1}
        >>> solid_harmonics(1, True)
        {(0, 0): 1, (1, 1): x, (1, -1): y, (1, 0): z}

    Args:
        l (int): Orbital angular moment
        return_all (bool): If true, return all computed solid harmonics
        vectorize (bool): If true, return vectorized functions (for numerical methods)
        standard_symbols (bool): If true (default), convert to standard symbol notation (e.g. x*y => xy)

    Returns:
        functions (dict): Dictionary of (l, ml) keys and symbolic function values
    """
    x, y, z = sy.symbols('x y z', imaginary=False)
    r2 = x**2 + y**2 + z**2
    desired_l = l
    # Recursion relations come from Molecular Electronic Structure, Helgaker 2nd ed.
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
        funcs = {}
        for key in s:
            if isinstance(s[key], int):
                funcs[key] = ([None], lambda r: r)
            else:
                symbols = sorted([str(sym) for sym in s[key].free_symbols])
                if symbols == ['x'] or symbols == ['y'] or symbols == ['z']:
                    funcs[key] = (symbols, lambda r: r)
                else:
                    f = sy.lambdify(symbols, s[key], 'numpy')
                    vec = nb.vectorize(['float64({})'.format(', '.join(['float64'] * len(symbols)))], nopython=True)
                    f = vec(f)
                    funcs[key] = (symbols, f)
        s = funcs
    if return_all:
        return s
    return {key: value for key, value in s.items() if key[0] == desired_l}
