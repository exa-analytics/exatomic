# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
r"""
Gaussian Type Basis Functions
#################################
This modules defines the functional forms of Gaussian type functions.

Primitive Cartesian Gaussian Functions (1D, 3D)
================================================
These functions are the original workhorses of quantum mechanics. They are
analytically integrable and differentiable but are not unique in the angular
space define by the orbital and spin angular momentum quantum numbers. They
are not typically normalized (normalization constants are omitted below).

.. math::

    G_m(q, a, A_q) = (q - A_q)^m e^{-a (q - A_q)^2} = q_A^m e^{-a q_A^2}

    G_{ijk}\left(\mathbf{r}, a, \mathbf{A}\right) =
        (x - A_x)^i e^{-a (x - A_x)^2}
        (y - A_y)^j e^{-a (y - A_y)^2}
        (z - A_z)^k e^{-a (z - A_z)^2}
        = x_A^iy_A^jz_A^k e^{-a r^2}
"""
import numba as nb
import sympy as sy


# Define some private symbols used in this module
# Note the assumptions used
_xyz = sy.symbols("x y z", real=True)
_ijk = sy.symbols("i j k", integer=True, positive=True)
_a = sy.symbols("a", positive=True)
_N = sy.symbols("N", positive=True)   # Conventional choice (+/- solutions)
_xyzenum = {'x': 0, 'y': 1, 'z': 2}


def primitive_gto_1d(q="x"):
    """
    Return a 1D primitive Cartesian Gaussian function.

    Args:
        q (str): Cartesian component

    Returns:
        expr: Symbolic expression for 1D primitive Cartesian Gaussian
    """
    nn = _xyzenum[q]
    q = _xyz[nn]
    n = _ijk[nn]
    return q**n*sy.exp(-_a*q**2)


def primitive_gto_3d():
    """
    Return a 3D primitive Cartesian Gaussian function.

    Returns:
        expr: Symbolic expression for 3D primitive Cartesian Gaussian
    """
    return sy.simplify(primitive_gto_1d("x")*primitive_gto_1d("y")*primitive_gto_1d("z"))


def normalize(expr, normsymbol=None, normkey=0):
    """
    Given an expression, with or without an explicitly defined normalization constant,
    solve for the value of the normalization constant and return the complete expression.
    """
    if normsymbol is None:
        normsymbol = _N
        expr = normsymbol*expr
    x, y, z = _xyz
    expr = sy.simplify(expr)
    integral = sy.integrate(expr**2, (x, -sy.oo, sy.oo), (y, -sy.oo, sy.oo), (z, -sy.oo, sy.oo))
    norms = sy.solve(sy.Eq(integral, 1), normsymbol)
    return sy.simplify(expr.subs({normsymbol: norms[normkey]}))


_prim1d = primitive_gto_1d()
_prim3d = primitive_gto_3d()



