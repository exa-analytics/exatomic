# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Numba Extensions
####################
The `numba`_ package provides a mechanism for compiling Python code. With
appropriate options, compilation can provide massive speedups to standard
Python code. This module provides reasonable compilation options as well as
a utility for compiling symbolic (or string) functions to 'numbafied' functions.

See Also:
    `sympy`_, `symengine`_, and the compilation engine `numba`_

.. _sympy: http://docs.sympy.org/latest/index.html
.. _symengine: https://github.com/symengine/symengine
.. _numba: http://numba.pydata.org/
"""
import numpy as np
import sympy as sy
import numba as nb
from warnings import warn
from platform import system
from sympy.utilities.lambdify import NUMPY_TRANSLATIONS, NUMPY_DEFAULT


npvars = vars(np)
npvars.update(NUMPY_DEFAULT)
npvars.update({k: getattr(np, v) for k, v in NUMPY_TRANSLATIONS.items()})
if "linux" in system().lower():
    jitkwargs = dict(nopython=True, nogil=True, parallel=True)
    veckwargs = dict(nopython=True, target="parallel")
else:
    jitkwargs = dict(nopython=True, nogil=True, parallel=False, cache=True)
    veckwargs = dict(nopython=True, target="cpu")


def numbafy(fn, args, compiler="jit", **nbkws):
    """
    Compile a string, sympy expression or symengine expression using numba.

    Not all functions are supported by Python's numerical package (numpy). For
    difficult cases, valid Python code (as string) may be more suitable than
    symbolic expressions coming from sympy, symengine, etc. When compiling
    vectorized functions, include valid signatures (see `numba`_ documentation).

    Args:
        fn: Symbolic expression as sympy/symengine expression or string
        args (iterable): Symbolic arguments
        compiler: String name or callable numba compiler
        nbkws: Compiler keyword arguments (if none provided, smart defaults are used)

    Returns:
        func: Compiled function

    Warning:
        For vectorized functions, valid signatures are (almost always) required.
    """
    kwargs = {}    # Numba kwargs to be updated by user
    if not isinstance(args, (tuple, list)):
        args = (args, )
    # Parameterize compiler
    if isinstance(compiler, str):
        compiler_ = getattr(nb, compiler, None)
        if compiler is None:
            raise AttributeError("No numba function with name {}.".format(compiler))
        compiler = compiler_
    if compiler in (nb.jit, nb.njit):
        kwargs.update(jitkwargs)
        sig = nbkws.pop("signature", None)
    else:
        kwargs.update(veckwargs)
        sig = nbkws.pop("signatures", None)
        if sig is None:
            warn("Vectorization without 'signatures' can lead to wrong results!")
    kwargs.update(nbkws)
    # Expand sympy expressions and create string for eval
    if isinstance(fn, sy.Expr):
        fn = sy.expand_func(fn)
    func = sy.lambdify(args, fn, modules='numpy')
    # Machine code compilation
    if sig is None:
        try:
            func = compiler(**kwargs)(func)
        except RuntimeError:
            kwargs['cache'] = False
            func = compiler(**kwargs)(func)
    else:
        try:
            func = compiler(sig, **kwargs)(func)
        except RuntimeError:
            kwargs['cache'] = False
            func = compiler(sig, **kwargs)(func)
    return func

