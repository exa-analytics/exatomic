# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Tests for :mod:`~exa.util.nbvars`
##################################
Test that :func:`~exa.util.nbvars.numbafy` supports strings, and sympy
and symengine expressions.
"""
import os
import pytest
import platform
import numpy as np
import sympy as sy
try:
    from numba.core.errors import TypingError
except ImportError:
    from numba.errors import TypingError
from exa.util.nbvars import numbafy


NUMBA_DISABLE_JIT = os.environ.get("NUMBA_DISABLE_JIT", 0) == 1 or platform.system() == "Windows"


@pytest.fixture
def sca():
    return 0.1

@pytest.fixture
def arr():
    return np.random.rand(10).astype("float32")

@pytest.fixture
def sig1():
    return ["float32(float32)"]

@pytest.fixture
def sig3():
    return ["float32(float32, float32, float32)"]


# Tests start here!
@pytest.mark.skipif(NUMBA_DISABLE_JIT, reason="Numba compilation disabled")
def test_simple_strings(arr, sig1, sca):
    """Test string functions."""
    fn = "sin(x)/x"
    func = numbafy(fn, "x", compiler="vectorize", signatures=sig1)
    assert np.allclose(func(arr), np.sin(arr)/arr) == True
    func = numbafy(fn, "x", compiler="jit")
    assert np.isclose(func(sca), np.sin(sca)/sca) == True


@pytest.mark.skipif(NUMBA_DISABLE_JIT, reason="Numba compilation disabled")
def test_fail_string(sca):
    """Test failure on untyped name."""
    fn = "Sin(x)/x"
    func = numbafy(fn, "x")
    with pytest.raises((TypingError, NameError)):
        func(sca)


@pytest.mark.skipif(NUMBA_DISABLE_JIT, reason="Numba compilation disabled")
def test_complex_strings(arr, sig3, sca):
    """Test more complicated string functions."""
    fn = "arccos(x)/y + exp(-y) + mod(z, 2)"
    func = numbafy(fn, ("x", "y", "z"), compiler="vectorize", signatures=sig3)
    result = func(arr, arr, arr)
    check = np.arccos(arr)/arr + np.exp(-arr) + np.mod(arr, 2)
    assert np.allclose(result, check) == True
    func = numbafy(fn, ("x", "y", "z"))
    result = func(sca, sca, sca)
    check = np.arccos(sca)/sca + np.exp(-sca) + np.mod(sca, 2)
    assert np.isclose(result, check) == True


@pytest.mark.skipif(NUMBA_DISABLE_JIT, reason="Numba compilation disabled")
def test_sympy(arr, sig3):
    """Test sympy expressions."""
    x, y, z = sy.symbols("x y z")
    fn = sy.acos(x)/y + sy.exp(-y) + sy.Mod(z, 2)
    func = numbafy(fn, (x, y, z), compiler="vectorize", signatures=sig3)
    result = func(arr, arr, arr)
    check = np.arccos(arr)/arr + np.exp(-arr) + np.mod(arr, 2)
    assert np.allclose(result, check) == True


@pytest.mark.skipif(NUMBA_DISABLE_JIT, reason="Numba compilation disabled")
def test_symengine(arr, sig3):
    """Test symengine."""
    try:
        import symengine as sge
        x, y, z = sge.var("x y z")
        fn = sge.acos(x)/y + sge.exp(-z)
        func = numbafy(fn, (x, y, z), compiler="vectorize", signatures=sig3)
        result = func(arr, arr, arr)
        check = np.arccos(arr)/arr + np.exp(-arr)
        assert np.allclose(result, check) == True
    except ImportError:
        pass

