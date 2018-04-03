# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Tests for symbolic basis functions
######################################
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
#import numpy as np
from unittest import TestCase
from ..basis import (cart_lml_count, spher_lml_count, solid_harmonics,
                     enum_cartesian, car2sph)
                     #Basis, BasisFunction,
                     #CartesianBasisFunction,
                     #SphericalBasisFunction,
                     #normalize, prim_normalize, sto_normalize,
                     #_prim_cart_norm, _prim_sphr_norm, _prim_sto_norm,
                     #_cont_norm)


class TestCartesianToSpherical(TestCase):
    def setUp(self):
        self.L = 6
        self.sh = solid_harmonics(6)

    def test_solid_harmonics(self):
        for L in range(self.L):
            for ml in range(-L, L + 1):
                self.assertIn(ml, self.sh[L])


    def test_car2sph(self):
        c2s = car2sph(self.sh, enum_cartesian)
        for L in range(self.L):
            c = cart_lml_count[L]
            s = spher_lml_count[L] if L else 3
            self.assertEqual(c2s[L].shape, (c, s))


#class TestBasisFunctions(TestCase):
#
#    def setUp(self):
#        self.bargs = (0, 0, 0, 0, [1.], [1.])
#        self.cargs = (0, 0, 0, 0, [1.], [1.], 0, 0, 0)
#        self.sargs = (0, 0, 0, 0, [1.], [1.], 0, 0)
#        # Add gaussian{True/False}
#
#    def test_abstract(self):
#        with self.assertRaises(TypeError) as ctx:
#            BasisFunction(*self.bargs)
#
#    def test_cartesian(self):
#        f = CartesianBasisFunction(*self.cargs)
#        N = _prim_cart_norm(f.alphas, f.l, f.m, f.n)
#        self.assertTrue(np.allclose(f.Ns, N))
#
#    def test_sto(self):
#        f = CartesianBasisFunction(*self.cargs, gaussian=False)
#        N = _prim_sto_norm(f.alphas, f.rpow)
#        self.assertTrue(np.allclose(f.Ns, N))
#
#    def test_spherical(self):
#        f = SphericalBasisFunction(*self.sargs)
#        N = _prim_sphr_norm(f.alphas, f.L)
#        self.assertTrue(np.allclose(f.Ns, N))
#
#
#class TestNumerical(TestCase):
#    def setUp(self):
#        self.ns = range(-2, 10)
#
#    def test_fac(self):
#        chk = np.array([0, 0, 1, 1, 2, 6, 24, 120,
#                        720, 5040, 40320, 362880])
#        for c, n in zip(chk, self.ns):
#            self.assertEqual(fac(n), c)
#
#    def test_fac2(self):
#        chk = np.array([0, 1, 1, 1, 2, 3, 8,
#                        15, 48, 105, 384, 945])
#        for c, n in zip(chk, self.ns):
#            self.assertEqual(fac2(n), c)
#
#    def test_dfac21(self):
#        for n in self.ns:
#            self.assertEqual(fac2(2*n-1), dfac21(n))
#
#
#class TestNormalization(TestCase):
#    def setUp(self):
#        self.alphas = []
#
#    def test_normalize(self):
#        pass
#
#    def test_prim_normalize(self):
#        pass
#
#    def test_sto_normalize(self):
#        pass
