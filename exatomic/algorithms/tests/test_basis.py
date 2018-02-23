# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Tests for symbolic basis functions
"""
import numpy as np
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
        # self.nsh = new_solid_harmonics(6)

    def test_solid_harmonics(self):
        for L in range(self.L):
            for ml in range(-L, L + 1):
                self.assertIn(ml, self.sh[L])

    # def test_new_solid_harmonics(self):
    #     for L in range(self.L):
    #         self.assertIn(L, self.nsh)
    #         for ml in range(-L, L + 1):
    #             self.assertIn(ml, self.nsh[L])

    # def test_symbolic(self):
    #     for L in range(self.L):
    #         for ml in range(-L, L + 1):
    #             self.assertEquals(self.sh[(L, ml)], self.nsh[L][ml])

    def test_car2sph(self):
        c2s = car2sph(self.sh, enum_cartesian)
        for L in range(self.L):
            c = cart_lml_count[L]
            s = spher_lml_count[L] if L else 3
            self.assertEqual(c2s[L].shape, (c, s))

    # def test_new_car2sph(self):
    #     c2s = new_car2sph(self.nsh, enum_cartesian)
    #     for L in range(self.L):
    #         c = cart_lml_count[L]
    #         s = spher_lml_count[L] if L else 3
    #         self.assertEqual(c2s[L].shape, (c, s))

    # def test_transform_matrices(self):
    #     for L in range(self.L):
