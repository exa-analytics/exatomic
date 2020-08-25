# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Tests for symbolic basis function utilities
#############################################
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from unittest import TestCase
from exatomic.base import resource
from exatomic import nwchem, molcas
from ..basis import (cart_lml_count, spher_lml_count, solid_harmonics,
                     enum_cartesian, car2sph, BasisFunctions)


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


class TestBasisFunctions(TestCase):

    def setUp(self):
        self.nw = nwchem.Output(resource('nw-ch3nh2-631g.out')).to_universe()
        mo = molcas.Output(resource('mol-ch3nh2-631g.out'))
        mo.add_orb(resource('mol-ch3nh2-631g.scforb'))
        self.mo = mo.to_universe()

    def test_basis_functions(self):
        nwfns = self.nw.basis_functions.evaluate()
        mofns = self.mo.basis_functions.evaluate()
        nw = sorted(list(nwfns[0].expand().as_coefficients_dict().values()))
        mo = sorted(list(mofns[0].expand().as_coefficients_dict().values()))
        for a, b in zip(nw, mo):
            self.assertTrue(np.isclose(np.float64(a), np.float64(b)))
        self.assertFalse(len(nwfns[11].expand().as_coefficients_dict()) ==
                         len(mofns[11].expand().as_coefficients_dict()))
