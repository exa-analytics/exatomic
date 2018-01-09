# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
import pandas as pd
from unittest import TestCase
from exatomic.core.basis import (BasisSet, BasisSetOrder, Overlap, Primitive,
                            spher_lml_count, cart_lml_count)

class TestBasisSet(TestCase):

    def setUp(self):
        adict = {col: [0] for col in BasisSet._columns}
        adict['frame'] = 0
        # Trivial basis set
        self.bs = BasisSet(adict, spherical=False)
        # Medium basis set
        self.mbs = BasisSet({'frame': 0,
                             'alpha': [5, 1, 1],
                             'shell': [0, 1, 0],
                               'set': [0, 0, 1],
                                 'd': [1, 1, 1],
                                 'L': [0, 1, 0],
                                 'n': [1, 2, 1]}, gaussian=False)
        # Large basis set
        self.lbs = BasisSet({'frame': 0,
                             'alpha': [5, 3, 1, 3, 1, 1, 3, 1, 1],
                             'shell': [0, 0, 0, 1, 1, 2, 0, 0, 1],
                               'set': [0, 0, 0, 0, 0, 0, 1, 1, 1],
                                 'd': [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 'L': [0, 0, 0, 1, 1, 2, 0, 0, 1]})

    def test_init(self):
        self.assertFalse(self.bs.spherical)
        self.assertFalse(self.mbs.gaussian)
        self.assertTrue(self.lbs.gaussian)
        self.assertTrue(self.lbs.spherical)

    def test_lmax(self):
        self.assertEquals(self.bs.lmax, 0)
        self.assertEquals(self.mbs.lmax, 1)
        self.assertEquals(self.lbs.lmax, 2)

    def test_shells(self):
        self.assertEquals(self.bs.shells, ['s'])
        self.assertEquals(self.mbs.shells, ['s', 'p'])
        self.assertEquals(self.lbs.shells, ['s', 'p', 'd'])

    def test_nshells(self):
        self.assertEquals(self.bs.nshells, 1)
        self.assertEquals(self.mbs.nshells, 2)
        self.assertEquals(self.lbs.nshells, 3)

    def test_functions_by_shell(self):
        self.assertEquals(type(self.bs.functions_by_shell()),
                          pd.Series)
        self.assertEquals(type(self.mbs.functions_by_shell()),
                          pd.Series)
        self.assertEquals(type(self.lbs.functions_by_shell()),
                          pd.Series)

    def test_primitives_by_shell(self):
        self.assertTrue(
            (self.bs.primitives_by_shell() == pd.Series({
            (0, 0): 1})).all())
        self.assertTrue(
            (self.mbs.primitives_by_shell() == pd.Series({
            (0, 0): 1, (0, 1): 1, (1, 0): 1})).all())
        self.assertTrue(
            (self.lbs.primitives_by_shell() == pd.Series({
            (0, 0): 3, (0, 1): 2, (0, 2): 1, (1, 0): 2,
            (1, 1) : 1})).all())

    def test_primitives(self):
        self.assertTrue(
            (self.bs.primitives(spher_lml_count) == pd.Series({
            0: 1})).all())
        self.assertTrue(
            (self.mbs.primitives(cart_lml_count) == pd.Series({
            0: 4, 1: 1})).all())
        self.assertTrue(
            (self.lbs.primitives(spher_lml_count) == pd.Series({
            0: 14, 1: 5})).all())
