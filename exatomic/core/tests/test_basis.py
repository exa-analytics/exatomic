# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
import pandas as pd
from unittest import TestCase
from exatomic.core.basis import (BasisSet, BasisSetOrder, Overlap,
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
    # 
    # def test_shells(self):
    #     self.assertEquals(self.bs.shells, ['s'])
    #     self.assertEquals(self.mbs.shells, ['s', 'p'])
    #     self.assertEquals(self.lbs.shells, ['s', 'p', 'd'])
    #
    # def test_nshells(self):
    #     self.assertEquals(self.bs.nshells, 1)
    #     self.assertEquals(self.mbs.nshells, 2)
    #     self.assertEquals(self.lbs.nshells, 3)

    def test_functions_by_shell(self):
        n = ['set', 'L']
        mfp = pd.MultiIndex.from_product
        mfa = pd.MultiIndex.from_arrays
        self.assertTrue((self.bs.functions_by_shell() ==
            pd.Series([1], index=mfp([[0], [0]], names=n))).all())
        self.assertTrue((self.mbs.functions_by_shell() ==
            pd.Series([1, 1, 1], index=mfa([[0, 0, 1], [0, 1, 0]], names=n))).all())
        self.assertTrue((self.lbs.functions_by_shell() ==
            pd.Series([1, 1, 1, 1, 1], index=mfa([[0, 0, 0, 1, 1],
                                                  [0, 1, 2, 0, 1]], names=n))).all())

    def test_primitives_by_shell(self):
        n = ['set', 'L']
        mfp = pd.MultiIndex.from_product
        mfa = pd.MultiIndex.from_arrays
        self.assertTrue((self.bs.primitives_by_shell() ==
            pd.Series([1], index=mfp([[0], [0]], names=n))).all())
        self.assertTrue((self.mbs.primitives_by_shell() ==
            pd.Series([1, 1, 1], index=mfa([[0, 0, 1], [0, 1, 0]], names=n))).all())
        self.assertTrue((self.lbs.primitives_by_shell() ==
            pd.Series([3, 2, 1, 2, 1], index=mfa([[0, 0, 0, 1, 1],
                                                  [0, 1, 2, 0, 1]], names=n))).all())

    def test_functions(self):
        n = ['set', 'L']
        mfp = pd.MultiIndex.from_product
        mfa = pd.MultiIndex.from_arrays
        self.assertTrue((self.bs.functions(False) ==
            pd.Series([1], index=mfp([[0], [0]], names=n))).all())
        self.assertTrue((self.bs.functions(True) ==
            pd.Series([1], index=mfp([[0], [0]], names=n))).all())
        self.assertTrue((self.mbs.functions(False) ==
            pd.Series([1, 3, 1], index=mfa([[0, 0, 1], [0, 1, 0]], names=n))).all())
        self.assertTrue((self.mbs.functions(True) ==
            pd.Series([1, 3, 1], index=mfa([[0, 0, 1], [0, 1, 0]], names=n))).all())
        self.assertTrue((self.lbs.functions(False) ==
            pd.Series([1, 3, 6, 1, 3], index=mfa([[0, 0, 0, 1, 1],
                                                  [0, 1, 2, 0, 1]], names=n))).all())
        self.assertTrue((self.lbs.functions(True) ==
            pd.Series([1, 3, 5, 1, 3], index=mfa([[0, 0, 0, 1, 1],
                                                  [0, 1, 2, 0, 1]], names=n))).all())


    def test_primitives(self):
        n = ['set', 'L']
        mfp = pd.MultiIndex.from_product
        mfa = pd.MultiIndex.from_arrays
        self.assertTrue((self.bs.primitives(False) ==
            pd.Series([1], index=mfp([[0], [0]], names=n))).all())
        self.assertTrue((self.bs.primitives(True) ==
            pd.Series([1], index=mfp([[0], [0]], names=n))).all())
        self.assertTrue((self.mbs.primitives(False) ==
            pd.Series([1, 3, 1], index=mfa([[0, 0, 1], [0, 1, 0]], names=n))).all())
        self.assertTrue((self.mbs.primitives(True) ==
            pd.Series([1, 3, 1], index=mfa([[0, 0, 1], [0, 1, 0]], names=n))).all())
        self.assertTrue((self.lbs.primitives(False) ==
            pd.Series([3, 6, 6, 2, 3], index=mfa([[0, 0, 0, 1, 1],
                                                  [0, 1, 2, 0, 1]], names=n))).all())
        self.assertTrue((self.lbs.primitives(True) ==
            pd.Series([3, 6, 5, 2, 3], index=mfa([[0, 0, 0, 1, 1],
                                                  [0, 1, 2, 0, 1]], names=n))).all())
