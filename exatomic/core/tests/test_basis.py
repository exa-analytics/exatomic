# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
import numpy as np
import pandas as pd
from unittest import TestCase
from exatomic.core.basis import BasisSet

class TestBasisSet(TestCase):

    def setUp(self):
        adict = {col: [0] for col in BasisSet._columns}
        adict['frame'] = 0
        # Trivial basis set
        self.bs = BasisSet(adict)
        self.bs['alpha'] = self.bs['alpha'].astype(np.float64)
        self.bs['d'] = self.bs['d'].astype(np.float64)
        # Medium basis set
        self.mbs = BasisSet({'frame': 0,
                             'alpha': [5., 1., 1.],
                                 'd': [1., 1., 1.],
                             'shell': [0, 1, 0],
                               'set': [0, 0, 1],
                                 'L': [0, 1, 0],
                                 'n': [1, 2, 1]})
        # Large basis set
        self.lbs = BasisSet({'frame': 0,
                             'alpha': [5., 3., 1., 3., 1., 1., 3., 1., 1.],
                                 'd': [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                             'shell': [0, 0, 0, 1, 1, 2, 0, 0, 1],
                               'set': [0, 0, 0, 0, 0, 0, 1, 1, 1],
                                 'L': [0, 0, 0, 1, 1, 2, 0, 0, 1]})

    def test_lmax(self):
        self.assertEqual(self.bs.lmax, 0)
        self.assertEqual(self.mbs.lmax, 1)
        self.assertEqual(self.lbs.lmax, 2)

    def test_shells(self):
        self.bs.shells()
        self.mbs.shells()
        self.lbs.shells()

    def test_functions_by_shell(self):
        n = ['set', 'L']
        mfp = pd.MultiIndex.from_product
        mfa = pd.MultiIndex.from_arrays
        self.assertTrue((self.bs.functions_by_shell() ==
            pd.Series([1], index=mfp([[0], [0]], names=n))).all())
        s = self.mbs.functions_by_shell()
        self.assertTrue((s[s != 0] ==
            pd.Series([1, 1, 1], index=mfa([[0, 0, 1], [0, 1, 0]], names=n))).all())
        s = self.lbs.functions_by_shell()
        self.assertTrue((s[s != 0] ==
            pd.Series([1, 1, 1, 1, 1], index=mfa([[0, 0, 0, 1, 1],
                                                  [0, 1, 2, 0, 1]], names=n))).all())

    def test_primitives_by_shell(self):
        n = ['set', 'L']
        mfp = pd.MultiIndex.from_product
        mfa = pd.MultiIndex.from_arrays
        self.assertTrue((self.bs.primitives_by_shell() ==
            pd.Series([1], index=mfp([[0], [0]], names=n))).all())
        s = self.mbs.primitives_by_shell()
        self.assertTrue((s[s != 0] ==
            pd.Series([1, 1, 1], index=mfa([[0, 0, 1], [0, 1, 0]], names=n))).all())
        s = self.lbs.primitives_by_shell()
        self.assertTrue((s[s != 0] ==
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
