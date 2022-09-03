# -*- coding: utf-8 -*-
# Copyright (c) 2015-2022, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
import numpy as np
import pandas as pd
from unittest import TestCase
from exatomic.base import resource
from exatomic.adf.tape21 import Tape21, MissingSection
from exatomic.interfaces.xyz import XYZ

class TestTape21(TestCase):
    def setUp(self):
        self.formald = Tape21(resource('adf-formald-freq.t21.ascii'))
        self.m1nb = Tape21(resource('adf-m1-nb.t21.ascii'))

    def test_parse_atom(self):
        totest = ['x', 'y', 'z']
        # test input_order=True
        self.m1nb.parse_atom(input_order=True)
        self.assertEqual(self.m1nb.atom.shape[0], 36)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.m1nb.atom))))
        test = XYZ(resource('m1-nb-input-order-atom.xyz'))
        test.parse_atom()
        self.assertTrue(np.allclose(self.m1nb.atom[totest].values,
                                    test.atom[totest].values))
        self.assertTrue(np.all(self.m1nb.atom['symbol'].values == \
                                test.atom['symbol'].values))
        # test input_order=False
        self.m1nb.parse_atom(input_order=False)
        self.assertEqual(self.m1nb.atom.shape[0], 36)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.m1nb.atom))))
        test = XYZ(resource('m1-nb-adf-order-atom.xyz'))
        test.parse_atom()
        self.assertTrue(np.allclose(self.m1nb.atom[totest].values,
                                    test.atom[totest].values, atol=1e-4))
        self.assertTrue(np.all(self.m1nb.atom['symbol'].values == \
                                test.atom['symbol'].values))
        self.formald.parse_atom()
        self.assertEqual(self.formald.atom.shape[0], 4)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.formald.atom))))

    def test_parse_frequency(self):
        self.formald.parse_frequency()
        self.assertEqual(self.formald.frequency.shape[0], 24)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.formald.frequency))))

