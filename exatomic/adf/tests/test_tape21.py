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
        self.nico4 = Tape21(resource('adf-nico4.t21.ascii'))
        self.pf3_one = Tape21(resource('ams-pf3-nmr-one.t21.ascii'))
        self.pf3_two = Tape21(resource('ams-pf3-nmr-two.t21.ascii'))
        self.pf3_all = Tape21(resource('ams-pf3-nmr-all.t21.ascii'))
        self.c2h2 = Tape21(resource('ams-c2h2-cpl.t21.ascii'))

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
        self.nico4.parse_atom()
        self.assertEqual(self.nico4.atom.shape[0], 9)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.nico4.atom))))

    def test_parse_frequency(self):
        self.formald.parse_frequency()
        self.assertEqual(self.formald.frequency.shape[0], 24)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.formald.frequency))))

    def test_parse_gradient(self):
        self.m1nb.parse_gradient(input_order=True)
        self.assertEqual(self.m1nb.gradient.shape[0], 36)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.m1nb.gradient))))
        self.formald.parse_gradient(input_order=False)
        self.assertEqual(self.formald.gradient.shape[0], 4)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.formald.gradient))))
        self.assertRaises(MissingSection, self.formald.parse_gradient,
                          input_order=True)

    def test_parse_nmr_shielding(self):
        # testing for only one NMR nucleus
        self.pf3_one.parse_nmr_shielding()
        self.assertEqual(self.pf3_one.nmr_shielding.shape[0], 1)
        self.assertTrue(np.all(self.pf3_one.nmr_shielding['atom'].values == [0]))
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.pf3_one.nmr_shielding))))
        # testing for only two NMR nuclei
        self.pf3_two.parse_nmr_shielding()
        self.assertEqual(self.pf3_two.nmr_shielding.shape[0], 2)
        self.assertTrue(np.all(self.pf3_two.nmr_shielding['atom'].values == [0,2]))
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.pf3_two.nmr_shielding))))
        # testing for all NMR nuclei
        self.pf3_all.parse_nmr_shielding()
        self.assertEqual(self.pf3_all.nmr_shielding.shape[0], 4)
        self.assertTrue(np.all(self.pf3_all.nmr_shielding['atom'].values == list(range(4))))
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.pf3_all.nmr_shielding))))

    def test_parse_j_coupling(self):
        self.c2h2.parse_j_coupling()
        self.assertEqual(self.c2h2.j_coupling.shape[0], 4)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.c2h2.j_coupling))))

