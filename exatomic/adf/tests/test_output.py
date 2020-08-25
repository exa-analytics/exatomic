# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
import numpy as np
import pandas as pd
from unittest import TestCase
from exatomic.base import resource
from exatomic.adf.output import Output


class TestADFOutput(TestCase):
    """Test the ADF output file editor."""

    def setUp(self):
        self.lu = Output(resource('adf-lu.out'))
        # TODO :: File with excitation
        self.pf3 = Output(resource('adf-pf3-nmr.out'))
        self.c2h2 = Output(resource('adf-c2h2-cpl.out'))
        self.ch4 = Output(resource('adf-ch4-opt-freq.out'))
        self.c2h2_nofrag = Output(resource('adf-c2h2-cpl-nofrag.out'))
        self.c2h3i = Output(resource('adf-c2h3i-opt.out'))
        self.nico4 = Output(resource('adf-nico4.out'))

    def test_parse_atom(self):
        self.lu.parse_atom()
        self.assertEqual(self.lu.atom.shape[0], 1)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.lu.atom))))
        self.pf3.parse_atom()
        self.assertEqual(self.pf3.atom.shape[0], 4)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.pf3.atom))))
        self.c2h2.parse_atom()
        self.assertEqual(self.c2h2.atom.shape[0], 4)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.c2h2.atom))))
        self.c2h2_nofrag.parse_atom()
        self.assertEqual(self.c2h2.atom.shape[0], 4)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.c2h2.atom))))
        self.c2h3i.parse_atom()
        self.assertEqual(self.c2h3i.atom.shape[0], 66)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.c2h3i.atom))))
        self.nico4.parse_atom()
        self.assertEqual(self.nico4.atom.shape[0], 9)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.nico4.atom))))
        self.ch4.parse_atom()
        self.assertEqual(self.ch4.atom.shape[0], 15)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.ch4.atom))))

    def test_parse_basis_set(self):
        self.lu.parse_basis_set()
        self.assertEqual(self.lu.basis_set.shape[0], 32)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.lu.basis_set))))

    def test_parse_basis_set_order(self):
        self.lu.parse_basis_set_order()
        self.assertEqual(self.lu.basis_set_order.shape[0], 109)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.lu.basis_set_order))))

    def test_parse_momatrix_and_to_universe(self):
        self.lu.parse_momatrix()
        uni = self.lu.to_universe()
        self.assertEqual(self.lu.momatrix.shape[0],
                         uni.basis_dims['ncc'] *
                         uni.basis_dims['ncs'])

    def test_parse_contribution(self):
        self.lu.parse_contribution()
        self.assertEqual(self.lu.contribution.shape[0], 78)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.lu.contribution))))

    def test_parse_orbital(self):
        self.lu.parse_orbital()
        self.assertEqual(self.lu.orbital.shape[0], 20)
        cols = list(set(self.lu.orbital._columns))
        test = pd.DataFrame(self.lu.orbital[cols])
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(test))))
        self.nico4.parse_orbital()
        self.assertEqual(self.nico4.orbital[cols].shape[0], 40)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.nico4.orbital[cols]))))

    def test_parse_nmr_shielding(self):
        self.pf3.parse_nmr_shielding()
        self.assertEqual(self.pf3.nmr_shielding.shape[0], 4)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(pd.DataFrame(self.pf3.nmr_shielding)))))

    def test_parse_j_coupling(self):
        self.c2h2.parse_j_coupling()
        self.assertEqual(self.c2h2.j_coupling.shape[0], 4)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.c2h2.j_coupling))))

    def test_parse_frequency(self):
        self.ch4.parse_frequency()
        self.assertEqual(self.ch4.frequency.shape[0], 45)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.ch4.frequency))))

    def test_parse_gradient(self):
        self.ch4.parse_gradient()
        self.assertEqual(self.ch4.gradient.shape[0], 15)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.ch4.gradient))))
        self.c2h3i.parse_gradient()
        self.assertEqual(self.c2h3i.gradient.shape[0], 66)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.c2h3i.gradient))))
