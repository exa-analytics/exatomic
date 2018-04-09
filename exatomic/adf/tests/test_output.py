# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
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

    def test_parse_atom(self):
        self.lu.parse_atom()
        self.assertEqual(self.lu.atom.shape[0], 1)
        self.assertTrue(np.all(pd.notnull(self.lu.atom)))

    def test_parse_basis_set(self):
        self.lu.parse_basis_set()
        self.assertEqual(self.lu.basis_set.shape[0], 32)
        self.assertTrue(np.all(pd.notnull(self.lu.basis_set)))

    def test_parse_basis_set_order(self):
        self.lu.parse_basis_set_order()
        self.assertEqual(self.lu.basis_set_order.shape[0], 109)
        self.assertTrue(np.all(pd.notnull(self.lu.basis_set_order)))

    def test_parse_momatrix_and_to_universe(self):
        self.lu.parse_momatrix()
        uni = self.lu.to_universe()
        self.assertEqual(self.lu.momatrix.shape[0],
                         uni.basis_dims['ncc'] *
                         uni.basis_dims['ncs'])

    def test_parse_contribution(self):
        self.lu.parse_contribution()
        self.assertEqual(self.lu.contribution.shape[0], 78)
        self.assertTrue(np.all(pd.notnull(self.lu.contribution)))

    def test_parse_orbital(self):
        self.lu.parse_orbital()
        self.assertEqual(self.lu.orbital.shape[0], 20)
        self.assertTrue(np.all(pd.notnull(self.lu.orbital)))
