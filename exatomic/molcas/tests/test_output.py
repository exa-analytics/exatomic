# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
import os
import numpy as np
import pandas as pd
from unittest import TestCase
from exatomic import Universe
from exatomic.molcas.output import Output, Orb


class TestOutput(TestCase):
    """Test the Molcas output file editor."""
    def setUp(self):
        cd = os.path.abspath(__file__).split(os.sep)[:-1]
        self.uo2sp = Output(os.sep.join(cd + ['mol-uo2-anomb.out']))
        self.mamcart = Output(os.sep.join(cd + ['mol-ch3nh2-631g.out']))
        self.mamsphr = Output(os.sep.join(cd + ['mol-ch3nh2-anovdzp.out']))

    def test_parse_atom(self):
        """Test the atom table parser."""
        self.uo2sp.parse_atom()
        self.assertEqual(self.uo2sp.atom.shape, (3, 8))
        self.assertTrue(np.all(pd.notnull(self.uo2sp.atom)))
        self.mamcart.parse_atom()
        self.assertEqual(self.mamcart.atom.shape, (7, 8))
        self.assertTrue(np.all(pd.notnull(self.mamcart.atom)))
        self.mamsphr.parse_atom()
        self.assertEqual(self.mamsphr.atom.shape, (7, 8))
        self.assertTrue(np.all(pd.notnull(self.mamsphr.atom)))

    def test_parse_basis_set_order(self):
        """Test the basis set order table parser."""
        self.uo2sp.parse_basis_set_order()
        self.assertEqual(self.uo2sp.basis_set_order.shape, (69, 9))
        self.assertTrue(np.all(pd.notnull(self.uo2sp.basis_set_order)))
        self.mamcart.parse_basis_set_order()
        self.assertEqual(self.mamcart.basis_set_order.shape, (28, 9))
        self.assertTrue(np.all(pd.notnull(self.mamcart.basis_set_order)))
        self.mamsphr.parse_basis_set_order()
        self.assertEqual(self.mamsphr.basis_set_order.shape, (53, 9))
        self.assertTrue(np.all(pd.notnull(self.mamsphr.basis_set_order)))

    def test_parse_basis_set(self):
        """Test the gaussian basis set table parser."""
        self.uo2sp.parse_basis_set()
        self.assertEqual(self.uo2sp.basis_set.shape, (451, 8))
        self.assertTrue(np.all(pd.notnull(self.uo2sp.basis_set)))
        self.mamcart.parse_basis_set()
        self.assertEqual(self.mamcart.basis_set.shape, (84, 8))
        self.assertTrue(np.all(pd.notnull(self.mamcart.basis_set)))
        self.mamsphr.parse_basis_set()
        self.assertEqual(self.mamsphr.basis_set.shape, (148, 8))
        self.assertTrue(np.all(pd.notnull(self.mamsphr.basis_set)))

    def test_to_universe(self):
        """Test that the Outputs can be converted to universes."""
        uni = self.uo2sp.to_universe()
        self.assertIs(type(uni), Universe)
        uni = self.mamcart.to_universe()
        self.assertIs(type(uni), Universe)
        uni = self.mamsphr.to_universe()
        self.assertIs(type(uni), Universe)


# class TestOrb(TestCase):
#     """Test the Molcas Orb file parser."""
#     def setUp(self):
#         cd = os.path.abspath(__file__).split(os.sep)[:-1]
#         self.uo2sporb = Orb(os.sep.join(cd + ['mol-uo2-anomb.scforb']))
#         self.mamcart = Orb(os.sep.join(cd + ['mol-ch3nh2-631g.scforb']))
#         self.mamsphr = Orb(os.sep.join(cd + ['mol-ch3nh2-anovdzp.scforb']))
#
#     def test_parse_momatrix(self):
#         """Test the momatrix table parser."""
#         self.uo2sporb.parse_momatrix()
#         self.assertEqual(self.uo2sporb.momatrix.shape, (4761, 4))
#         self.assertTrue(np.all(pd.notnull(self.uo2sporb.momatrix)))
#         self.mamcart.parse_momatrix()
#         self.assertEqual(self.mamcart.momatrix.shape, (784, 4))
#         self.assertTrue(np.all(pd.notnull(self.mamcart.momatrix)))
#         self.mamsphr.parse_momatrix()
#         self.assertEqual(self.mamsphr.momatrix.shape, (2809, 4))
#         self.assertTrue(np.all(pd.notnull(self.mamsphr.momatrix)))
