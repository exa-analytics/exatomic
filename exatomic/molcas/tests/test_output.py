# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
import os
import numpy as np
import pandas as pd
from unittest import TestCase
from exatomic import Universe
from exatomic.base import resource
from exatomic.molcas.output import Output, Orb


class TestOutput(TestCase):
    """Test the Molcas output file editor."""
    def setUp(self):
        #self.cdz = Output(resource('mol-carbon-dz.out'))
        self.uo2sp = Output(resource('mol-uo2-anomb.out'))
        self.mamcart = Output(resource('mol-ch3nh2-631g.out'))
        self.mamsphr = Output(resource('mol-ch3nh2-anovdzp.out'))

    def test_add_orb(self):
        """Test adding orbital file functionality."""
        self.mamcart.add_orb(resource('mol-ch3nh2-631g.scforb'))
        self.assertTrue(hasattr(self.mamcart, 'momatrix'))
        self.assertTrue(hasattr(self.mamcart, 'orbital'))
        with self.assertRaises(ValueError):
            self.mamcart.add_orb(resource('mol-ch3nh2-631g.scforb'))
        self.mamcart.add_orb(resource('mol-ch3nh2-631g.scforb'),
                             mocoefs='same')
        self.assertTrue('same' in self.mamcart.momatrix.columns)
        self.assertTrue('same' in self.mamcart.orbital.columns)
        self.mamcart.add_orb(resource('mol-ch3nh2-631g.scforb'),
                             mocoefs='diff', orbocc='diffocc')
        self.assertTrue('diff' in self.mamcart.momatrix.columns)
        self.assertTrue('diffocc' in self.mamcart.orbital.columns)


    def test_add_overlap(self):
        """Test adding an overlap matrix."""
        pass


    def test_parse_atom(self):
        """Test the atom table parser."""
        self.uo2sp.parse_atom()
        self.assertEqual(self.uo2sp.atom.shape[0], 3)
        self.assertTrue(np.all(pd.notnull(self.uo2sp.atom)))
        self.mamcart.parse_atom()
        self.assertEqual(self.mamcart.atom.shape[0], 7)
        self.assertTrue(np.all(pd.notnull(self.mamcart.atom)))
        self.mamsphr.parse_atom()
        self.assertEqual(self.mamsphr.atom.shape[0], 7)
        self.assertTrue(np.all(pd.notnull(self.mamsphr.atom)))

    def test_parse_basis_set_order(self):
        """Test the basis set order table parser."""
        self.uo2sp.parse_basis_set_order()
        self.assertEqual(self.uo2sp.basis_set_order.shape[0], 69)
        self.assertTrue(np.all(pd.notnull(self.uo2sp.basis_set_order)))
        self.mamcart.parse_basis_set_order()
        self.assertEqual(self.mamcart.basis_set_order.shape[0], 28)
        self.assertTrue(np.all(pd.notnull(self.mamcart.basis_set_order)))
        self.mamsphr.parse_basis_set_order()
        self.assertEqual(self.mamsphr.basis_set_order.shape[0], 53)
        self.assertTrue(np.all(pd.notnull(self.mamsphr.basis_set_order)))

    def test_parse_basis_set(self):
        """Test the gaussian basis set table parser."""
        self.uo2sp.parse_basis_set()
        self.assertEqual(self.uo2sp.basis_set.shape[0], 451)
        self.assertTrue(np.all(pd.notnull(self.uo2sp.basis_set)))
        self.mamcart.parse_basis_set()
        self.assertEqual(self.mamcart.basis_set.shape[0], 84)
        self.assertTrue(np.all(pd.notnull(self.mamcart.basis_set)))
        self.mamsphr.parse_basis_set()
        self.assertEqual(self.mamsphr.basis_set.shape[0], 148)
        self.assertTrue(np.all(pd.notnull(self.mamsphr.basis_set)))

    def test_to_universe(self):
        """Test that the Outputs can be converted to universes."""
        uni = self.uo2sp.to_universe()
        self.assertIs(type(uni), Universe)
        uni = self.mamcart.to_universe()
        self.assertIs(type(uni), Universe)
        uni = self.mamsphr.to_universe()
        self.assertIs(type(uni), Universe)


class TestOrb(TestCase):
    """Test the Molcas Orb file parser."""

    def test_parse_momatrix(self):
        """Test the momatrix table parser."""
        uo2sp = Orb(resource('mol-uo2-anomb.scforb'))
        uo2sp.parse_momatrix()
        self.assertEqual(uo2sp.momatrix.shape[0], 4761)
        self.assertTrue(np.all(pd.notnull(uo2sp.momatrix)))
        self.assertTrue(np.all(pd.notnull(uo2sp.orbital)))
        mamcart = Orb(resource('mol-ch3nh2-631g.scforb'))
        mamcart.parse_momatrix()
        self.assertEqual(mamcart.momatrix.shape[0], 784)
        self.assertTrue(np.all(pd.notnull(mamcart.momatrix)))
        self.assertTrue(np.all(pd.notnull(mamcart.orbital)))
        mamsphr = Orb(resource('mol-ch3nh2-anovdzp.scforb'))
        mamsphr.parse_momatrix()
        self.assertEqual(mamsphr.momatrix.shape[0], 2809)
        self.assertTrue(np.all(pd.notnull(mamsphr.momatrix)))
        self.assertTrue(np.all(pd.notnull(mamsphr.orbital)))
