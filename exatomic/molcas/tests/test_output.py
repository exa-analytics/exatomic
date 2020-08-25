# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
from unittest import TestCase
import h5py
import numpy as np
import pandas as pd
from exatomic import Universe
from exatomic.base import resource
from exatomic.molcas.output import Output, Orb, HDF

# TODO : change df.shape[0] == num to len(df.index) == num everywhere

class TestOutput(TestCase):
    """Test the Molcas output file editor."""
    def setUp(self):
        self.cdz = Output(resource('mol-carbon-dz.out'))
        self.uo2sp = Output(resource('mol-uo2-anomb.out'))
        self.mamcart = Output(resource('mol-ch3nh2-631g.out'))
        self.mamsphr = Output(resource('mol-ch3nh2-anovdzp.out'))
        self.c2h6 = Output(resource('mol-c2h6-basis.out'))

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
        uni = self.mamcart.to_universe()
        self.assertTrue(hasattr(uni, 'momatrix'))
        self.assertTrue(hasattr(uni, 'orbital'))


    def test_add_overlap(self):
        """Test adding an overlap matrix."""
        self.cdz.add_overlap(resource('mol-carbon-dz.overlap'))
        self.assertTrue(hasattr(self.cdz, 'overlap'))
        uni = self.cdz.to_universe()
        self.assertTrue(hasattr(uni, 'overlap'))


    def test_parse_atom(self):
        """Test the atom table parser."""
        self.uo2sp.parse_atom()
        self.assertEqual(self.uo2sp.atom.shape[0], 3)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.uo2sp.atom))))
        self.mamcart.parse_atom()
        self.assertEqual(self.mamcart.atom.shape[0], 7)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.mamcart.atom))))
        self.mamsphr.parse_atom()
        self.assertEqual(self.mamsphr.atom.shape[0], 7)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.mamsphr.atom))))

    def test_parse_basis_set_order(self):
        """Test the basis set order table parser."""
        self.uo2sp.parse_basis_set_order()
        self.assertEqual(self.uo2sp.basis_set_order.shape[0], 69)
        cols = list(set(self.uo2sp.basis_set_order._columns))
        test = pd.DataFrame(self.uo2sp.basis_set_order[cols])
        self.assertTrue(np.all(pd.notnull(test)))
        self.mamcart.parse_basis_set_order()
        self.assertEqual(self.mamcart.basis_set_order.shape[0], 28)
        cols = list(set(self.mamcart.basis_set_order._columns))
        test = pd.DataFrame(self.mamcart.basis_set_order[cols])
        self.assertTrue(np.all(pd.notnull(test)))
        self.mamsphr.parse_basis_set_order()
        self.assertEqual(self.mamsphr.basis_set_order.shape[0], 53)
        cols = list(set(self.mamsphr.basis_set_order._columns))
        test = pd.DataFrame(self.mamsphr.basis_set_order[cols])
        self.assertTrue(np.all(pd.notnull(test)))

    def test_parse_basis_set(self):
        """Test the gaussian basis set table parser."""
        self.uo2sp.parse_basis_set()
        self.assertEqual(self.uo2sp.basis_set.shape[0], 451)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.uo2sp.basis_set))))
        self.mamcart.parse_basis_set()
        self.assertEqual(self.mamcart.basis_set.shape[0], 84)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.mamcart.basis_set))))
        self.mamsphr.parse_basis_set()
        self.assertEqual(self.mamsphr.basis_set.shape[0], 148)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.mamsphr.basis_set))))
        self.c2h6.parse_basis_set()
        self.assertTrue(hasattr(self.c2h6, 'basis_set'))

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

    def test_parse_old_uhf(self):
        sym = Orb(resource('mol-c2h6-old-sym.uhforb'))
        nym = Orb(resource('mol-c2h6-old-nosym.uhforb'))
        sym.parse_momatrix()
        nym.parse_momatrix()
        self.assertTrue(sym.momatrix.shape[0] == 274)
        self.assertTrue(nym.momatrix.shape[0] == 900)

    def test_parse_old_orb(self):
        sym = Orb(resource('mol-c2h6-old-sym.scforb'))
        nym = Orb(resource('mol-c2h6-old-nosym.scforb'))
        sym.parse_momatrix()
        nym.parse_momatrix()
        self.assertTrue(sym.momatrix.shape[0] == 274)
        self.assertTrue(nym.momatrix.shape[0] == 900)

    def test_parse_uhf(self):
        sym = Orb(resource('mol-c2h6-sym.uhforb'))
        nym = Orb(resource('mol-c2h6-nosym.uhforb'))
        sym.parse_momatrix()
        nym.parse_momatrix()
        self.assertTrue(sym.momatrix.shape[0] == 274)
        self.assertTrue(nym.momatrix.shape[0] == 900)

    def test_parse_orb(self):
        sym = Orb(resource('mol-c2h6-sym.scforb'))
        nym = Orb(resource('mol-c2h6-nosym.scforb'))
        sym.parse_momatrix()
        nym.parse_momatrix()
        self.assertTrue(sym.momatrix.shape[0] == 274)
        self.assertTrue(nym.momatrix.shape[0] == 900)


    def test_parse_momatrix(self):
        """Test the momatrix table parser."""
        uo2sp = Orb(resource('mol-uo2-anomb.scforb'))
        uo2sp.parse_momatrix()
        self.assertEqual(uo2sp.momatrix.shape[0], 4761)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(uo2sp.momatrix))))
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(uo2sp.orbital))))
        mamcart = Orb(resource('mol-ch3nh2-631g.scforb'))
        mamcart.parse_momatrix()
        self.assertEqual(mamcart.momatrix.shape[0], 784)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(mamcart.momatrix))))
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(mamcart.orbital))))
        mamsphr = Orb(resource('mol-ch3nh2-anovdzp.scforb'))
        mamsphr.parse_momatrix()
        self.assertEqual(mamsphr.momatrix.shape[0], 2809)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(mamsphr.momatrix))))
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(mamsphr.orbital))))


class TestHDF(TestCase):

    def setUp(self):
        self.nym = HDF(resource('mol-c2h6-nosym-scf.hdf5'))
        self.sym = HDF(resource('mol-c2h6-sym-scf.hdf5'))

    def test_parse_atom(self):
        self.sym.parse_atom()
        self.nym.parse_atom()
        self.assertTrue(self.sym.atom.shape[0] == 8)
        self.assertTrue(self.nym.atom.shape[0] == 8)

    def test_parse_basis_set_order(self):
        self.sym.parse_basis_set_order()
        self.nym.parse_basis_set_order()
        self.assertTrue(self.sym.basis_set_order.shape[0] == 30)
        self.assertTrue(self.nym.basis_set_order.shape[0] == 30)

    def test_parse_orbital(self):
        self.sym.parse_orbital()
        self.nym.parse_orbital()
        self.assertTrue(self.sym.orbital.shape[0] == 30)
        self.assertTrue(self.nym.orbital.shape[0] == 30)

    def test_parse_overlap(self):
        self.sym.parse_overlap()
        self.nym.parse_overlap()
        self.assertTrue(self.sym.overlap.shape[0])
        self.assertTrue(self.nym.overlap.shape[0])

    def test_parse_momatrix(self):
        self.sym.parse_momatrix()
        self.nym.parse_momatrix()
        self.assertTrue(self.nym.momatrix.shape[0] == 900)
        with self.assertRaises(AttributeError):
            self.assertTrue(self.sym.momatrix)

    def test_to_universe(self):
        self.sym.to_universe()
        self.nym.to_universe()
