# -*- coding: utf-8 -*-
# Copyright (c) 2015-2022, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
from unittest import TestCase
import numpy as np
import pandas as pd
import h5py
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
        self.formald = Output(resource('mol-formald-rassi.out'))
        self.formald_al = Output(resource('mol-formald-alaska.out'))
        self.form_pt2 = Output(resource('mol-formald-caspt2.out'))
        self.nichxn_ras = Output(resource('mol-nichxn3-rasscf.out'))
        self.ucl6_ras = Output(resource('mol-ucl6-rasscf.out'))

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
        self.formald_al.parse_atom(seward=False)
        self.assertEqual(self.formald_al.atom.shape[0], 32)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.formald_al.atom))))

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

    def test_parse_sf_dipole_moment(self):
        # test the parser
        self.formald.parse_sf_dipole_moment()
        self.assertEqual(self.formald.sf_dipole_moment.shape[0], 30)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.formald.sf_dipole_moment))))

    def test_parse_sf_quadrupole_moment(self):
        # test the parser
        self.formald.parse_sf_quadrupole_moment()
        self.assertEqual(self.formald.sf_quadrupole_moment.shape[0], 60)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.formald.sf_quadrupole_moment))))

    def test_parse_sf_angmom(self):
        # test the parser
        self.cdz.parse_sf_angmom()
        self.assertEqual(self.cdz.sf_angmom.shape[0], 24)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.cdz.sf_angmom))))
        # test the parser
        self.formald.parse_sf_angmom()
        self.assertEqual(self.formald.sf_angmom.shape[0], 30)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.formald.sf_angmom))))

    def test_parse_sf_oscillator(self):
        self.formald.parse_sf_oscillator()
        self.assertEqual(self.formald.sf_oscillator.shape[0], 6)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.formald.sf_oscillator))))

    def test_parse_so_oscillator(self):
        self.formald.parse_so_oscillator()
        self.assertEqual(self.formald.so_oscillator.shape[0], 6)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.formald.so_oscillator))))

    def test_parse_sf_energy(self):
        self.cdz.parse_sf_energy()
        self.assertEqual(self.cdz.sf_energy.shape[0], 8)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.cdz.sf_energy))))
        self.formald.parse_sf_energy()
        self.assertEqual(self.formald.sf_energy.shape[0], 10)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.formald.sf_energy))))
        self.uo2sp.parse_sf_energy()
        self.assertEqual(self.uo2sp.sf_energy.shape[0], 4)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.uo2sp.sf_energy))))

    def test_parse_so_energy(self):
        self.cdz.parse_so_energy()
        self.assertEqual(self.cdz.so_energy.shape[0], 24)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.cdz.so_energy))))
        self.formald.parse_so_energy()
        self.assertEqual(self.formald.so_energy.shape[0], 10)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.formald.so_energy))))

    def test_parse_frequency(self):
        self.formald_al.parse_frequency()
        self.assertEqual(self.formald_al.frequency.shape[0], 24)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.formald_al.frequency))))

    def test_parse_gradient(self):
        self.formald_al.parse_gradient()
        self.assertEqual(self.formald_al.gradient.shape[0], 32)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.formald_al.gradient))))

    def test_parse_natural_occ(self):
        # test the size of both axes as the number of orbitals is
        # specific to each case
        # simple test cases
        self.cdz.parse_natural_occ()
        self.assertEqual(self.cdz.natural_occ.shape, (15, 8))
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.cdz.natural_occ))))
        self.uo2sp.parse_natural_occ()
        self.assertEqual(self.uo2sp.natural_occ.shape, (4, 8))
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.uo2sp.natural_occ))))
        # test more than one row in a symmetry
        self.nichxn_ras.parse_natural_occ()
        self.assertEqual(self.nichxn_ras.natural_occ.shape, (12, 16))
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.nichxn_ras.natural_occ))))
        # test case for more than one symmetry
        self.ucl6_ras.parse_natural_occ()
        self.assertEqual(self.ucl6_ras.natural_occ.shape, (70, 12))
        # ensure that the values in symmetry 1 do not contain a nan
        grouped = self.ucl6_ras.natural_occ.groupby('symmetry')
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(grouped.get_group(1)))))
        # ensure that the values for the available orbitals in symmetry 2 are real
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(grouped.get_group(2)[range(7)]))))
        # ensure that the last column is a nan value as the number of columns in
        # symmetry 2 is less than 1
        self.assertFalse(np.all(pd.notnull(pd.DataFrame(grouped.get_group(2)[7]))))

    def test_parse_caspt2_energy(self):
        self.form_pt2.parse_caspt2_energy()
        self.assertEqual(self.form_pt2.caspt2_energy.shape[0], 26)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.form_pt2.caspt2_energy))))

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
