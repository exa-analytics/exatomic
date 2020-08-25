# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
#import os
import numpy as np
import pandas as pd
from unittest import TestCase

from exatomic import gaussian
from exatomic.base import resource
from exatomic.gaussian import Output, Fchk

class TestFchk(TestCase):
    def setUp(self):
        self.mam1 = Fchk(resource('g09-ch3nh2-631g.fchk'))
        self.mam2 = Fchk(resource('g09-ch3nh2-augccpvdz.fchk'))
        self.mam3 = Fchk(resource('g16-methyloxirane-def2tzvp-freq.fchk'))
        self.mam4 = Fchk(resource('g16-h2o2-def2tzvp-freq.fchk'))
        self.nitro_nmr = Fchk(resource('g16-nitromalonamide-6-31++g-nmr.fchk'))

    def test_parse_atom(self):
        self.mam1.parse_atom()
        self.assertEqual(self.mam1.atom.shape[0], 7)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.mam1.atom))))
        self.mam2.parse_atom()
        self.assertEqual(self.mam2.atom.shape[0], 7)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.mam2.atom))))

    def test_parse_basis_set(self):
        self.mam1.parse_basis_set()
        self.assertEqual(self.mam1.basis_set.shape[0], 32)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.mam1.basis_set))))
        self.mam2.parse_basis_set()
        self.assertEqual(self.mam2.basis_set.shape[0], 53)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.mam2.basis_set))))

    def test_parse_orbital(self):
        self.mam1.parse_orbital()
        self.assertEqual(self.mam1.orbital.shape[0], 28)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.mam1.orbital))))
        self.mam2.parse_orbital()
        self.assertEqual(self.mam2.orbital.shape[0], 91)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.mam2.orbital))))

    def test_parse_momatrix(self):
        self.mam1.parse_momatrix()
        self.assertEqual(self.mam1.momatrix.shape[0], 784)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.mam1.momatrix))))
        self.mam2.parse_momatrix()
        self.assertEqual(self.mam2.momatrix.shape[0], 8281)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.mam2.momatrix))))

    def test_parse_basis_set_order(self):
        self.mam1.parse_basis_set_order()
        self.assertEqual(self.mam1.basis_set_order.shape[0], 28)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.mam1.basis_set_order))))
        self.mam2.parse_basis_set_order()
        self.assertEqual(self.mam2.basis_set_order.shape[0], 91)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.mam2.basis_set_order))))

    def test_parse_frame(self):
        self.mam1.parse_frame()
        self.assertEqual(self.mam1.frame.shape[0], 1)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.mam1.frame))))
        self.mam2.parse_frame()
        self.assertEqual(self.mam2.frame.shape[0], 1)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.mam2.frame))))

    def test_parse_frequency(self):
        self.mam3.parse_frequency()
        self.assertEqual(self.mam3.frequency.shape[0], 240)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.mam3.frequency))))
        self.mam4.parse_frequency()
        self.assertEqual(self.mam4.frequency.shape[0], 24)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.mam4.frequency))))

    def test_parse_frequency_ext(self):
        self.mam3.parse_frequency_ext()
        self.assertEqual(self.mam3.frequency_ext.shape[0], 24)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.mam3.frequency_ext))))
        self.mam4.parse_frequency_ext()
        self.assertEqual(self.mam4.frequency_ext.shape[0], 6)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.mam4.frequency_ext))))

    def test_parse_gradient(self):
        self.mam3.parse_gradient()
        self.assertEqual(self.mam3.gradient.shape[0], 10)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.mam3.gradient))))
        self.mam4.parse_gradient()
        self.assertEqual(self.mam4.gradient.shape[0], 4)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.mam4.gradient))))

    def test_shielding_tensor(self):
        self.nitro_nmr.parse_nmr_shielding()
        self.assertEqual(self.nitro_nmr.nmr_shielding.shape[0], 15)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.nitro_nmr.nmr_shielding))))

    def test_to_universe(self):
        """Test the to_universe method."""
        mam1 = self.mam1.to_universe(ignore=True)
        mam2 = self.mam2.to_universe(ignore=True)
        for uni in [mam1, mam2]:
            # cannot add frequency and frequency_ext attributes as they require
            # very specific inputs
            for attr in ['atom', 'basis_set', 'basis_set_order',
                         'momatrix', 'orbital', 'frame']:
                self.assertTrue(hasattr(uni, attr))

class TestOutput(TestCase):
    """
    This test ensures that the parsing functionality works on
    a smattering of output files that were generated with the
    Gaussian software package. Target syntax is for Gaussian
    09.
    """
    def setUp(self):
        # TODO : add some cartesian basis set files
        self.uo2 = Output(resource('g09-uo2.out'))
        self.mam3 = Output(resource('g09-ch3nh2-631g.out'))
        self.mam4 = Output(resource('g09-ch3nh2-augccpvdz.out'))
        # need two because of the current limitations in the parse_frequency code
        self.meth_opt = Output(resource('g16-methyloxirane-def2tzvp-opt.out'))
        self.meth_freq = Output(resource('g16-methyloxirane-def2tzvp-freq.out'))
        self.nap_tddft = Output(resource('g16-naproxen-def2tzvp-tddft.out'))
        self.h2o2_tddft = Output(resource('g16-h2o2-def2tzvp-tddft.out'))
        self.nap_opt = Output(resource('g16-naproxen-def2tzvp-opt.out'))
        self.nitro_nmr = Output(resource('g16-nitromalonamide-6-31++g-nmr.out'))
        # to test having both a geometry optimization and frequencies calculation
        self.meth_opt_freq_hp = Output(resource('g16-methyloxirane-def2tzvp-opt-freq.out'))

    def test_parse_atom(self):
        self.uo2.parse_atom()
        self.assertEqual(self.uo2.atom.shape[0], 3)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.uo2.atom))))
        self.mam3.parse_atom()
        self.assertEqual(self.mam3.atom.shape[0], 7)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.mam3.atom))))
        self.mam4.parse_atom()
        self.assertEqual(self.mam4.atom.shape[0], 7)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.mam4.atom))))
        self.meth_opt.parse_atom()
        self.assertEqual(self.meth_opt.atom.shape[0], 120)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.meth_opt.atom))))
        self.nap_opt.parse_atom()
        self.assertEqual(self.nap_opt.atom.shape[0], 806)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.nap_opt.atom))))
        self.meth_opt_freq_hp.parse_atom()
        self.assertEqual(self.meth_opt_freq_hp.atom.shape[0], 130)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.meth_opt_freq_hp.atom))))

    def test_parse_basis_set(self):
        self.uo2.parse_basis_set()
        self.assertEqual(self.uo2.basis_set.shape[0], 49)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.uo2.basis_set))))
        self.mam3.parse_basis_set()
        self.assertEqual(self.mam3.basis_set.shape[0], 32)
        cols = list(set(self.mam3.basis_set._columns))
        test = pd.DataFrame(self.mam3.basis_set[cols])
        self.assertTrue(np.all(pd.notnull(test)))
        self.mam4.parse_basis_set()
        self.assertEqual(self.mam4.basis_set.shape[0], 53)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.mam4.basis_set))))

    def test_parse_orbital(self):
        self.uo2.parse_orbital()
        self.assertEqual(self.uo2.orbital.shape[0], 141)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.uo2.orbital))))
        self.mam3.parse_orbital()
        self.assertEqual(self.mam3.orbital.shape[0], 28)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.mam3.orbital))))
        self.mam4.parse_orbital()
        self.assertEqual(self.mam4.orbital.shape[0], 91)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.mam4.orbital))))
        self.meth_opt.parse_orbital()
        self.assertEqual(self.meth_opt.orbital.shape[0], 160)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.meth_opt.orbital))))
        self.nap_tddft.parse_orbital()
        self.assertEqual(self.nap_tddft.orbital.shape[0], 611)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.nap_tddft.orbital))))

    def test_parse_momatrix(self):
        self.uo2.parse_momatrix()
        self.assertEqual(self.uo2.momatrix.shape[0], 19881)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.uo2.momatrix))))
        self.mam3.parse_momatrix()
        self.assertEqual(self.mam3.momatrix.shape[0], 784)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.mam3.momatrix))))
        self.mam4.parse_momatrix()
        self.assertEqual(self.mam4.momatrix.shape[0], 8281)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.mam4.momatrix))))

    def test_parse_basis_set_order(self):
        self.uo2.parse_basis_set_order()
        self.assertEqual(self.uo2.basis_set_order.shape[0], 141)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.uo2.basis_set_order))))
        self.mam3.parse_basis_set_order()
        self.assertEqual(self.mam3.basis_set_order.shape[0], 28)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.mam3.basis_set_order))))
        self.mam4.parse_basis_set_order()
        self.assertEqual(self.mam4.basis_set_order.shape[0], 91)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.mam4.basis_set_order))))

    def test_parse_frame(self):
        self.uo2.parse_frame()
        self.assertEqual(self.uo2.frame.shape[0], 1)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.uo2.frame))))
        self.mam3.parse_frame()
        self.assertEqual(self.mam3.frame.shape[0], 1)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.mam3.frame))))
        self.mam4.parse_frame()
        self.assertEqual(self.mam4.frame.shape[0], 1)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.mam4.frame))))
        self.meth_opt.parse_frame()
        self.assertEqual(self.meth_opt.frame.shape[0], 12)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.meth_opt.frame))))
        self.nap_opt.parse_frame()
        self.assertEqual(self.nap_opt.frame.shape[0], 26)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.nap_opt.frame))))

    def test_parse_frequency_ext(self):
        self.meth_freq.parse_frequency_ext()
        self.assertEqual(self.meth_freq.frequency_ext.shape[0], 24)
        self.assertEqual(self.meth_freq.frequency_ext.shape[1], 5)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.meth_freq.frequency_ext))))
        self.meth_opt_freq_hp.parse_frequency_ext()
        self.assertEqual(self.meth_opt_freq_hp.frequency_ext.shape[0], 24)
        self.assertEqual(self.meth_opt_freq_hp.frequency_ext.shape[1], 5)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.meth_opt_freq_hp.frequency_ext))))

    def test_parse_frequency(self):
        self.meth_freq.parse_frequency()
        self.assertEqual(self.meth_freq.frequency.shape[0], 240)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.meth_freq.frequency))))
        self.meth_opt_freq_hp.parse_frequency()
        self.assertEqual(self.meth_opt_freq_hp.frequency.shape[0], 240)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.meth_opt_freq_hp.frequency))))

    def test_parse_excitation(self):
        self.nap_tddft.parse_excitation()
        self.assertEqual(self.nap_tddft.excitation.shape[0], 15)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.nap_tddft.excitation))))
        self.h2o2_tddft.parse_excitation()
        self.assertEqual(self.h2o2_tddft.excitation.shape[0], 32)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.h2o2_tddft.excitation))))

    def test_shielding_tensor(self):
        self.nitro_nmr.parse_nmr_shielding()
        self.assertEqual(self.nitro_nmr.nmr_shielding.shape[0], 15)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.nitro_nmr.nmr_shielding))))

    def test_parse_gradient(self):
        self.meth_opt.parse_gradient()
        self.assertEqual(self.meth_opt.gradient.shape[0], 120)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.meth_opt.gradient))))
        self.nap_opt.parse_gradient()
        self.assertEqual(self.nap_opt.gradient.shape[0], 806)
        self.assertTrue(np.all(pd.notnull(pd.DataFrame(self.nap_opt.gradient))))

    def test_to_universe(self):
        """Test the to_universe method."""
        uo2 = self.uo2.to_universe(ignore=True)
        mam3 = self.mam3.to_universe(ignore=True)
        #meth_opt = self.meth_opt.to_universe(ignore=True)
        for uni in [uo2, mam3]:
            for attr in ['atom', 'basis_set', 'basis_set_order',
                         'momatrix', 'orbital', 'frame']:
                self.assertTrue(hasattr(uni, attr))
