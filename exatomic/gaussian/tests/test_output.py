# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
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

    def test_parse_atom(self):
        self.mam1.parse_atom()
        self.assertEqual(self.mam1.atom.shape[0], 7)
        self.assertTrue(np.all(pd.notnull(self.mam1.atom)))
        self.mam2.parse_atom()
        self.assertEqual(self.mam2.atom.shape[0], 7)
        self.assertTrue(np.all(pd.notnull(self.mam2.atom)))

    def test_parse_basis_set(self):
        self.mam1.parse_basis_set()
        self.assertEqual(self.mam1.basis_set.shape[0], 32)
        self.assertTrue(np.all(pd.notnull(self.mam1.basis_set)))
        self.mam2.parse_basis_set()
        self.assertEqual(self.mam2.basis_set.shape[0], 53)
        self.assertTrue(np.all(pd.notnull(self.mam2.basis_set)))

    def test_parse_orbital(self):
        self.mam1.parse_orbital()
        self.assertEqual(self.mam1.orbital.shape[0], 28)
        self.assertTrue(np.all(pd.notnull(self.mam1.orbital)))
        self.mam2.parse_orbital()
        self.assertEqual(self.mam2.orbital.shape[0], 91)
        self.assertTrue(np.all(pd.notnull(self.mam2.orbital)))

    def test_parse_momatrix(self):
        self.mam1.parse_momatrix()
        self.assertEqual(self.mam1.momatrix.shape[0], 784)
        self.assertTrue(np.all(pd.notnull(self.mam1.momatrix)))
        self.mam2.parse_momatrix()
        self.assertEqual(self.mam2.momatrix.shape[0], 8281)
        self.assertTrue(np.all(pd.notnull(self.mam2.momatrix)))

    def test_parse_basis_set_order(self):
        self.mam1.parse_basis_set_order()
        self.assertEqual(self.mam1.basis_set_order.shape[0], 28)
        self.assertTrue(np.all(pd.notnull(self.mam1.basis_set_order)))
        self.mam2.parse_basis_set_order()
        self.assertEqual(self.mam2.basis_set_order.shape[0], 91)
        self.assertTrue(np.all(pd.notnull(self.mam2.basis_set_order)))

    def test_parse_frame(self):
        self.mam1.parse_frame()
        self.assertEqual(self.mam1.frame.shape[0], 1)
        self.assertTrue(np.all(pd.notnull(self.mam1.frame)))
        self.mam2.parse_frame()
        self.assertEqual(self.mam2.frame.shape[0], 1)
        self.assertTrue(np.all(pd.notnull(self.mam2.frame)))

    def test_to_universe(self):
        """Test the to_universe method."""
        mam1 = self.mam1.to_universe(ignore=True)
        mam2 = self.mam2.to_universe(ignore=True)
        for uni in [mam1, mam2]:
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
        #        a geometry optimization and
        #        maybe properties? like the frequency
        #        and tddft calcs
        self.uo2 = Output(resource('g09-uo2.out'))
        self.mam3 = Output(resource('g09-ch3nh2-631g.out'))
        self.mam4 = Output(resource('g09-ch3nh2-augccpvdz.out'))

    def test_parse_atom(self):
        self.uo2.parse_atom()
        self.assertEqual(self.uo2.atom.shape[0], 3)
        self.assertTrue(np.all(pd.notnull(self.uo2.atom)))
        self.mam3.parse_atom()
        self.assertEqual(self.mam3.atom.shape[0], 7)
        self.assertTrue(np.all(pd.notnull(self.mam3.atom)))
        self.mam4.parse_atom()
        self.assertEqual(self.mam4.atom.shape[0], 7)
        self.assertTrue(np.all(pd.notnull(self.mam4.atom)))

    def test_parse_basis_set(self):
        self.uo2.parse_basis_set()
        self.assertEqual(self.uo2.basis_set.shape[0], 49)
        self.assertTrue(np.all(pd.notnull(self.uo2.basis_set)))
        self.mam3.parse_basis_set()
        self.assertEqual(self.mam3.basis_set.shape[0], 32)
        self.assertTrue(np.all(pd.notnull(self.mam3.basis_set)))
        self.mam4.parse_basis_set()
        self.assertEqual(self.mam4.basis_set.shape[0], 53)
        self.assertTrue(np.all(pd.notnull(self.mam4.basis_set)))

    def test_parse_orbital(self):
        self.uo2.parse_orbital()
        self.assertEqual(self.uo2.orbital.shape[0], 141)
        self.assertTrue(np.all(pd.notnull(self.uo2.orbital)))
        self.mam3.parse_orbital()
        self.assertEqual(self.mam3.orbital.shape[0], 28)
        self.assertTrue(np.all(pd.notnull(self.mam3.orbital)))
        self.mam4.parse_orbital()
        self.assertEqual(self.mam4.orbital.shape[0], 91)
        self.assertTrue(np.all(pd.notnull(self.mam4.orbital)))

    def test_parse_momatrix(self):
        self.uo2.parse_momatrix()
        self.assertEqual(self.uo2.momatrix.shape[0], 19881)
        self.assertTrue(np.all(pd.notnull(self.uo2.momatrix)))
        self.mam3.parse_momatrix()
        self.assertEqual(self.mam3.momatrix.shape[0], 784)
        self.assertTrue(np.all(pd.notnull(self.mam3.momatrix)))
        self.mam4.parse_momatrix()
        self.assertEqual(self.mam4.momatrix.shape[0], 8281)
        self.assertTrue(np.all(pd.notnull(self.mam4.momatrix)))

    def test_parse_basis_set_order(self):
        self.uo2.parse_basis_set_order()
        self.assertEqual(self.uo2.basis_set_order.shape[0], 141)
        self.assertTrue(np.all(pd.notnull(self.uo2.basis_set_order)))
        self.mam3.parse_basis_set_order()
        self.assertEqual(self.mam3.basis_set_order.shape[0], 28)
        self.assertTrue(np.all(pd.notnull(self.mam3.basis_set_order)))
        self.mam4.parse_basis_set_order()
        self.assertEqual(self.mam4.basis_set_order.shape[0], 91)
        self.assertTrue(np.all(pd.notnull(self.mam4.basis_set_order)))

    def test_parse_frame(self):
        self.uo2.parse_frame()
        self.assertEqual(self.uo2.frame.shape[0], 1)
        self.assertTrue(np.all(pd.notnull(self.uo2.frame)))
        self.mam3.parse_frame()
        self.assertEqual(self.mam3.frame.shape[0], 1)
        self.assertTrue(np.all(pd.notnull(self.mam3.frame)))
        self.mam4.parse_frame()
        self.assertEqual(self.mam4.frame.shape[0], 1)
        self.assertTrue(np.all(pd.notnull(self.mam4.frame)))

    def test_to_universe(self):
        """Test the to_universe method."""
        uo2 = self.uo2.to_universe(ignore=True)
        mam3 = self.mam3.to_universe(ignore=True)
        for uni in [uo2, mam3]:
            for attr in ['atom', 'basis_set', 'basis_set_order',
                         'momatrix', 'orbital', 'frame']:
                self.assertTrue(hasattr(uni, attr))
