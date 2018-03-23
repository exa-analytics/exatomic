# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
import os
import bz2
from os.path import abspath, join
import numpy as np
import pandas as pd
from unittest import TestCase

import exatomic
from exatomic import gaussian

from exatomic.gaussian import Output, Fchk

class TestFchk(TestCase):
    def setUp(self):
        path = abspath(join(abspath(exatomic.__file__), "../static/gaussian/"))
        fp = join(path, 'g09-ch3nh2-631g.fchk.bz2')
        with bz2.open(fp) as f:
            self.mam1 = Fchk(f.read().decode('utf-8'))
        fp = join(path, 'g09-ch3nh2-augccpvdz.fchk.bz2')
        with bz2.open(fp) as f:
            self.mam2 = Fchk(f.read().decode('utf-8'))

    def test_parse_atom(self):
        self.mam1.parse_atom()
        self.assertEqual(self.mam1.atom.shape, (7, 7))
        self.assertTrue(np.all(pd.notnull(self.mam1.atom)))
        self.mam2.parse_atom()
        self.assertEqual(self.mam2.atom.shape, (7, 7))
        self.assertTrue(np.all(pd.notnull(self.mam2.atom)))

    def test_parse_basis_set(self):
        self.mam1.parse_basis_set()
        self.assertEqual(self.mam1.basis_set.shape, (32, 6))
        self.assertTrue(np.all(pd.notnull(self.mam1.basis_set)))
        self.mam2.parse_basis_set()
        self.assertEqual(self.mam2.basis_set.shape, (53, 6))
        self.assertTrue(np.all(pd.notnull(self.mam2.basis_set)))

    def test_parse_orbital(self):
        self.mam1.parse_orbital()
        self.assertEqual(self.mam1.orbital.shape, (28, 6))
        self.assertTrue(np.all(pd.notnull(self.mam1.orbital)))
        self.mam2.parse_orbital()
        self.assertEqual(self.mam2.orbital.shape, (91, 6))
        self.assertTrue(np.all(pd.notnull(self.mam2.orbital)))

    def test_parse_momatrix(self):
        self.mam1.parse_momatrix()
        self.assertEqual(self.mam1.momatrix.shape, (784, 4))
        self.assertTrue(np.all(pd.notnull(self.mam1.momatrix)))
        self.mam2.parse_momatrix()
        self.assertEqual(self.mam2.momatrix.shape, (8281, 4))
        self.assertTrue(np.all(pd.notnull(self.mam2.momatrix)))

    def test_parse_basis_set_order(self):
        self.mam1.parse_basis_set_order()
        self.assertEqual(self.mam1.basis_set_order.shape, (28, 6))
        self.assertTrue(np.all(pd.notnull(self.mam1.basis_set_order)))
        self.mam2.parse_basis_set_order()
        self.assertEqual(self.mam2.basis_set_order.shape, (91, 6))
        self.assertTrue(np.all(pd.notnull(self.mam2.basis_set_order)))

    def test_parse_frame(self):
        self.mam1.parse_frame()
        self.assertEqual(self.mam1.frame.shape, (1, 1))
        self.assertTrue(np.all(pd.notnull(self.mam1.frame)))
        self.mam2.parse_frame()
        self.assertEqual(self.mam2.frame.shape, (1, 1))
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
        path = abspath(join(abspath(exatomic.__file__), "../static/gaussian/"))
        fp = join(path, 'g09-uo2.out.bz2')
        with bz2.open(fp) as f:
            self.uo2 = Output(f.read().decode('utf-8'))
        fp = join(path, 'g09-ch3nh2-631g.out.bz2')
        with bz2.open(fp) as f:
            self.mam = Output(f.read().decode('utf-8'))

    def test_parse_atom(self):
        self.uo2.parse_atom()
        self.assertEqual(self.uo2.atom.shape, (3, 7))
        self.assertTrue(np.all(pd.notnull(self.uo2.atom)))
        self.mam.parse_atom()
        self.assertEqual(self.mam.atom.shape, (7, 7))
        self.assertTrue(np.all(pd.notnull(self.mam.atom)))

    def test_parse_basis_set(self):
        self.uo2.parse_basis_set()
        self.assertEqual(self.uo2.basis_set.shape, (49, 6))
        self.assertTrue(np.all(pd.notnull(self.uo2.basis_set)))
        self.mam.parse_basis_set()
        self.assertEqual(self.mam.basis_set.shape, (32, 6))
        self.assertTrue(np.all(pd.notnull(self.mam.basis_set)))

    def test_parse_orbital(self):
        self.uo2.parse_orbital()
        self.assertEqual(self.uo2.orbital.shape, (141, 6))
        self.assertTrue(np.all(pd.notnull(self.uo2.orbital)))
        self.mam.parse_orbital()
        self.assertEqual(self.mam.orbital.shape, (28, 7))
        self.assertTrue(np.all(pd.notnull(self.mam.orbital)))

    def test_parse_momatrix(self):
        self.uo2.parse_momatrix()
        self.assertEqual(self.uo2.momatrix.shape, (19881, 4))
        self.assertTrue(np.all(pd.notnull(self.uo2.momatrix)))
        self.mam.parse_momatrix()
        self.assertEqual(self.mam.momatrix.shape, (784, 4))
        self.assertTrue(np.all(pd.notnull(self.mam.momatrix)))

    def test_parse_basis_set_order(self):
        self.uo2.parse_basis_set_order()
        self.assertEqual(self.uo2.basis_set_order.shape, (141, 5))
        self.assertTrue(np.all(pd.notnull(self.uo2.basis_set_order)))
        self.mam.parse_basis_set_order()
        self.assertEqual(self.mam.basis_set_order.shape, (28, 5))
        self.assertTrue(np.all(pd.notnull(self.mam.basis_set_order)))

    def test_parse_frame(self):
        self.uo2.parse_frame()
        self.assertEqual(self.uo2.frame.shape, (1, 5))
        self.assertTrue(np.all(pd.notnull(self.uo2.frame)))
        self.mam.parse_frame()
        self.assertEqual(self.mam.frame.shape, (1, 6))
        self.assertTrue(np.all(pd.notnull(self.mam.frame)))

    def test_to_universe(self):
        """Test the to_universe method."""
        uo2 = self.uo2.to_universe(ignore=True)
        mam = self.mam.to_universe(ignore=True)
        for uni in [uo2, mam]:
            for attr in ['atom', 'basis_set', 'basis_set_order',
                         'momatrix', 'orbital', 'frame']:
                self.assertTrue(hasattr(uni, attr))
