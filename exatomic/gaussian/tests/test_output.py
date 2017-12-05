# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
import os
import numpy as np
import pandas as pd
from unittest import TestCase
from exatomic import Universe
from exatomic.gaussian import Output
# def decompress(fname):
#     """Helper function for decompressing example."""
#     path = os.path.abspath(os.path.join(os.path.abspath(exatomic.__file__), "../static/gaussian/" + fname + ".bz2"))
#     with bz2.open(path) as f:
#         if 'fchk' in fname:
#             return gaussian.Fchk(f.read().decode("utf-8"))
#         return gaussian.Output(f.read().decode("utf-8"))

class TestOutput(TestCase):
    """
    This test ensures that the parsing functionality works on
    a smattering of output files that were generated with the
    Gaussian software package. Target syntax is for Gaussian
    09.
    """
    pass
#    def setUp(self):
#        """Grab the output file as an editor."""
#        # TODO : add some cartesian basis set files
#        #        a geometry optimization and
#        #        maybe properties? like the frequency
#        #        and tddft calcs
#        # uo2sp == UO2(2+) single point calculation
#        cd = os.path.abspath(__file__).split(os.sep)[:-1]
#        self.uo2sp = Output(os.sep.join(cd + ['gaussian-uo2.out']))
#
#    def test_parse_atom(self):
#        """Test the atom table parser."""
#        self.uo2sp.parse_atom()
#        self.assertEqual(self.uo2sp.atom.shape, (3, 7))
#        self.assertTrue(np.all(pd.notnull(self.uo2sp.atom)))
#
#    def test_parse_basis_set(self):
#        """Test the gaussian basis set table parser."""
#        self.uo2sp.parse_basis_set()
#        self.assertEqual(self.uo2sp.basis_set.shape[0], 49)
#        self.assertTrue(np.all(pd.notnull(self.uo2sp.basis_set)))
#
#    def test_parse_orbital(self):
#        """Test the orbital table parser."""
#        self.uo2sp.parse_orbital()
#        self.assertEqual(self.uo2sp.orbital.shape, (141, 6))
#        self.assertTrue(np.all(pd.notnull(self.uo2sp.orbital)))
#
#    def test_parse_momatrix(self):
#        """Test the momatrix table parser."""
#        self.uo2sp.parse_momatrix()
#        self.assertEqual(self.uo2sp.momatrix.shape, (19881, 4))
#        self.assertTrue(np.all(pd.notnull(self.uo2sp.momatrix)))
#
#    def test_parse_basis_set_order(self):
#        """Test the basis set order table parser."""
#        self.uo2sp.parse_basis_set_order()
#
#    #def test__basis_set(self):
#    #    Include spherical/cartesian
#    #    pass
#
#    #def test__basis_set_order(self):
#    #    Include spherical/cartesian
#    #    pass
#
#    def test_parse_frame(self):
#        """Test the frame table parser."""
#        self.uo2sp.parse_frame()
#        self.assertEqual(self.uo2sp.frame.shape, (1, 5))
#        self.assertTrue(np.all(pd.notnull(self.uo2sp.frame)))
#
#    def test_to_universe(self):
#        """Test the to_universe method."""
#        uni = self.uo2sp.to_universe()
#        self.assertTrue(isinstance(uni, Universe))
