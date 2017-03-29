# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0

import os
import numpy as np
import pandas as pd
try:
    from exa.test.tester import UnitTester
except:
    from exa.tester import UnitTester
from exatomic.molcas.output import Output, Orb


class TestOutput(UnitTester):
    """Test the Molcas output file editor."""

    def setUp(self):
        cd = os.path.abspath(__file__).split(os.sep)[:-1]
        self.uo2sp = Output(os.sep.join(cd + ['mol-uo2-anomb.out']))

    def test_parse_atom(self):
        """Test the atom table parser."""
        self.uo2sp.parse_atom()
        self.assertEqual(self.uo2sp.atom.shape, (3, 8))
        self.assertTrue(np.all(pd.notnull(self.uo2sp.atom)))

    def test_parse_basis_set_order(self):
        """Test the basis set order table parser."""
        self.uo2sp.parse_basis_set_order()
        self.assertEqual(self.uo2sp.basis_set_order.shape, (69, 8))
        self.assertTrue(np.all(pd.notnull(self.uo2sp.basis_set_order)))

    def test__basis_set_map(self):
        """Test the gaussian basis set map helper parser."""
        df = self.uo2sp._basis_set_map()
        self.assertEqual(df.shape, (6, 5))
        self.assertTrue(np.all(pd.notnull(df)))

    def test_parse_basis_set(self):
        """Test the gaussian basis set table parser."""
        self.uo2sp.parse_basis_set()
        self.assertEqual(self.uo2sp.basis_set.shape, (451, 6))
        self.assertTrue(np.all(pd.notnull(self.uo2sp.basis_set)))


class TestOrb(UnitTester):
    """Test the Molcas Orb file parser."""

    def setUp(self):
        cd = os.path.abspath(__file__).split(os.sep)[:-1]
        self.uo2sporb = Orb(os.sep.join(cd + ['mol-uo2-anomb.scforb']))

    def test_parse_momatrix(self):
        """Test the momatrix table parser."""
        self.uo2sporb.parse_momatrix()
        self.assertEqual(self.uo2sporb.momatrix.shape, (4761, 4))
        self.assertTrue(np.all(pd.notnull(self.uo2sporb.momatrix)))
