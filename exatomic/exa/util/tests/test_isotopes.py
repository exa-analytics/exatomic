# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Tests for :mod:`~exa.isotopes`
#############################################
Note that elements and isotope singletons are created on module import.
"""
from unittest import TestCase
from exa.util import isotopes


class TestIsotopes(TestCase):
    """Tests for isotope and element generation."""
    def test_created(self):
        """Check that elements and isotope objects were created."""
        iso = isotopes.as_df()
        for sym in iso['symbol'].unique():
            self.assertTrue(hasattr(isotopes, sym))

    def test_element(self):
        """Test element attributes."""
        self.assertEqual(isotopes.H.Z, 1)
        self.assertGreater(isotopes.H.mass, 1.007)
        self.assertGreater(isotopes.H.radius, 0.6)

    def test_isotope(self):
        """Test isotope attributes."""
        self.assertEqual(isotopes.H['1'].Z, 1)
        self.assertEqual(isotopes.H['1'].A, 1)
        self.assertGreater(isotopes.H['1'].mass, 1.007)
        self.assertGreater(isotopes.H['1'].radius, 0.6)

