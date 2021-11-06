# -*- coding: utf-8 -*-
# Copyright (c) 2015-2022, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Tests for :mod:`~exatomic.exa.core.numerical`
#####################################
"""
from unittest import TestCase
import numpy as np
import pandas as pd
from pandas.core.dtypes.dtypes import CategoricalDtype
from exatomic.exa.core.numerical import Numerical, Series, DataFrame


class TSeries(Series):
    _sname = 'testing'
    _iname = 'index'
    _stype = np.float64
    _itype = np.int64
    _precision = 2


class TDF0(DataFrame):
    _index = 'index'
    _columns = ['column']


class TDF1(DataFrame):
    _cardinal = ('group', np.int64)
    _index = 'index'
    _columns = ['column', 'type']
    _categories = {'type': str}


class TDF2(DataFrame):
    _cardinal = ('group', str)
    _index = 'index'
    _columns = ['x', 'y', 'z', 'type']
    _categories = {'type': np.int64}


class NumericalTest(TestCase):
    def setUp(self):
        self.numerical = Numerical()

    def test_slice(self):
        with self.assertRaises(AttributeError):
            self.numerical.slice_naive(0)


class SeriesTest(TestCase):
    def setUp(self):
        self.series = TSeries(np.random.rand(10))

    def test_underattr(self):
        """
        Test to ensure the (class level) underscore attributes (of
        :class:`~exatomic.exa.core.tests.test_numerical.TestingSeries`) are respected.
        """
        self.assertTrue(self.series.name == TSeries._sname)
        self.assertTrue(self.series.index.name == TSeries._iname)

    def test_copy(self):
        """Test :func:`~exatomic.exa.core.numerical.Series.copy`."""
        cp = self.series.copy()
        self.assertTrue(cp.eq(self.series).all())
        self.assertIsInstance(cp, self.series.__class__)


class DF0Test(TestCase):
    """
    Test a basic example of an instance of :class:`~exatomic.exa.core.numerical.DataFrame`.
    """
    def setUp(self):
        column = np.random.rand(10)
        self.df = TDF0.from_dict({'column': column})

    def test_copy(self):
        """Test :func:`~exatomic.exa.core.numerical.DataFrame.copy`."""
        cp = self.df.copy()
        self.assertTrue(cp.eq(self.df).all().all())    # All columns are equal
        self.assertIsInstance(cp, self.df.__class__)


class DF1Test(TestCase):
    """
    Test an example instance of :class:`~exatomic.exa.core.numerical.DataFrame` with groupby.
    """
    def setUp(self):
        column = np.random.rand(10)
        group = [0, 0, 0, 0, 1, 1, 1, 2, 2, 3]
        typ = ['A']*5 + ['B']*5
        self.df = TDF1.from_dict({'column': column, 'type': typ, 'group': group})

    def test_copy(self):
        """Test :func:`~exatomic.exa.core.numerical.DataFrame.copy`."""
        cp = self.df.copy()
        self.assertTrue(cp.eq(self.df).all().all())    # All columns are equal
        self.assertIsInstance(cp, self.df.__class__)

    def test_categories(self):
        """Test that categoricals are being handled correctly."""
        self.assertIsInstance(self.df['type'].dtype, CategoricalDtype)

    def test_cardinal_groupby(self):
        grps = self.df.cardinal_groupby()
        self.assertEqual(list(grps.groups.keys()), [0, 1, 2, 3])


class DF2Test(TestCase):
    """
    Test an example instance of :class:`~exatomic.exa.core.numerical.DataFrame` with groupby.
    """
    def setUp(self):
        """Create instance of :class:`~exatomic.exa.core.test.test_numerical.TestingDF2`."""
        x = np.random.rand(10)
        y = np.random.rand(10)
        z = np.random.rand(10)
        typ = [0, 0, 0, 0, 1, 1, 1, 2, 2, 3]
        group = ['A']*5 + ['B']*5
        self.df = TDF2.from_dict({'x': x, 'y': y, 'z': z, 'type': typ,
                                  'group': group})

    def test_copy(self):
        """Test :func:`~exatomic.exa.core.numerical.DataFrame.copy`."""
        cp = self.df.copy()
        self.assertTrue(cp.eq(self.df).all().all())    # All columns are equal
        self.assertIsInstance(cp, self.df.__class__)

    def test_categories(self):
        """Test that categoricals are being handled correctly."""
        self.assertIsInstance(self.df['type'].dtype, CategoricalDtype)
        self.assertIsInstance(self.df['group'].dtype, CategoricalDtype)
        self.df._revert_categories()
        self.assertNotIsInstance(self.df['type'].dtype, CategoricalDtype)
        self.assertNotIsInstance(self.df['group'].dtype, CategoricalDtype)
        self.df._set_categories()

    def test_slice_cardinal(self):
        df1 = self.df.slice_cardinal(["A"])
        self.assertEqual(df1.shape, (5, 5))
