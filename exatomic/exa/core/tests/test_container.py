# -*- coding: utf-8 -*-
# Copyright (c) 2015-2022, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Tests for :mod:`~exatomic.exa.core.container`
#######################################
"""
import sys
from os import remove
from unittest import TestCase
from tempfile import mkdtemp
import pandas as pd
from pandas.core.dtypes.dtypes import CategoricalDtype
from exatomic.exa import Container, TypedMeta, DataFrame, Series, Field


class DummyDataFrame(DataFrame):
    _index = 'index'
    _categories = {'cat': str}
    _columns = ['x', 'y', 'z', 'cat']


class DummySeries(Series):
    _precision = 3
    _sname = 'field'
    _iname = 'value'


class DummyMeta(TypedMeta):
    s0 = DummySeries
    s1 = DummySeries
    df = DummyDataFrame


class DummyContainer(Container, metaclass=DummyMeta):
    pass


class TestContainer(TestCase):
    @classmethod
    def setUpClass(cls):
        x = [0, 0, 0, 0, 0]
        y = [1.1, 2.2, 3.3, 4.4, 5.5]
        z = [0.5, 1.5, 2.5, 3.5, 4.5]
        cat = ['cube', 'sphere', 'cube', 'sphere', 'cube']
        group = [0, 0, 1, 1, 1]
        cls.container = DummyContainer()
        cls.container._test = False
        cls.container.s0 = DummySeries(y)
        cls.container.s1 = DummySeries(cat, dtype='category')
        cls.container.df = DataFrame.from_dict({'x': x, 'y': y, 'z': z, 'cat': cat, 'group': group})
        cls.container._cardinal = "df"

    def test_attributes(self):
        self.assertIsInstance(self.container.s0, DummySeries)
        self.assertIsInstance(self.container.s1.dtype, CategoricalDtype)
        self.assertIsInstance(self.container.df, DummyDataFrame)

    def test_copy(self):
        cp = self.container.copy()
        self.assertIsNot(self.container, cp)
        cp = self.container.copy(name="name", description="descr", meta={'key': "value"})
        self.assertEqual(cp.name, "name")
        self.assertEqual(cp.description, "descr")
        self.assertDictEqual(cp.meta, {'key': "value"})

    def test_concat(self):
        with self.assertRaises(NotImplementedError):
            self.container.concat()

    def test_slice_naive(self):
        c = self.container[[0]].copy()
        self.assertEqual(c.df.shape, (1, 5))
        c = self.container[1:]
        self.assertEqual(c.df.shape, (4, 5))
        c = self.container.slice_naive([0])
        self.assertEqual(c.df.shape, (1, 5))
        c = self.container.slice_naive(0)
        self.assertEqual(c.df.shape, (1, 5))
        c = self.container.slice_naive(slice(0, 1))
        self.assertEqual(c.df.shape, (1, 5))

    def test_getsizeof(self):
        size_bytes = sys.getsizeof(self.container)
        self.assertIsInstance(size_bytes, int)
        self.assertTrue(size_bytes > 100)

    def test_memory_usage(self):
        mem = self.container.memory_usage()
        self.assertEqual(mem.shape, (5, ))
        mem = self.container.memory_usage(True)
        self.assertIsInstance(mem, str)

    def test_save_load_to_hdf(self):
        tmpdir = mkdtemp()
        path = self.container.save()
        self.assertTrue(path.endswith(".hdf5"))
        remove(path)
        path = self.container.save(tmpdir)
        self.assertTrue(path.endswith(".hdf5"))
        remove(path)
        with self.assertRaises(ValueError):
            self.container.save(tmpdir + "/stuff.things")
        self.container.to_hdf(path)
        c = Container.load(path)
        self.assertEqual(c.df.shape, self.container.df.shape)
        c = Container.from_hdf(path)
        self.assertEqual(c.df.shape, self.container.df.shape)
        remove(path)

    def test_dunder(self):
        c = Container(x=DataFrame())
        self.assertTrue(hasattr(c, "x"))
        del c["x"]
        self.assertFalse(hasattr(c, "x"))
        with self.assertRaises(AttributeError):
            c["x"]
