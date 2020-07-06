# -*- coding: utf-8 -*-
## Copyright (c) 2015-2020, Exa Analytics Development Team
## Distributed under the terms of the Apache License 2.0
"""
Tests for :mod:`~exatomic.interfaces.cube`
#############################################
"""
import numpy as np
from unittest import TestCase
from exatomic.base import resource, staticdir
from exatomic.interfaces.cube import Cube, uni_from_cubes
from sys import platform


class TestCube(TestCase):
    """Tests cube reading and writing."""

    def setUp(self):
        self.lg = Cube(resource('mol-carbon-dz-1.cube'))
        self.sm1 = Cube(resource('adf-lu-35.cube'))
        self.sm2 = Cube(resource('adf-lu-36.cube'))
        self.uni = uni_from_cubes(staticdir() + '/cube/', ext='*lu*cube')

    def test_parse_atom(self):
        self.lg.parse_atom()
        self.sm1.parse_atom()
        self.sm2.parse_atom()
        self.assertEqual(self.lg.atom.shape[0], 1)
        self.assertEqual(self.sm1.atom.shape[0], 1)
        self.assertEqual(self.sm2.atom.shape[0], 1)

    def test_parse_field(self):
        self.lg.parse_field()
        self.sm1.parse_field()
        self.sm2.parse_field()
        self.assertEqual(self.lg.field.shape[0], 1)
        self.assertEqual(self.sm1.field.shape[0], 1)
        self.assertEqual(self.sm2.field.shape[0], 1)
        self.assertEqual(self.lg.field.field_values[0].shape[0], 132651)
        self.assertEqual(self.sm1.field.field_values[0].shape[0], 4913)
        self.assertEqual(self.sm2.field.field_values[0].shape[0], 4913)

    def test_to_universe(self):
        lg = self.lg.to_universe()
        sm1 = self.sm1.to_universe()
        sm2 = self.sm2.to_universe()
        for uni in [lg, sm1, sm2]:
            for attr in ['atom', 'field']:
                self.assertTrue(hasattr(uni, attr))

    def test_uni_from_cubes_rotate_and_write(self):
        self.assertEqual(self.uni.field.shape[0], 2)
        self.assertEqual(len(self.uni.field.field_values), 2)
        rot = self.uni.field.rotate(0, 1, np.pi / 4)
        self.assertEqual(rot.shape[0], 2)
        if "win" not in platform.casefold():
            f = Cube.from_universe(self.uni, 1)
            self.assertEqual(len(f), 874)
