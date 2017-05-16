# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Tests for the XYZ Editor
##########################
The tests here use some contrived examples.
"""
import numpy as np
from unittest import TestCase
from exa import units
from exatomic.atom import Atom


simple0 = """3

O 0.0 0.0 0.0
H 0.0  0.0  0.0
H 0.0  0.0  0.0
"""

simple1 = """3
1 comment and blank space
O 0.0 0.0 0.0
H 0.0  0.0  0.0
H 0.0  0.0  0.0


"""

simple2 = """3
custom columns 1
O 8 0.0 0.0 0.0
H 1 0.0  0.0  0.0
H 1 0.0  0.0  0.0
"""

traj = """3
step 1
O 0.0 0.0 0.0
H 0.0  0.0  0.0
H 0.0  0.0  0.0
3
step 2
O 0.0 0.0 0.0
H 0.0  0.0  0.0
H 0.0  0.0  0.0
"""

variable = """3
molecule 1
O 0.0 0.0 0.0
H 0.0  0.0  0.0
H 0.0  0.0  0.0
6
molecule 2
O 0.0 0.0 0.0
H 0.0  0.0  0.0
H 0.0  0.0  0.0
O 0.0 0.0 0.0
H 0.0  0.0  0.0
H 0.0  0.0  0.0
5
molecule 3
O 0.0 0.0 0.0
H 0.0  0.0  0.0
H 0.0  0.0  0.0
O 0.0 0.0 0.0
H 0.0  0.0  0.0
"""


class TestXYZ(TestCase):
    """Tests for :class:`XYZ`."""
    def test_simple0(self):
        """Test parser on the simple xyz."""
        xyz = XYZ(simple0)
        self.assertIsInstance(xyz, XYZ)
        self.assertEqual(len(xyz), 5)
        self.assertIsInstance(xyz.atom, Atom)
        self.assertEqual(len(xyz.atom), 3)
        self.assertDictEqual(xyz.comments, {0: ""})
        self.assertDictEqual(xyz.atom.meta, {'unit': units.Angstrom})

    def test_simple1(self):
        """Test parser on simple1."""
        xyz = XYZ(simple1)
        self.assertIsInstance(xyz, XYZ)
        self.assertEqual(len(xyz), 7)
        self.assertIsInstance(xyz.atom, Atom)
        self.assertEqual(len(xyz.atom), 3)
        self.assertDictEqual(xyz.comments, {0: "1 comment and blank space"})
        self.assertDictEqual(xyz.atom.meta, {'unit': units.Angstrom})

    def test_simple2(self):
        """Test parsing with custom columns."""
        xyz = XYZ(simple2, columns=("symbol", "Z", "x", "y", "z"))
        self.assertIsInstance(xyz, XYZ)
        self.assertEqual(len(xyz), 5)
        self.assertIsInstance(xyz.atom, Atom)
        self.assertTupleEqual(xyz.atom.shape, (3, 6))
        self.assertDictEqual(xyz.comments, {0: "custom columns 1"})
        self.assertDictEqual(xyz.atom.meta, {'unit': units.Angstrom})

    def test_traj(self):
        """Test trajectory xyz parsing."""
        xyz = XYZ(traj)
        self.assertIsInstance(xyz, XYZ)
        self.assertEqual(len(xyz), 10)
        self.assertIsInstance(xyz.atom, Atom)
        self.assertTrue(np.all(xyz.atom.groupby("frame").size() == 3))
        self.assertDictEqual(xyz.comments, {0: "step 1", 1: "step 2"})
        self.assertDictEqual(xyz.atom.meta, {'unit': units.Angstrom})

    def test_variable(self):
        """Test variable nat xyz-like files."""
        xyz = XYZ(variable)
        self.assertIsInstance(xyz, XYZ)
        self.assertIsInstance(xyz.atom, Atom)
        self.assertDictEqual(xyz.atom.meta, {'unit': units.Angstrom})
        self.assertEqual(xyz.atom.groupby("frame").ngroups, 3)
        self.assertDictEqual(xyz.comments, {0: "molecule 1", 1: "molecule 2",
                                            2: "molecule 3"})

    def test_write(self):
        self.fail("write not implemented")

    def test_from_universe(self):
        self.fail("from_universe not implemented")

    def test_from_atom(self):
        self.fail("from_atom not implemented")
