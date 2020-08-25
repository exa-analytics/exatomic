# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np
from unittest import TestCase
from exatomic import XYZ
from ..traits import (atom_traits, two_traits, 
                      frame_traits, uni_traits)

h2 = '''2

H 0. 0. -0.35
C 0. 0.  0.35'''

# TODO : need a simple universe with field

class TestTraits(TestCase):
    def setUp(self):
        xyz = XYZ(h2)
        xyz.parse_atom()
        self.uni = xyz.to_universe()

    def test_atom_traits(self):
        atom = atom_traits(self.uni.atom)
        self.assertEqual(atom['atom_x'], '[[0.0,0.0]]')
        self.assertEqual(atom['atom_y'], '[[0.0,0.0]]')
        self.assertEqual(atom['atom_z'], '[[-0.661,0.661]]')
        self.assertEqual(atom['atom_s'], '[[1,0]]')
        # Alphabetical order of categories
        self.assertTrue(np.isclose(atom['atom_cr'][0], 1.4172945))
        self.assertTrue(np.isclose(atom['atom_cr'][1], 0.60471232))
        self.assertEqual(atom['atom_c'][0], '#909090')
        self.assertEqual(atom['atom_c'][1][0], '#') # H changing colors
        atom = atom_traits(self.uni.atom, atomcolors={'H': '#000000'},
                                          atomradii={'H': 1.0})
        self.assertEqual(atom['atom_c'][1], '#000000')
        self.assertTrue(np.isclose(atom['atom_cr'][1], 1.0))

    def test_two_traits(self):
        two = two_traits(self.uni)
        self.assertEqual(two['two_b0'], '[[0]]')
        self.assertEqual(two['two_b1'], '[[1]]')

    def test_frame_traits(self):
        frame = frame_traits(self.uni)
        self.assertEqual(frame, {})

    def test_uni_traits(self):
        unargs, _, _ = uni_traits(self.uni)
        for at in ['x', 'y', 'z', 's', 'cr', 'c']:
            self.assertTrue('atom_' + at in unargs)
        for b in ['b0', 'b1']:
            self.assertTrue('two_' + b in unargs)
