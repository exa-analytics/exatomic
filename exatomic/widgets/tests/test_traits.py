# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0

import numpy as np
from unittest import TestCase

from exatomic import XYZ
from ..traits import (atom_traits, field_traits,
                      two_traits, frame_traits,
                      uni_traits)

h2 = '''2

H 0. 0. -0.35
C 0. 0.  0.35'''

# TODO : need a simple universe with field

class TestTraits(TestCase):

    def setUp(self):
        self.uni = XYZ(h2).to_universe()

    def test_atom_traits(self):
        atom = atom_traits(self.uni.atom)
        self.assertEqual(atom['atom_x'], '[[0.0,0.0]]')
        self.assertEqual(atom['atom_y'], '[[0.0,0.0]]')
        self.assertEqual(atom['atom_z'], '[[-0.661,0.661]]')
        self.assertEqual(atom['atom_s'], '[[1,0]]')
        # Alphabetical order of categories
        self.assertTrue(np.isclose(atom['atom_r'][0], 0.708647))
        self.assertTrue(np.isclose(atom['atom_r'][1], 0.302356))
        self.assertEqual(atom['atom_c'][0], '#909090')
        self.assertEqual(atom['atom_c'][1][0], '#') # H changing colors
        atom = atom_traits(self.uni.atom, atomcolors={'H': '#000000'},
                                          atomradii={'H': 1.0})
        self.assertEqual(atom['atom_c'][1], '#000000')
        self.assertTrue(np.isclose(atom['atom_r'][1], 0.5))

    def test_field_traits(self):
        pass

    def test_two_traits(self):
        two = two_traits(self.uni)
        self.assertEqual(two['two_b0'], '[[0]]')
        self.assertEqual(two['two_b1'], '[[1]]')

    def test_frame_traits(self):
        frame = frame_traits(self.uni)
        self.assertEqual(frame, {})

    def test_uni_traits(self):
        unargs, flds = uni_traits(self.uni)
        for at in ['x', 'y', 'z', 's', 'r', 'c']:
            self.assertTrue('atom_' + at in unargs)
        for b in ['b0', 'b1']:
            self.assertTrue('two_' + b in unargs)
