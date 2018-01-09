# -*- coding: utf-8 -*-
## Copyright (c) 2015-2017, Exa Analytics Development Team
## Distributed under the terms of the Apache License 2.0
"""
Tests for :mod:`~exatomic.filetypes.cube`
#################################
"""
from unittest import TestCase
import os
from exatomic import Universe
from exatomic.core.atom import Atom
from exatomic.core.field import AtomicField
from exatomic.interfaces.cube import Cube
from exatomic.gaussian import Output


class TestCube(TestCase):
    """Tests cube reading and writing."""
    pass

#    def setUp(self):
#        cd = os.path.abspath(__file__).split(os.sep)[:-1]
#        self.h2o = Output(os.sep.join(cd + ['h2o.out'])).to_universe()
#        self.h2o.add_molecular_orbitals(vector=[2])
#        self.cubed = Cube.from_universe(self.h2o, 0)
#
#    def test_from_universe(self):
#        self.assertIs(type(self.cubed), Cube)
#
#    def test_parse_atom(self):
#        self.cubed.parse_atom()
#        self.assertIs(type(self.cubed.atom), Atom)
#
#    def test_parse_field(self):
#        self.cubed.parse_atom()
#        self.assertIs(type(self.cubed.field), AtomicField)
#
#    def test_to_universe(self):
#        uni = self.cubed.to_universe()
#        self.assertIs(type(uni), Universe)
