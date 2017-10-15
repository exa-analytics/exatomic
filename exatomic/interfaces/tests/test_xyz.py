# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Tests for :mod:`~exatomic.xyz`
################################

"""
#from io import StringIO
#from exa.test import UnitTester
#from exatomic.xyz import read_xyz
#from exatomic import _np as np
#
#
#xyzfl = """3
#1 comment
#H 0.0  0.0 0.0
#H 0.0  0.7 0.0
#H 0.0 -0.7 0.0
#2
#comments 2
#H 0.0  0.0 0.0
#H 0.0 -0.7 0.0
#"""
#
#
#class TestXYZ(UnitTester):
#    """
#    """
#    def test_read_xyz(self):
#        """
#        """
#        pass
#
#
#
#    def setUp(self):
#        self.raw = xyz._rawdf(StringIO(xyzfl))
#        self.idx = xyz._index(self.raw)
#
#    def test_read_xyz(self):
#        to = xyz._parse_xyz(self.raw, 'A', self.idx)
#        self.assertTrue(np.all(np.array(to.index.levels[0]) == np.array([0, 1])))
#        self.assertTrue(np.all(np.array(to.index.levels[1]) == np.array([0, 1, 2])))
