# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
import numpy as np
from unittest import TestCase

from exatomic.base import resource
from exatomic.interfaces import XYZ

class TestAtom(TestCase):
    def setUp(self):
        self.h2o = XYZ(resource('H2O.xyz'))
        self.h2o.parse_atom()
        self.cols = ['x', 'y', 'z']

    def test_center(self):
        centered = self.h2o.atom.center(0)
        center = np.array([[ 0.        ,  0.        ,  0.        ],
                           [ 1.86954385, -0.05270446,  0.06149169],
                           [-0.54683005, -1.16477049,  1.35871308]])
        self.assertTrue(np.allclose(centered[['x','y','z']].values, center))

    def test_translate(self):
        disp = 0.1
        trans_x = self.h2o.atom.translate(dx=disp)
        self.assertTrue(np.allclose(trans_x['x'].values, self.h2o.atom['x']+disp))
        trans_y = self.h2o.atom.translate(dy=disp)
        self.assertTrue(np.allclose(trans_y['y'].values, self.h2o.atom['y']+disp))
        trans_z = self.h2o.atom.translate(dz=disp)
        self.assertTrue(np.allclose(trans_z['z'].values, self.h2o.atom['z']+disp))
        trans_vec = self.h2o.atom.translate(vector=[1, 2.5, 1])
        vec_data = np.array([[-9.07214576,  3.6018993 ,  0.96396292],
                             [-7.20260191,  3.54919484,  1.02545461],
                             [-9.61897581,  2.43712881,  2.32267601]])
        self.assertTrue(np.allclose(trans_vec[self.cols].values, vec_data))

    def test_rotate(self):
        rot_data = np.array([[6.342922101, -7.901243041, -0.036037077],
                             [5.058222648, -6.542008224,  0.025454611],
                             [7.553206450, -7.464293163,  1.322676007]])
        rotated = self.h2o.atom.rotate(theta=135.0)
        self.assertTrue(np.allclose(rotated[self.cols].values, rot_data))


#"""
#Tests for the Atom DataFrame
##############################
#The tests here use some contrived examples.
#"""
#import numpy as np
#from unittest import TestCase
#from exa import units
#from exa.core.dataframe import ColumnError
#from exatomic.atom import Atom
#
#
#class TestAtom(TestCase):
#    """Tests for :class:`~exatomic.atom.Atom`."""
#    def test_init(self):
#        """Test that the atom dataframe raises errors correctly."""
#        with self.assertRaises(ColumnError):
#            Atom()
