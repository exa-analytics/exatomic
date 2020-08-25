# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
import numpy as np
from unittest import TestCase

from exatomic.base import resource
from exatomic.interfaces import XYZ

class TestAtom(TestCase):
    def setUp(self):
        self.h2o = XYZ(resource('H2O.xyz'))
        self.h2o.parse_atom()
        self.h2 = XYZ(resource('H2.xyz'))
        self.h2.parse_atom()
        self.znpor = XYZ(resource('ZnPorphyrin.xyz'))
        self.znpor.parse_atom()
        self.cols = ['x', 'y', 'z']

    def test_center(self):
        # test vanilla centering algorithm
        centered = self.h2o.atom.center(idx=0)
        center = np.array([[ 0.        ,  0.        ,  0.        ],
                           [ 1.86954385, -0.05270446,  0.06149169],
                           [-0.54683005, -1.16477049,  1.35871308]])
        self.assertTrue(np.allclose(centered[self.cols].values, center))
        # test center of nuclear charge algorithm with a displaced H2 molecule
        center_nc = np.array([[ 0.00019274,  0.00011528, -0.00017576]])
        centered = self.znpor.atom.center(to='NuclChrg')
        self.assertTrue(np.allclose(centered.head(1)[self.cols].values, center_nc))
        center_nc = np.array([[-0.66832999,  0.02918682,  0.        ],
                              [ 0.66832999, -0.02918682,  0.        ]])
        centered = self.h2.atom.center(to='NuclChrg')
        self.assertTrue(np.allclose(centered[self.cols].values, center_nc))
        # test center of mass centering
        # for h2 it is the same as center of nuclear charge
        centered = self.h2.atom.center(to='Mass')
        self.assertTrue(np.allclose(centered[self.cols].values, center_nc))
        center_mass = np.array([[ 0.00018765,  0.00011287, -0.00019444 ]])
        centered = self.znpor.atom.center(to='Mass')
        print(centered.head(1)[self.cols].values)
        self.assertTrue(np.allclose(centered.head(1)[self.cols].values, center_mass))
        # make sure we raise an error when a centering method that has not been implemented
        # is given
        with self.assertRaises(NotImplementedError):
            self.h2.atom.center(idx=0, to='UnknownCenteringMethod')
        with self.assertRaises(TypeError):
            self.h2.atom.center()

    def test_translate(self):
        # displace in each coordinate direction by 0.1 Bohr
        disp = 0.1
        trans_x = self.h2o.atom.translate(dx=disp)
        self.assertTrue(np.allclose(trans_x['x'].values, self.h2o.atom['x']+disp))
        trans_y = self.h2o.atom.translate(dy=disp)
        self.assertTrue(np.allclose(trans_y['y'].values, self.h2o.atom['y']+disp))
        trans_z = self.h2o.atom.translate(dz=disp)
        self.assertTrue(np.allclose(trans_z['z'].values, self.h2o.atom['z']+disp))
        # displace by a vector dx=1, dy=2.5, dz=1 Bohr
        trans_vec = self.h2o.atom.translate(vector=[1, 2.5, 1])
        vec_data = np.array([[-9.07214576,  3.6018993 ,  0.96396292],
                             [-7.20260191,  3.54919484,  1.02545461],
                             [-9.61897581,  2.43712881,  2.32267601]])
        self.assertTrue(np.allclose(trans_vec[self.cols].values, vec_data))

    def test_rotate(self):
        # rotate 135 degrees
        rot_data = np.array([[6.342922101, -7.901243041, -0.036037077],
                             [5.058222648, -6.542008224,  0.025454611],
                             [7.553206450, -7.464293163,  1.322676007]])
        rotated = self.h2o.atom.rotate(theta=135.0)
        self.assertTrue(np.allclose(rotated[self.cols].values, rot_data))
        # rotate 0 degrees
        # should return the same coordinates
        rotated = self.h2o.atom.rotate(theta=0)
        self.assertTrue(np.allclose(rotated[self.cols].values, self.h2o.atom.last_frame[self.cols]))

    def test_align_to_axis(self):
        # align to x axis
        align_data = np.array([[ 0.          , 0.          , 0.          ],
                               [ 1.8712970607, 0.          ,-0.          ],
                               [-0.4688643421,-1.1790804580, 1.3754088998]])
        aligned = self.h2o.atom.align_to_axis(adx0=0, adx1=1, axis=[1,0,0])
        self.assertTrue(np.allclose(aligned[self.cols].values, align_data))
        # test aligning alog z axis and centering to the cnc
        align_nc = np.array([[0., 0., -0.668966999],
                             [0., 0.,  0.668966999]])
        aligned = self.h2.atom.align_to_axis(adx0=0, adx1=1, axis=[0, 0, 1], center_to='NuclChrg')
        self.assertTrue(np.allclose(aligned[self.cols].values, align_nc))


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
