# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Matrix Tests
###############
Testing for Matrix base classes and numba funcs.
"""
import numpy as np
from unittest import TestCase
from exatomic.core.matrices import (_symmetric_from_square,
                                    _symmetric_to_square,
                                    _square_from_square,
                                    _square_to_square)


class TestNumbaFuncs(TestCase):
    """Test the numba functions for the correct indexing and reshaping."""
    def setUp(self):
        self.tidx0 = np.array([0, 1, 1, 2, 2, 2])
        self.tidx1 = np.array([0, 0, 1, 0, 1, 2])
        self.tval1 = np.array([1, 0, 1, 0, 0, 1])
        self.tval2 = np.array([1., 0.5, 1., 0.5, 0.5, 1.])
        self.tchks = (np.ones((3,3)) + np.eye(3)) * 0.5
        self.tsqre = np.array([[0, 1, 2], [1, 4, 7], [2, 7, 8]])
        self.tschk = np.array([0, 1, 4, 2, 7, 8])
        self.sidx0 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        self.sidx1 = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        self.svals = np.array(list(range(9)))
        self.schks = self.svals.reshape(3,3)
        self.schkf = self.svals.reshape(3,3, order='F')

    def test__symmetric_to_square(self):
        """Tests that 1D arrays get reshaped correctly."""
        gss = _symmetric_to_square(self.tidx0, self.tidx1, self.tval1)
        self.assertTrue(np.allclose(gss, np.eye(3)))
        gss = _symmetric_to_square(self.tidx1, self.tidx0, self.tval1)
        self.assertTrue(np.allclose(gss, np.eye(3)))
        gss = _symmetric_to_square(self.tidx0, self.tidx1, self.tval2)
        self.assertTrue(np.allclose(gss, self.tchks))
        gss = _symmetric_to_square(self.tidx1, self.tidx0, self.tval2)
        self.assertTrue(np.allclose(gss, self.tchks))

    def test__symmetric_from_square(self):
        """
        Tests that square arrays get indexed and flattend correctly.
        Test should be a symmetric matrix.
        """
        gsdx, gss = _symmetric_from_square(self.tsqre)
        self.assertTrue(np.allclose(gsdx[:,0], self.tidx0))
        self.assertTrue(np.allclose(gsdx[:,1], self.tidx1))
        self.assertTrue(np.allclose(gsdx[:,2], np.zeros(len(gsdx))))
        self.assertTrue(np.allclose(gss, self.tschk))

    def test__square_to_square(self):
        """Tests that 1D arrays get reshaped correctly."""
        gss = _square_to_square(self.sidx0, self.sidx1, self.svals)
        self.assertTrue(np.allclose(gss, self.schks))
        gss = _square_to_square(self.sidx1, self.sidx0, self.svals)
        self.assertTrue(np.allclose(gss, self.schkf))

    # def test__square_from_square(self):
    #     """Tests that square arrays are flattened and indexed correctly."""
    #     gsdx, gss = _square_from_square(self.schks)
    #     self.assertTrue(np.allclose(gsdx[:,0], self.sidx0))
    #     self.assertTrue(np.allclose(gsdx[:,1], self.sidx1))
    #     self.assertTrue(np.allclose(gsdx[:,2], np.zeros(len(gsdx))))
    #     self.assertTrue(np.allclose(gss, self.svals))
