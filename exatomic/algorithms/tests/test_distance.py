# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Two Body Properties Computations
#####################################
"""
import numpy as np
from unittest import TestCase
from exatomic.algorithms.distance import cartmag


class Test3DOperations(TestCase):
    def test_mag(self):
        """
        Uses :func:`~numpy.allclose` with default tolerances for checking equality.
        """
        n = 5
        x = np.random.rand(n)
        y = np.random.rand(n)
        z = np.random.rand(n)
        check = (x**2 + y**2 + z**2)**0.5
        result = cartmag(x, y, z)
        self.assertTrue(np.allclose(check, result))
