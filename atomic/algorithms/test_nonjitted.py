# -*- coding: utf-8 -*-

import unittest
import numpy as np
from nonjitted import expand

class TestExpand(unittest.TestCase):

    def test_expand(self):
        a = np.arange(0, 100, 5) + 2
        b = np.array([3] * 20)
        fidx, oidx, expd = expand(a, b)
        self.assertEqual(fidx.sum(), 570)
        self.assertEqual(oidx.sum(), 60)
        self.assertEqual(expd.sum(), 3030)
