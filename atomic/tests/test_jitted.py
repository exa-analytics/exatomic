# -*- coding: utf-8 -*-

# Hacky imports
import sys
sys.path.insert(0, '/home/tjd/Programs/analytics-exa/atomic')
sys.path.insert(0, '/home/tjd/Programs/analytics-exa/exa')

from exa.testers import UnitTester
from atomic import _np as np
from atomic.algorithms.jitted import expand

class TestJExpand(UnitTester):

    def test_expand(self):
        a = np.arange(0, 100, 5) + 2
        b = np.array([3] * 20)
        fidx, oidx, expd = expand(a, b)
        self.assertEqual(fidx.sum(), 570)
        self.assertEqual(oidx.sum(), 60)
        self.assertEqual(expd.sum(), 3030)
