# -*- coding: utf-8 -*-
'''
Tests for :mod:`~exatomic.algorithms.jitted`
=============================================
'''
from exa.testers import UnitTester
from exatomic import _np as np
from exatomic.algorithms.jitted import expand


class TestJitted(UnitTester):
    '''
    '''
    def test_expand(self):
        a = np.arange(0, 100, 5) + 2
        b = np.array([3] * 20)
        fidx, oidx, expd = expand(a, b)
        self.assertEqual(fidx.sum(), 570)
        self.assertEqual(oidx.sum(), 60)
        self.assertEqual(expd.sum(), 3030)
