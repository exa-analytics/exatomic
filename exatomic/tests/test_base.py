# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
from exatomic.base import sym2isomass
from unittest import TestCase

class TestBase(TestCase):
    def test_sym2isomass(self):
        symbol = ['Ni']
        expected = [57.93534241]
        close = []
        dict = sym2isomass(symbol)
        for symb, exp in zip(symbol, expected):
            close.append(abs(dict[symb] - exp) < 1e-6)
        self.assertTrue(all(close))
        symbol = ['Ni', 'H', 'C']
        expected = [57.93534241, 1.00782503223, 12.0000]
        close = []
        dict = sym2isomass(symbol)
        for symb, exp in zip(symbol, expected):
            close.append(abs(dict[symb] - exp) < 1e-6)
        self.assertTrue(all(close))
        
#import pytest
#
#params = [(['Ni'], [57.93534241]),
#          (['Ni', 'H', 'C'], [57.93534241, 1.00782503223, 12.0000])]
#
#@pytest.mark.parametrize("symbol, expected", params)
#def test_sym2isomass(symbol, expected):
#    dict = sym2isomass(symbol)
#    close = []
#    for symb, exp in zip(symbol, expected):
#        close.append(abs(dict[symb] - exp) < 1e-6)
#    assert all(close)

