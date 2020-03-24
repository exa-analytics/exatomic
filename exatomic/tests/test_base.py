# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
from exatomic.base import sym2isomass
import pytest

params = [(['Ni'], [57.93534241]),
          (['Ni', 'H', 'C'], [57.93534241, 1.00782503223, 12.0000])]

@pytest.mark.parametrize("symbol, expected", params)
def test_sym2isomass(symbol, expected):
    dict = sym2isomass(symbol)
    close = []
    for symb, exp in zip(symbol, expected):
        close.append(abs(dict[symb] - exp) < 1e-6)
    assert all(close)

