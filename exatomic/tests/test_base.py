# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
from exatomic.base import sym2isomass
import pytest

params = [(['Ni'], None, [57.93534241]),
          (['Ni', 'H', 'C'], None, [57.93534241, 1.00782503223, 12.0000]),
          (['Ni', 'H', 'C'], [64, 2, 13], [63.92796682, 2.01410177812, 13.00335483507])]

@pytest.mark.parametrize("symbol, isotope, expected", params)
def test_sym2isomass(symbol, isotope, expected):
    dict = sym2isomass(symbol, isotope)
    close = []
    for symb, exp in zip(symbol, expected):
        close.append(abs(dict[symb] - exp) < 1e-6)
    assert all(close)

