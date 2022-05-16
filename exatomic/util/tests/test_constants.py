# -*- coding: utf-8 -*-
# Copyright (c) 2015-2022, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
import pandas as pd
import numpy as np
import pytest

from exatomic.base import resource
from exatomic.util import constants

@pytest.mark.parametrize('actual,test',
                         [(1.66053906660e-27, 'atomic_mass_constant'),
                          (6.62607015e-34, 'Planck_constant'),
                          (1.380649e-23, 'Boltzmann_constant'),
                          (9.1093837015e-31, 'electron_mass'),
                          (1.602176634e-19, 'elementary_charge')])
def test_constants(actual, test):
    test_value = getattr(constants, test)
    assert np.isclose(test_value.value, actual)

