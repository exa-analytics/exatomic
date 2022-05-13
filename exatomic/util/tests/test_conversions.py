# -*- coding: utf-8 -*-
# Copyright (c) 2015-2022, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
import pandas as pd
import numpy as np
import pytest

from exatomic.base import resource
from exatomic.util import conversions

@pytest.mark.parametrize('actual,test',
                         [(6.241509074e18, 'J2eV'),
                          (27.211386245988, 'Ha2eV'),
                          (2.1947463136320e7, 'Ha2inv_m'),
                          (2.1947463136320e5, 'Ha2inv_cm'),
                          (6.0221407621e26, 'Kg2u')])
def test_conversion(actual, test):
    test_value = getattr(conversions, test)
    assert np.isclose(test_value.value, actual)

