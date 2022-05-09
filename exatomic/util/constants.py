# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Physical Constants
#######################################
Tabulated physical constants from `NIST`_. Note that all constants are float
objects (with a slightly modified repr). This means that math operations can
be performed with them directly. Note that units and uncertainty are included
for each value.

.. code-block:: python

    constants.Planck_constant         # Planck_constant(6.62607004e-34 +/-8.1e-42)
    constants.Planck_constant.unit    # J s
    constants.Planck_constant.error   # 9.1e-42

.. _NIST: https://www.nist.gov/
"""
import sys as _sys
import pandas as _pd
from exatomic.base import resource as _resource


class Constant(float):
    """
    Physical constant with value, units, and uncertainty.

    .. code-block:: python

        constants.Planck_constant         # Planck_constant(6.62607004e-34 ±8.1e-42 / J s)
        constants.Planck_constant.unit    # J s
        constants.Planck_constant.error   # 9.1e-42
    """
    def __new__(cls, name, units, value, error):
        return super(Constant, cls).__new__(cls, value)

    def __init__(self, name, units, value, error):
        float.__init__(value)
        self.name = name
        self.units = units
        self.error = error
        self.value = value

    def __repr__(self):
        if self.units.strip() != '':
            return "{}({} ±{} / {})".format(self.name, self.value, self.error, self.units)
        else:
            return "{}({} ±{})".format(self.name, self.value, self.error)


def _create():
    df = _pd.read_csv(_path, compression='xz')
    for quan, unit, err, val in zip(df['quantity'], df['unit'],
                                    df['uncertainty'], df['value']):
        setattr(_this, quan, Constant(name=quan, units=unit, value=val,
                                      error=err))

_this = _sys.modules[__name__]
_path = _resource("nist-constants-2018.csv.xz")
if not hasattr(_this, "Planck_constant"):
    _create()

