# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Unit Conversions
########################################
Values are reported with respect to the base SI unit for a given quantity.
Conversion factors can be generated using the syntax, Quantity[from, to];
see the example below.

.. code-block:: python

    from exa.util.units import Energy
    Energy["eV"]         # Value of eV in SI units
    Energy["eV", "J"]    # Same as above
    Energy["eV", "Ha"]   # Conversion factor between eV and Ha (Hartree atomic unit)
"""
import bz2 as _bz2
import json as _json
import os as _os
import sys as _sys
import numpy as _np
import pandas as _pd
if not hasattr(_bz2, "open"):
    _bz2.open = _bz2.BZ2File


class Unit(object):
    @property
    def values(self):
        return self._values

    def __setitem__(self, key, value):
        self._values[key] = value

    def __getitem__(self, key):
        if isinstance(key, str):
            k = self._values[_np.isclose(self._values, 1.0)].index[0]
            return self._values[k]/self._values[key]
        elif isinstance(key, (list, tuple)):
            return self._values[key[1]]/self._values[key[0]]

    def __init__(self, values, name):
        self._values = _pd.Series(values)
        self._name = name


def _create():
    def creator(name, data):
        _ = data.pop("dimensions", None)
        _ = data.pop("aliases", None)
        return Unit(data, name)

    with _bz2.open(_path, "rb") as f:
        dct = _json.loads(f.read().decode("utf-8"))
    for name, data in dct.items():
        setattr(_this, name.title(), creator(name, data))


_resource = "../../static/units.json.bz2"    # HARDCODED
_this = _sys.modules[__name__]         # Reference to this module
_path = _os.path.abspath(_os.path.join(_os.path.abspath(__file__), _resource))
if not hasattr(_this, "Energy"):
    _create()

