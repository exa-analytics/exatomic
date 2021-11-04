# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Tests for :mod:`~exa.units`
#############################################
Basic checks that units have been created.
"""
import numpy as np
from exa.util import units


def test_count():
    """Check that all unit types have been created."""
    assert hasattr(units, "Acceleration") == True
    assert hasattr(units, "Energy") == True
    assert hasattr(units, "Length") == True
    assert hasattr(units, "Time") == True
    assert hasattr(units, "Mass") == True


def test_units():
    """Check attribute values."""
    assert np.isclose(units.Energy['J'], 1.0)
    assert np.isclose(units.Length['au', 'Angstrom'], 0.52918)
    assert np.isclose(units.Length['Angstrom'], 1E-10)

