# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Tests for Physical Constants
#############################################
Physical constants are created using the factory paradigm.
"""
from exa.util import constants


def test_created():
    """Check that constants were created."""
    assert len(dir(constants)) > 300
    assert hasattr(constants, "Planck_constant") == True


def test_attrs():
    """Check attributes of constants."""
    assert hasattr(constants.Planck_constant, "value")
    assert hasattr(constants.Planck_constant, "units")
    assert hasattr(constants.Planck_constant, "name")
    assert hasattr(constants.Planck_constant, "error")

