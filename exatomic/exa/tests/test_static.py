# -*- coding: utf-8 -*-
# Copyright (c) 2015-2022, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Tests for :mod:`~exatomic.exa.static`
#############################################
"""
import os
from unittest import TestCase
from exatomic.exa import static


def test_static_dir():
    """Test :func:`~exatomic.exa.static.staticdir`."""
    assert os.path.isdir(static.staticdir())
    return True


def test_resource():
    assert os.path.exists(static.resource("units.json.bz2"))
    return True


class Tester(TestCase):
    def test_static_dir(self):
        self.assertTrue(test_static_dir())

    def test_resource(self):
        self.assertTrue(test_resource())
