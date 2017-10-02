# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Tests for :mod:`~exatomic.interfaces.xyz`
############################################
"""
import os
from unittest import TestCase
from exatomic.interfaces.xyz import XYZ


class TestXYZ(TestCase):
    def setUp(self):
        """Determine path to static data."""
        self.xyz = os.path.abspath(os.path.join(os.path.abspath(__file__),
                                                "../../../static/xyz/"))
        self.assertTrue(os.path.exists(self.xyz))

    def test_trajectory(self):
        pass
