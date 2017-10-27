# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Tests for :mod:`~exatomic.xyz`
################################
"""
from unittest import TestCase
from exatomic.base import resource
from exatomic.xyz import XYZ


check = {'H2O.xyz.bz2': 3, 'H2.xyz.bz2': 2, 'O2.xyz.bz2': 2, 'UCl6.xyz.bz2': 7,
         'UO2.xyz.bz2': 3, 'ZnPorphyrin.xyz.bz2': 77, 'ZrO2.xyz.bz2': 3,
         'H2O.traj.xyz.bz2': 225600, 'benzene.xyz.bz2': 12, 'CH3NH2.xyz.bz2': 7,
         'methyl_134_triazole.xyz.bz2': 11, 'kaolinite.xyz.bz2': 26,
         'magnesite.xyz.bz2': 60, 'zabuyelite.xyz.bz2': 24}


class TestXYZ(TestCase):
    """
    Test that the parser opens a wide range of files and correctly parses the
    atom table.
    """
    def test_sizes(self):
        for name, size in check.items():
            self.assertEqual(XYZ(resource(name)).atom.shape[0], size)
