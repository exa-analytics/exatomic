# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Tests for :mod:`~exatomic.xyz`
################################
"""
from unittest import TestCase
from exatomic.base import resource
from exatomic import XYZ


check = {'H2O.xyz': 3, 'H2.xyz': 2, 'O2.xyz': 2, 'UCl6.xyz': 7,
         'UO2.xyz': 3, 'ZnPorphyrin.xyz': 77, 'ZrO2.xyz': 3,
         'H2O.traj.xyz': 225600, 'benzene.xyz': 12, 'CH3NH2.xyz': 7,
         'methyl_134_triazole.xyz': 11, 'kaolinite.xyz': 26,
         'magnesite.xyz': 60, 'zabuyelite.xyz': 24}


class TestXYZ(TestCase):
    """
    Test that the parser opens a wide range of files and correctly parses the
    atom table.
    """
    def test_sizes(self):
        for name, size in check.items():
            xyz = XYZ(resource(name))
            xyz.parse_atom()
            self.assertEqual(xyz.atom.shape[0], size)
