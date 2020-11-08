# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
import numpy as np
from unittest import TestCase

from exatomic.base import resource
from exatomic import gaussian

class TestTwo(TestCase):
    def setUp(self):
        ed = gaussian.Output(resource('g16-methyloxirane-def2tzvp-opt.out'))
        self.methyl = ed.to_universe()

    def test_get_bond(self):
        bonds = self.methyl.atom_two.get_bond(0, 1, unit='Angstrom')
        expected = [1.424108, 1.442680, 1.454052, 1.455723, 1.455893,
                    1.455827, 1.455813, 1.455824, 1.455838, 1.455841,
                    1.455839, 1.455837]
        assert np.allclose(bonds['dr'].values, expected, atol=1e-6)

