# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""Tests for computing the overlap."""
import os
import bz2
import numpy as np

from unittest import TestCase

import exatomic
from exatomic.base import staticdir
from exatomic.core.basis import Overlap
from exatomic.molcas import Output as MolOutput


class TestMolcasOverlap(TestCase):
    def setUp(self):
        adir = os.sep.join([staticdir(), 'molcas'])
        with bz2.open(os.path.join(adir, 'mol-carbon-dz.out.bz2')) as f:
            dz = MolOutput(f.read().decode('utf-8')).to_universe()
        with bz2.open(os.path.join(adir, 'mol-carbon-dz.overlap.bz2')) as f:
            dz.overlap = Overlap.from_column(f.read().decode('utf-8'))

        with bz2.open(os.path.join(adir, 'mol-li-ano.out.bz2')) as f:
            li = MolOutput(f.read().decode('utf-8')).to_universe()
        with bz2.open(os.path.join(adir, 'mol-li-ano.overlap.bz2')) as f:
            li.overlap = Overlap.from_column(f.read().decode('utf-8'))

        with bz2.open(os.path.join(adir, 'mol-zno2-dz.out.bz2')) as f:
            zno2 = MolOutput(f.read().decode('utf-8')).to_universe()
        with bz2.open(os.path.join(adir, 'mol-zno2-dz.overlap.bz2')) as f:
            zno2.overlap = Overlap.from_column(f.read().decode('utf-8'))

        with bz2.open(os.path.join(adir, 'mol-npo2-ano.out.bz2')) as f:
            npo2 = MolOutput(f.read().decode('utf-8')).to_universe()
        with bz2.open(os.path.join(adir, 'mol-npo2-ano.overlap.bz2')) as f:
            npo2.overlap = Overlap.from_column(f.read().decode('utf-8'))
        self.unis = [dz, li, zno2, npo2]

    def test_overlap(self):
        for uni in self.unis:
            ovls = uni.basis_functions.integrals().square().values.astype(np.float64)
            n = np.isclose(ovls, uni.overlap.square().values,
                           rtol=5e-5, atol=1e-12).sum() \
                / (ovls.shape[0] * ovls.shape[1])
            self.assertTrue(n > 0.999)
