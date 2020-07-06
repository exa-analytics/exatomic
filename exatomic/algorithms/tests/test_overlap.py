# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""Tests for computing the overlap."""
import numpy as np
from unittest import TestCase
from exatomic.base import resource
from exatomic.core.basis import Overlap
from exatomic.molcas import Output as MolOutput


class TestMolcasOverlap(TestCase):
    def setUp(self):
        dz = MolOutput(resource('mol-carbon-dz.out'))
        dz.add_overlap(resource('mol-carbon-dz.overlap'))

        li = MolOutput(resource('mol-li-ano.out'))
        li.add_overlap(resource('mol-li-ano.overlap'))

        zno2 = MolOutput(resource('mol-zno2-dz.out'))
        zno2.add_overlap(resource('mol-zno2-dz.overlap'))

        npo2 = MolOutput(resource('mol-npo2-ano.out'))
        npo2.add_overlap(resource('mol-npo2-ano.overlap'))

        self.unis = [dz.to_universe(), li.to_universe(),
                     zno2.to_universe(), npo2.to_universe()]

    def test_overlap(self):
        for uni in self.unis:
            ovls = uni.basis_functions.integrals()
            self.assertTrue(isinstance(ovls, Overlap))
            ovls = ovls.square().values
            n = np.isclose(ovls, uni.overlap.square().values,
                           rtol=5e-5, atol=1e-12).sum() \
                / (ovls.shape[0] * ovls.shape[1])
            self.assertTrue(n > 0.999)
