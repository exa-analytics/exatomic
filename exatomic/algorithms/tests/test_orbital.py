# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""Tests for computing orbitals, densities and orbital angular momenta."""
import numpy as np
from unittest import TestCase
from exatomic import Universe, nwchem, molcas
from exatomic.base import resource
from exatomic.algorithms.orbital_util import compare_fields
from exatomic.algorithms.orbital import (add_molecular_orbitals,
                                         add_orb_ang_mom,
                                         add_density)


class TestMolcasOrbital(TestCase):

    def setUp(self):
        self.chk = Universe.load(resource('mol-carbon-dz-valid.hdf5'))
        kws = {'field_params': self.chk.field.loc[0], 'verbose': False}
        uni = Universe.load(resource('mol-carbon-dz.hdf5'))
        add_molecular_orbitals(uni, vector=range(5), **kws)
        add_density(uni, mocoefs='coef', **kws)
        add_orb_ang_mom(uni, rcoefs='lreal', icoefs='limag', **kws)
        add_density(uni, mocoefs='sx', **kws)
        add_density(uni, mocoefs='sy', **kws)
        add_density(uni, mocoefs='sz', **kws)
        self.uni = uni


    def test_compare_fields(self):
        res = compare_fields(self.uni, self.chk, verbose=False)
        self.assertTrue(np.isclose(len(res), sum(res), rtol=5e-4))



class TestADFOrbital(TestCase):

    def test_compare_fields(self):
        chk = Universe.load(resource('adf-lu-valid.hdf5'))
        uni = Universe.load(resource('adf-lu.hdf5'))
        uni.add_molecular_orbitals(vector=range(8, 60), verbose=False,
                                   field_params=chk.field.loc[0])
        res = compare_fields(chk, uni, signed=False, verbose=False)
        self.assertTrue(np.isclose(len(res), sum(res), rtol=5e-4))


class TestNWChemOrbital(TestCase):

    def test_compare_fields(self):
        nw = nwchem.Output(resource('nw-ch3nh2-631g.out')).to_universe()
        mo = molcas.Output(resource('mol-ch3nh2-631g.out'))
        mo.add_orb(resource('mol-ch3nh2-631g.scforb'))
        mo = mo.to_universe()
        nw.add_molecular_orbitals(vector=range(3, 10), verbose=False)
        mo.add_molecular_orbitals(vector=range(3, 10), verbose=False)
        res = compare_fields(nw, mo, signed=False, rtol=5e-3)
        self.assertTrue(np.isclose(sum(res), len(res), rtol=5e-3))
