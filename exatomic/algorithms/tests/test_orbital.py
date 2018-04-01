# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""Tests for computing orbitals, densities and orbital angular momenta."""
import os
import bz2
import numpy as np
from unittest import TestCase

import exatomic
from exatomic import molcas
from exatomic.base import staticdir
from exatomic.molcas import Output as MolOutput
from exatomic.molcas import Orb
from exatomic.core.basis import Overlap
from exatomic.interfaces.cube import Cube
from exatomic.algorithms.orbital_util import compare_fields
from exatomic.algorithms.orbital import (add_molecular_orbitals,
                                         add_orb_ang_mom, add_density)


class TestMolcasOrbital(TestCase):
    def setUp(self):
        adir = os.sep.join([staticdir(), 'molcas'])
        with bz2.open(os.path.join(adir, 'mol-carbon-dz.out.bz2')) as f:
            uni = MolOutput(f.read().decode('utf-8')).to_universe()
        with bz2.open(os.path.join(adir, 'mol-carbon-dz.overlap.bz2')) as f:
            uni.overlap = Overlap.from_column(f.read().decode('utf-8'))
        with bz2.open(os.path.join(adir, 'mol-carbon-dz.scforb.bz2')) as f:
            orb = Orb(f.read().decode('utf-8'))
            uni.momatrix = orb.momatrix
            uni.orbital = orb.orbital
        fls = ['mol-carbon-dz-sodizl-r', 'mol-carbon-dz-sodizl-i',
               'mol-carbon-dz-sodizs-x', 'mol-carbon-dz-sodizs-y',
               'mol-carbon-dz-sodizs-z']
        cols = ['lreal', 'limag', 'sx', 'sy', 'sz']
        for fl, col in zip(fls, cols):
            with bz2.open(os.path.join(adir, fl + '.bz2')) as f:
                orb = Orb(f.read().decode('utf-8'))
                uni.momatrix[col] = orb.momatrix['coef']
                uni.orbital[col] = orb.orbital['occupation']

        flds, cubfmt = [], 'mol-carbon-dz-{}.cube.bz2'.format
        for i, c in enumerate(['1', '2', '3', '4', '5', 'dens',
                               'orb-x', 'orb-y', 'orb-z', 'orb-zz',
                               'spin-x', 'spin-y', 'spin-z']):
            with bz2.open(os.path.join(adir, cubfmt(c))) as f:
                if not i: chk = Cube(f.read().decode('utf-8')).to_universe()
                else: flds.append(Cube(f.read().decode('utf-8')).field)
        chk.add_field(flds)

        kws = {'field_params': chk.field.loc[0], 'verbose': False}
        # DEBUG
        aa, bb, cc = uni.enumerate_shells()
        for i in range(len(cc)):
            print({k: type(getattr(cc[i], k)) for k in dir(cc[i])})
        #return uni, kws
        # /DEBUG
        add_molecular_orbitals(uni, vector=range(5), **kws)
        add_density(uni, mocoefs='coef', **kws)
        add_orb_ang_mom(uni, rcoefs='lreal', icoefs='limag', **kws)
        add_density(uni, mocoefs='sx', **kws)
        add_density(uni, mocoefs='sy', **kws)
        add_density(uni, mocoefs='sz', **kws)
        self.uni = uni
        self.chk = chk

    def test_compare_fields(self):
        res = compare_fields(self.uni, self.chk, verbose=False)
        self.assertEquals(len(res), sum(res))
