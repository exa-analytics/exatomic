# -*- coding: utf-8 -*-
'''
Tests for :mod:`~exatomic.pdb`
===============================
'''
from requests import get as _get
from exa.testers import UnitTester
from exa.utils import mkpath
from exatomic import _np as np
from exatomic import _os as os
from exatomic import pdb


_selfpath = os.path.dirname(os.path.realpath(__file__))
_pdbpath = mkpath(_selfpath, 'static', '3nir.pdb')


class TestReadPDB(UnitTester):
    def setUp(self):
        self.lp = _pdbpath
        self.rp = _pdbpath.split(os.sep)[-1].replace('.pdb', '')
        self.rf = pdb._path_handler(self.rp)
        self.ret = pdb.read_pdb(self.lp)
        self.lf = self.ret['metadata']['text']

    def test__remote_path(self):
        gd = pdb._remote_path(self.rp)
        self.assertTrue(type(gd) is list)
        with self.assertRaises(FileNotFoundError):
            bd = pdb._remote_path('XXXX')

    def test__local_path(self):
        tlp = pdb._local_path(self.lp)
        self.assertEqual(len(tlp), 2043)

    def test__path_handler(self):
        self.assertTrue(len(self.rp) == 4)
        self.assertEqual(
            ' '.join(self.lf).split(),
            ' '.join(self.rf).split()
        )

    def test__pre_process_pdb(self):
        trds = pdb._pre_process_pdb(self.lf)
        self.assertEqual(trds['nrf'], 1)
        self.assertEqual(trds['nat'], 892)

    def test_read_pdb(self):
        self.assertTrue('frame' in self.ret)
        self.assertTrue('one' in self.ret)
        self.assertTrue('metadata' in self.ret)
        self.assertEqual(self.ret['one'].shape, (892, 5))
