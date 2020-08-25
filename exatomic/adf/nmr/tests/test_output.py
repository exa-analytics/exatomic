# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
#import numpy as np
#import pandas as pd
#from unittest import TestCase
#from exatomic.base import resource
#from exatomic.adf.nmr.output import Output
#
#class TestOutput(TestCase):
#    def setUp(self):
#        self.pf3 = Output(resource('adf-pf3-nmr.out'))
#        self.c2h2 = Output(resource('adf-c2h2-cpl.out'))
#
#    def test_atom(self):
#        self.pf3.parse_atom()
#        self.assertEqual(self.pf3.atom.shape[0], 4)
#        self.assertTrue(np.all(pd.notnull(self.pf3.atom)))
#        self.c2h2.parse_atom()
#        self.assertEqual(self.c2h2.atom.shape[0], 4)
#        self.assertTrue(np.all(pd.notnull(self.c2h2.atom)))
#
#    def test_nmr_shielding(self):
#        self.pf3.parse_nmr_shielding()
#        self.assertEqual(self.pf3.nmr_shielding.shape[0], 4)
#        self.assertTrue(np.all(pd.notnull(self.pf3.nmr_shielding)))
#
#    def test_j_coupling(self):
#        self.c2h2.parse_j_coupling()
#        self.assertEqual(self.c2h2.j_coupling.shape[0], 4)
#        self.assertTrue(np.all(pd.notnull(self.c2h2.j_coupling)))
#
