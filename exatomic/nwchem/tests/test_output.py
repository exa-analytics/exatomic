# -*- coding: utf-8 -*-
## Copyright (c) 2015-2020, Exa Analytics Development Team
## Distributed under the terms of the Apache License 2.0
"""
Tests for :mod:`~exatomic.nwchem.output`
#############################################
"""
#import numpy as np
#import pandas as pd
from unittest import TestCase
from exatomic.base import resource
from exatomic.nwchem.output import Output


class TestNWChemOutput(TestCase):
    def setUp(self):
        self.mam1 = Output(resource('nw-ch3nh2-631g.out'))
        self.mam2 = Output(resource('nw-ch3nh2-augccpvdz.out'))
        self.nap_roa = Output(resource('nw-naproxen-def2tzvp-roa.out'))
        self.meth_roa = Output(resource('nw-methyloxirane-def2tzvp-roa.out'))

    def test_parse_atom(self):
        self.mam1.parse_atom()
        self.mam2.parse_atom()
        self.assertEqual(self.mam1.atom.shape[0], 7)
        self.assertEqual(self.mam2.atom.shape[0], 7)

    def test_parse_orbital(self):
        self.mam1.parse_orbital()
        self.mam2.parse_orbital()
        self.assertEqual(self.mam1.orbital.shape[0], 28)
        self.assertEqual(self.mam2.orbital.shape[0], 91)

    def test_parse_basis_set(self):
        self.mam1.parse_basis_set()
        self.mam2.parse_basis_set()
        self.assertEqual(self.mam1.basis_set.shape[0], 32)
        self.assertEqual(self.mam2.basis_set.shape[0], 57)


    def test_parse_basis_set_order(self):
        self.mam1.parse_basis_set_order()
        self.mam2.parse_basis_set_order()
        self.assertEqual(self.mam1.basis_set_order.shape[0], 28)
        self.assertEqual(self.mam2.basis_set_order.shape[0], 91)

    def test_parse_frame(self):
        self.mam1.parse_frame()
        self.mam2.parse_frame()
        self.assertEqual(self.mam1.frame.shape[0], 1)
        self.assertEqual(self.mam2.frame.shape[0], 1)

    def test_parse_momatrix(self):
        self.mam1.parse_momatrix()
        self.mam2.parse_momatrix()
        self.assertEqual(self.mam1.momatrix.shape[0], 784)
        self.assertEqual(self.mam2.momatrix.shape[0], 8281)

    def test_parse_roa(self):
        self.nap_roa.parse_roa()
        self.assertEqual(self.nap_roa.roa.shape[0], 10)
        self.meth_roa.parse_roa()
        self.assertEqual(self.meth_roa.roa.shape[0], 10)

    def test_parse_gradient(self):
        self.nap_roa.parse_gradient()
        self.assertEqual(self.nap_roa.gradient.shape[0], 31)
        self.meth_roa.parse_gradient()
        self.assertEqual(self.meth_roa.gradient.shape[0], 10)

    def test_to_universe(self):
        self.mam1.to_universe()
        self.mam2.to_universe()
