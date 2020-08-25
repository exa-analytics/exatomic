# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
#import os
#from unittest import TestCase
#from exatomic import Universe
#from exatomic.gaussian import Output, Input
#from exatomic.gaussian.inputs import _handle_args


#class TestInput(TestCase):
#    """Tests the input file generation functionality for Gaussian."""
#    pass
#    def setUp(self):
#        fl = Output(os.sep.join(__file__.split(os.sep)[:-1]
#                                + ['gaussian-uo2.out']))
#        self.uni = Universe(atom=fl.atom)
#        self.keys = ['link0', 'route', 'basis', 'ecp']
#        self.lisopt = [('key1', 'value1'), ('key2', 'value2')]
#        self.dicopt = {'key1': 'value1', 'key2': 'value2'}
#        self.tupopt = (('key1', 'value1'), ('key2', 'value2'))
#        self.stropt = 'value'
#
#    def test_from_universe(self):
#        """Test the from_universe class method for input generation."""
#        fl = Input.from_universe(self.uni, link0=self.lisopt,
#                                 route=self.dicopt, basis=self.tupopt)
#        self.assertEqual(fl[0][0], '%')
#        self.assertEqual(fl[2][0], '#')
#        self.assertEqual(len(fl.find('****')), 2)
#        self.assertEqual(len(fl), 18)
#
#    def test__handle_args(self):
#        """Test the argument handler helper function."""
#        for key in self.keys:
#            lval = _handle_args(key, self.lisopt)
#            self.assertEqual(lval, _handle_args(key, self.tupopt))
#            self.assertEqual(self.stropt, _handle_args(key, self.stropt))
