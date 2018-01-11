# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Test :mod:`~exatomic.nwchem.scf.output`
########################################
In testing the comprehensive output parser we test the core parsers application
to NWChem's HF module (task called 'scf').
"""
from unittest import TestCase
from exa.core.parser import Sections
from exatomic.base import resource
from exatomic.nwchem.scf.output import Output


testfiles = ("Dimer.out.bz2", "He-prjbas.out.bz2", "heme6a1-frag.out.bz2")


class TestHFOutput(TestCase):
    """
    """
    def setUp(self):
        """Open all files to test."""
        self.outputs = {}
        for testfile in testfiles:
            self.outputs[testfile] = Output(resource(testfile))

    def test_parse_sections(self):
        """Test that parsing sections works."""
        for name, ed in self.outputs.items():
            with self.subTest(name=name):
                self.assertIsInstance(ed.sections, Sections)
