# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0

try:
    from exa.test.tester import UnitTester
except:
    from exa.tester import UnitTester
from exatomic.molcas.editor import Editor


class TestEditor(UnitTester):
    """Tests that metadata is set appropriately for Molcas editors."""

    def test_no_meta(self):
        """Test that program metadata is set by default."""
        fl = Editor('')
        self.assertTrue(fl.meta['program'] == 'molcas')

    def test_with_meta(self):
        """Test that passed metadata is respected and program is set."""
        fl = Editor('', meta={'meta': 'data'})
        self.assertEqual(fl.meta['meta'], 'data')
        self.assertEqual(fl.meta['program'], 'molcas')
