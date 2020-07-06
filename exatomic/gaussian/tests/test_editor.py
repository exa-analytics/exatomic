# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
from unittest import TestCase
from exatomic.gaussian.editor import Editor


class TestEditor(TestCase):
    """Tests that metadata is set appropriately for Gaussian editors."""

    def test_no_meta(self):
        """Test that program metadata is set by default."""
        fl = Editor('', ignore=True)
        self.assertTrue(fl.meta['program'] == 'gaussian')

    def test_with_meta(self):
        """Test that passed metadata is respected and program is set."""
        fl = Editor('', meta={'meta': 'data'}, ignore=True)
        self.assertEqual(fl.meta['meta'], 'data')
        self.assertEqual(fl.meta['program'], 'gaussian')
