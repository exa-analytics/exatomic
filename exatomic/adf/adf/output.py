# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Parser for 'A D F' Calculations
######################################
This module provides the main parser (user facing) for parsing 'A D F' output
files. Although there are some natural sections within an 'A D F' output, such
as 'COMPUTATION', 'RESULTS', etc. the specific parsers are not organized in
terms of these sections. Each module within this directory provides a single
parser, specific to a given piece of data.
"""
import re
from exa import Parser, Typed


class Output(Parser):
    """
    Parser for the 'A D F' calculation(s) of an ADF output file.
    """
    _start = re.compile(r"^\s*\*\s*\|\s*A D F\s*\|\s*\*")
    _end = re.compile(r"^\s*A D F   E X I T")
