# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Composite QE Output
#######################
Since a number of executables that are part of the Quantum ESPRESSO suite of
quantum chemistry tools share the same format, this module provides a number
of base parsers.
"""
import re
from exa import Parser


class Output(Parser):
    """
    The base parser for QE output files.
    """
    pass


class AtomPos(Parser):
    """
    Generic parser for ATOMIC_POSITIONS sections of output files.
    """
    _start = re.compile("^ATOMIC_POSITIONS", re.MULTILINE)
    _stop = 0

    def _parse(self):
        pass


class QESCF(Parser):
    """
    Generic parser for SCF loop sections of output files.
    """
    _start = "Self-consistent Calculation"
    _stop = re.compile("^\s*End of self-consistent calculation$", re.MULTILINE)


Output.add_parsers(AtomPos, QESCF)
