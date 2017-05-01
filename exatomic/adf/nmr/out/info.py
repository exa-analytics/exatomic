# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
NMR InfoParser Subsection
######################
The info subsection contains nuclear coordinates, basis information, and other
calculation information.
"""
import six
from exa.core import Meta, Sections, Parser
from exatomic.atom import Atom


class InfoParserMeta(Meta):
    """Defines the objects parsed by NMRInfoParserParser."""
    atom = Atom
    _descriptions = {}


class InfoParser(six.with_metaclass(InfoParserMeta, Sections)):
    """
    Subsection of an 'N M R' region containing nuclear coordinates, basis,
    scaling and calculation information, etc.
    """
    name = "info"
    description = "Subsection containing nuclear coordinates, basis, etc. of an NMR output."
    _key_sep = "[<>]+$"
    _key_sections = ["geninfo", "atom", "basis", "tensor", "blank", "switches", "potential"]

    def _parse(self):
        delims = self.regex(self._key_sep, text=False)[self._key_sep]
        ends = delims[1:]
        ends.append(len(self))
        dct = {'start': delims, 'parser': self._key_sections, 'end': ends}
        self._sections_helper(dct)


class AtomParserMeta(Meta):
    """Parser for the atom subsection of the info subsection of an NMR output."""
    atom = Atom
    _descriptions = {'atom': "Table of nuclear coordinates"}


class AtomParser(six.with_metaclass(AtomParserMeta, Parser)):
    """Parser for the atom table in an NMR output."""
    name = "atom"
    description = "Nuclear coordinate table from NMR output"
    _key_widths = (18, 8, 13, 13, 13)
    _key_names = ("symbol", "label", "x", "y", "z")
    _key_str0 = ("(", "")
    _key_str1 = (")", "")
    _key_str2 = (":", "")
    _key_frame = ("frame", 0)

    def _parse(self):
        """Parse the atom table."""
        atom = self[6:-4].to_data("fwf", widths=self._key_widths, names=self._key_names)
        col = self._key_names[0]
        atom[col] = atom[col].str.strip()
        col = self._key_names[1]
        atom[col] = atom[col].str.replace(*self._key_str0)
        atom[col] = atom[col].str.replace(*self._key_str1)
        atom[col] = atom[col].str.replace(*self._key_str2)
        atom[col] = atom[col].astype(int)
        atom[self._key_frame[0]] = self._key_frame[1]
        self.atom = atom


InfoParser.add_section_parsers(AtomParser)
