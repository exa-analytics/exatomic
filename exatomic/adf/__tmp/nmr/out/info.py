## -*- coding: utf-8 -*-
## Copyright (c) 2015-2017, Exa Analytics Development Team
## Distributed under the terms of the Apache License 2.0
#"""
#NMR InfoParser Subsection
#######################
#The info subsection contains nuclear coordinates, basis information, and other
#calculation information.
#"""
#from exa.typed import TypedProperty
#from exa.core.parser import Sections, Parser
#from exatomic.atom import Atom
#
#
#class AtomParser(Parser):
#    """Parser for the atom table in an NMR output."""
#    _key_widths = (18, 8, 13, 13, 13)
#    _key_names = ("symbol", "label", "x", "y", "z")
#    _key_str0 = ("(", "")
#    _key_str1 = (")", "")
#    _key_str2 = (":", "")
#    _key_frame = ("frame", 0)
#    atom = TypedProperty(Atom)
#
#    def _parse(self):
#        """Parse the atom table."""
#        atom = self[6:-4].to_data("fwf", widths=self._key_widths, names=self._key_names)
#        col = self._key_names[0]
#        atom[col] = atom[col].str.strip()
#        col = self._key_names[1]
#        atom[col] = atom[col].str.replace(*self._key_str0)
#        atom[col] = atom[col].str.replace(*self._key_str1)
#        atom[col] = atom[col].str.replace(*self._key_str2)
#        atom[col] = atom[col].astype(int)
#        atom[self._key_frame[0]] = self._key_frame[1]
#        self.atom = atom
#
#
#class InfoParser(Sections):
#    """
#    Subsection of an 'N M R' region containing nuclear coordinates, basis,
#    scaling and calculation information, etc.
#    """
#    _key_sep = "[<>]+$"
#    _key_sections = ["geninfo", AtomParser, "basis", "tensor", "blank", "switches", "potential"]
#    atom = TypedProperty(Atom)
#
#    def _parse(self):
#        delims = self.regex(self._key_sep, text=False)[self._key_sep]
#        ends = delims[1:]
#        ends.append(len(self))
#        self._sections_helper(start=delims, parser=self._key_sections, end=ends)
