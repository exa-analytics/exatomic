# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
All-Electron Output Editors
##############################
"""
import six
from exa.special import LazyFunction
from exa.tex import text_value_cleaner
from exa.core import DataFrame, SectionsMeta, Parser
from exatomic.nwchem.pseudopotentials.ps import PAWOut


class AEOutMeta(SectionsMeta):
    """Defines data objects for AEOutput."""
    info = dict
    data = DataFrame
    grid = LazyFunction
    _descriptions = {'aedata': "All electron data",
                     'info': "Energy data",
                     'grid': "Grid information"}

class AEOut(six.with_metaclass(AEOutMeta, Parser)):
    """
    """
    _key_delim0 = "******"
    _key_delim1 = ":"
    _key_delim2 = "="
    description = "Parser for 'El_out' NWChem output files (El is an element sybol)."

    def _parse(self):
        """
        """
        key0, value0 = str(self[5]).split(self._key_delim1)
        key1, value1 = str(self[6]).split(self._key_delim1)
        self.info = {key0.strip(): text_value_cleaner(value0),
                     key1.strip(): text_value_cleaner(value1)}
        sections = self.find(self._key_delim0, which='lineno')[self._key_delim0]
        self.grid = PAWOut(self[:sections[0]]).grid
        self.data = self[sections[0]+4:sections[1]-2].to_data(delim_whitespace=True)
        self.data['l'] = self.data['l'].str.upper()
        for line in self[sections[1]:]:
            if self._key_delim2 in line:
                key, value = line.split(self._key_delim2)
                self.info[key.strip()] = text_value_cleaner(value)
