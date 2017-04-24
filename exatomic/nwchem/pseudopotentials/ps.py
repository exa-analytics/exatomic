# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Pseudized Output Editors
##########################
"""
import six
import numpy as np
import pandas as pd
from exa.special import LazyFunction
from exa.core import DataFrame, SectionsMeta, Parser
from exatomic.nwchem.pseudopotentials.grid import scaled_logspace


class PAWOutMeta(SectionsMeta):
    """Defines data objects for PAWOutput."""
    data = DataFrame
    info = dict
    grid = LazyFunction
    _descriptions = {'psdata': "Pseudized channel data",
                     'info': "Atom, charge, and core info",
                     'grid': "Grid information"}


class PAWOut(six.with_metaclass(PAWOutMeta, Parser)):
    """
    Parser for the ``El_paw`` output file (El is an element symbol).

    See Also:
        AEOutput
    """
    _key_delims = (" Atom information :", " Logarithmic grid information ( r(i)=r0*pow(a,i) )",
                   " Compensation charge information :", " Paw potential information :",
                   " Basis information :", " Core information :")
    description = "Parser for 'El_paw' NWChem output files (El is an element sybol)."

    def _parse(self):
        """
        """
        self.info = {}
        self.data = DataFrame()
        sections = self.find(*self._key_delims, which='lineno')
        for section, start in sections.items():
            if section == " Paw potential information :":
                self._key_subsection_parsers[section](start[0]+1, delnl=True)
            else:
                self._key_subsection_parsers[section](start[0]+1)

    def _parse_info_like(self, start, save=True):
        """
        """
        sep = ':' if ':' in self[start+1] else "="
        k = 0
        dct = {}
        for line in self[start:]:
            if sep not in line:
                k += 1
                if k == 2:
                    break
            else:
                key, value = line.split(sep)
                dct[key.strip()] = text_value_cleaner(value)
        if save:
            self.info.update(dct)
        else:
            return dct

    def _parse_grid(self, start):
        """
        """
        grid = self._parse_info_like(start, save=False)
        self.grid = LazyFunction(scaled_logspace, grid['r0'], grid['a'], grid['N'])

    def _parse_paw(self, start, delnl=False):
        """
        """
        m = len(self.info)
        self._parse_info_like(start)
        n = len(self.info)
        k = start + n - m + 2
        l = 0
        for line in self[k:]:
            if line == "":
                break
            l += 1
        df = self[k:k+l].to_data(delim_whitespace=True)
        if delnl:
            del df['nl']
        self._merge_df(df)

    def _merge_df(self, df):
        if len(self.data) == 0:
            self.data = df
        else:
            self.data = pd.concat((self.data, df), axis=1)
        if 'nl' in self.data.columns:
            self.data['nl'] = self.data['nl'].str.upper()

    def __init__(self, *args, **kwargs):
        super(PAWOut, self).__init__(*args, **kwargs)
        self._key_parsers = (self._parse_info_like, self._parse_grid,
                             self._parse_info_like, self._parse_paw,
                             self._parse_paw, self._parse_info_like)
        self._key_subsection_parsers = dict(zip(self._key_delims, self._key_parsers))
