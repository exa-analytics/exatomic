# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
'''
exnbo Output Editor
####################################
'''
import re
import numpy as np
import pandas as pd
from io import StringIO

from exa.relational.isotope import symbol_to_z, z_to_symbol

from exa import Editor
from exatomic import Length
from exatomic.basis import GaussianBasisSet

csv_args = {'delim_whitespace': True}

class Output(Editor):

    def parse_nao(self):
        found = self.find(_re_nao_start, _re_nao_stop01)
        regex = self.regex(_re_nao_stop02, _re_nao_stop03, keys_only=True)
        starts = [i[0] + 2 for i in found[_re_nao_start]]
        stops = [i[0] for i in found[_re_nao_stop01]]
        if regex[_re_nao_stop03]:
            stops = regex[_re_nao_stop03]
        elif regex[_re_nao_stop02]:
            keys = regex[_re_nao_stop02]
            stops = [keys[0]] + keys[2::2]
        dfs = []
        spins = [-1, 0, 1]
        nrcol = len(self[starts[0]].replace("( ", "(").split()) + 1
        for i, (start, stop) in enumerate(zip(starts, stops)):
            columns = found[_re_nao_start][i][1].split()
            lines = ''
            if len(columns) < len(found[_re_nao_start][0][1].split()):
                for line in self[start:stop]:
                    if line.strip():
                        lines += line.replace('( ', '(') + ' ' + spins[i] + '\n'
                dfs.append(pd.read_csv(StringIO(lines), name=columns + ['Spin'], **csv_args))
            else:
                for line in self[start:stop]:
                    if line.strip():
                        lines += line.replace('( ', '(') + '\n'
                dfs.append(pd.read_csv(StringIO(lines), names=columns, **csv_args))
        df = pd.concat(dfs).reset_index(drop=True)
        split = df['Type(AO)'].str.extract('([A-z]{3})\((.*)\.)', expand=False)
        split.columns = ['type', 'ao']
        del df['Type(AO)']
        self.nao = pd.concat([df, split], axis=1)

    def parse_nbo(self):
        pass

    def parse_nlmo(self):
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

# Regex for NBO output file
_re_nao_start = 'NAO Atom No lang   Type(AO)    Occupancy'
_re_nao_stop01 = ' Summary of Natural Population Analysis'
_re_nao_stop02 = 'low occupancy.*core orbitals found on'
_re_nao_stop03 = 'electrons found in the effective core potential'

class MOMatrix(Editor):
    """
    The NBO code has the ability to dump any orbital transformation it performs
    such as the NBOAO, NLMOAO, etc. As long as it is in the AO basis, it is
    possible to add this momatrix to a corresponding universe's momatrix and
    view these orbitals.
    """

    def parse_momatrix(self, nbas, column_name=None):
        start = 3
        ncol = len(self[start].split())
        if nbas <= ncol:
            nrows = ncol
            occrows = ncol
        else:
            add = 1 if nbas % ncol else 0
            nrows = nbas * (nbas // ncol + add)
            occrows = nbas // ncol + add
        stop = start + nrows
        occstart = stop
        occstop = occstart + occrows
        momat = self.pandas_dataframe(start, stop, range(ncol)).stack().reset_index(drop=True)
        occvec = self.pandas_dataframe(occstart, occstop, range(ncol)).stack().reset_index(drop=True)
        momat.index.name = column_name if column_name is not None else 'coefficient1'
        occvec.index.name = column_name if column_name is not None else 'coefficient1'
        self.momatrix = momat
        self.occupation_vector = occvec

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
