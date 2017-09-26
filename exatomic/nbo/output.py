# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
'''
exnbo Output Editor
####################################
'''
<<<<<<< HEAD
#import re
#import numpy as np
#import pandas as pd
#from io import StringIO
#
#from exa.relational.isotope import symbol_to_z, z_to_symbol
#
#from .editor import Editor
#from exatomic import Length
#from exatomic.basis import GaussianBasisSet
#
#csv_args = {'delim_whitespace': True}
#
#class Output(Editor):
#
#    _to_universe = Editor.to_universe
#
#    def to_universe(self):
#        raise NotImplementedError('This editor has no parse_atom method.')
#
#    def parse_nao(self):
#        found = self.find(_re_nao_start, _re_nao_stop01)
#        regex = self.regex(_re_nao_stop02, _re_nao_stop03, keys_only=True)
#        starts = [i[0] + 2 for i in found[_re_nao_start]]
#        stops = [i[0] for i in found[_re_nao_stop01]]
#        if regex[_re_nao_stop03]:
#            stops = regex[_re_nao_stop03]
#        elif regex[_re_nao_stop02]:
#            keys = regex[_re_nao_stop02]
#            stops = [keys[0]] + keys[2::2]
#        dfs = []
#        spins = [-1, 0, 1]
#        nrcol = len(self[starts[0]].replace("( ", "(").split()) + 1
#        for (lno, col), start, stop, spin in zip(found[_re_nao_start], starts, stops, spins):
#            columns = col.split()
#            lines = [line.replace("( ", "(") for line in self[start:stop]]
#            dfs.append(pd.read_csv(StringIO('\n'.join(lines)), names=columns, **csv_args))
#            dfs[-1]['spin'] = spin
#        df = pd.concat(dfs).reset_index(drop=True)
#        split = df['Type(AO)'].str.extract('([A-z]{3})\((.*)\)', expand=False)
#        split.columns = ['type', 'ao']
#        del df['Type(AO)']
#        self.nao = pd.concat([df, split], axis=1)
#
#    def parse_nbo(self):
#        pass
#
#    def parse_nlmo(self):
#        pass
#
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)
#
## Regex for NBO output file
#_re_nao_start = 'NAO Atom No lang   Type(AO)    Occupancy'
#_re_nao_stop01 = ' Summary of Natural Population Analysis'
#_re_nao_stop02 = 'low occupancy.*core orbitals found on'
#_re_nao_stop03 = 'electrons found in the effective core potential'
#
#class MOMatrix(Editor):
#    """
#    The NBO code has the ability to dump any orbital transformation it performs
#    such as the NBOAO, NLMOAO, etc. As long as it is in the AO basis, it is
#    possible to add this momatrix to a corresponding universe's momatrix and
#    view these orbitals.
#    """
#    _to_universe = Editor.to_universe
#
#    def to_universe(self):
#        raise NotImplementedError('This editor has no parse_atom method.')
#
#    def parse_momatrix(self, nbas, column=None):
#        start = 3
#        ncol = len(self[start].split())
#        if nbas <= ncol:
#            nrows = ncol
#            occrows = ncol
#        else:
#            add = 1 if nbas % ncol else 0
#            nrows = nbas * (nbas // ncol + add)
#            occrows = nbas // ncol + add
#        stop = start + nrows
#        occstart = stop
#        occstop = occstart + occrows
#        momat = self.pandas_dataframe(start, stop, range(ncol)).stack().reset_index(drop=True)
#        occvec = self.pandas_dataframe(occstart, occstop, range(ncol)).stack().reset_index(drop=True)
#        momat.index.name = column if column is not None else 'coef1'
#        occvec.index.name = column if column is not None else 'coef1'
#        self.momatrix = momat
#        self.occupation_vector = occvec
#
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)
=======
import re
import numpy as np
import pandas as pd
from io import StringIO

from exa.relational.isotope import symbol_to_z, z_to_symbol

from .editor import Editor
from exatomic import Length
from exatomic.orbital import Orbital

csv_args = {'delim_whitespace': True}

class Output(Editor):

    _to_universe = Editor.to_universe

    def to_universe(self):
        raise NotImplementedError('This editor has no parse_atom method.')

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
        for (lno, col), start, stop, spin in zip(found[_re_nao_start], starts, stops, spins):
            columns = col.split()
            lines = [line.replace("( ", "(") for line in self[start:stop]]
            dfs.append(pd.read_csv(StringIO('\n'.join(lines)), names=columns, **csv_args))
            dfs[-1]['spin'] = spin
        df = pd.concat(dfs).reset_index(drop=True)
        split = df['Type(AO)'].str.extract('([A-z]{3})\((.*)\)', expand=False)
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
    _to_universe = Editor.to_universe

    def to_universe(self):
        raise NotImplementedError('This editor has no parse_atom method.')

    def parse_momatrix(self, nbas, column=None, os=False):
        """
        Requires the number of basis functions in this matrix.
        """
        column = 'coef' if column is None else column
        start = 3 if not os else 4
        ncol = len(self[start].split())
        if nbas <= ncol:
            nrows = ncol
            occrows = ncol
        else:
            add = 1 if nbas % ncol else 0
            nrows = nbas * (nbas // ncol + add)
            occrows = nbas // ncol + add
        # This code is repetitive with the beta spin parsing below
        # generalize for i in rnage(2): etc. etc.
        stop = start + nrows
        occstart = stop
        occstop = occstart + occrows
        coef = self.pandas_dataframe(start, stop, range(ncol)
                                     ).stack().reset_index(drop=True)
        occvec = self.pandas_dataframe(occstart, occstop, range(ncol)
                                       ).stack().reset_index(drop=True)
        orbital = np.repeat(range(nbas), nbas)
        chi = np.tile(range(nbas), nbas)
        self.momatrix = pd.DataFrame.from_dict({'orbital': orbital, 'chi': chi,
                                                column: coef, 'frame': 0})
        self.orbital = Orbital.from_occupation_vector(occvec)
        self.occupation_vector = {column: occvec}
        if os:
            start = self.find_next('BETA  SPIN', start=stop, keys_only=True) + 1
            stop = start + nrows
            occstart = stop
            occstop = occstart + occrows
            beta = self.pandas_dataframe(start, stop, range(ncol)
                                        ).stack().reset_index(drop=True)
            betaocc = self.pandas_dataframe(occstart, occstop, range(ncol),
                                            ).stack().reset_index(drop=True)
            self.momatrix['coef1'] = beta
            betaorb = Orbital.from_occupation_vector(betaocc)
            self.orbital = pd.concat([self.orbital, betaorb]).reset_index(drop=True)
            self.occupation_vector[column + '1'] = betaocc

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
>>>>>>> tjd_master
