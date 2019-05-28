# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
ADF NMR Output Editor
#############################
Editor class for parsing the NMR data from an ADF calculation
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from collections import defaultdict
import re
import six
import numpy as np
import pandas as pd
from io import StringIO
from exa.util.units import Length
from exa import TypedMeta
from exatomic.base import sym2z
#from exatomic.algorithms.basis import lmap, enum_cartesian
#from exatomic.algorithms.numerical import dfac21
from exatomic.core.atom import Atom
from exatomic.core.tensor import NMRshielding
#from exatomic.core.basis import BasisSet, BasisSetOrder
#from ..core.orbital import Orbital, Excitation, MOMatrix
from ..editor import Editor

class OutMeta(TypedMeta):
    atom = Atom
    nmr_shielding = NMRshielding

class Output(six.with_metaclass(OutMeta, Editor)):
    """ADF NMR parser"""
    def parse_atom(self):
        _reatom = "NUCLEAR COORDINATES (ANGSTROMS):"
        found = self.find(_reatom, keys_only=True)
        if len(found) > 1:
            raise NotImplementedError("We can only parse outputs from a single NMR calculation")
        start = found[0] + 3
        stop = start
        while self[stop].strip(): stop += 1
        # a bit of a hack to make sure that there is no formatting change depending on the
        # number of atoms in the molecule as the index is right justified so if there are
        # more than 100 atoms it will fill the alloted space for the atom index and change the
        # delimitter and therefore the number of columns
        self[start:stop] = map(lambda x: x.replace('(', ''), self[start:stop])
        atom = self.pandas_dataframe(start, stop, ncol=5)
        #atom.drop(1, axis='columns', inplace=True)
        atom.columns = ['symbol', 'set', 'x', 'y', 'z']
        for c in ['x', 'y', 'z']: atom[c] *= Length['Angstrom', 'au']
        atom['Z'] = atom['symbol'].map(sym2z)
        atom['frame'] = 0
        atom['set'] = list(map(lambda x: x.replace('):', ''), atom['set']))
        atom['set'] = atom['set'].astype(int)
        self.atom = atom

    def parse_nmr_shielding(self):
        _reatom = "N U C L E U S :"
        _reshield = "==== total shielding tensor"
        found = self.find(_reatom, keys_only=True)
        if not found:
            raise NotImplementedError("Could not find {} in output".format(_reatom))
        dfs = []
        for start in found:
            start_shield = self.find(_reshield, keys_only=True, start=start)[0] + start + 2
            end_shield = start_shield + 3
            symbol, index = self[start].split()[-1].split('(')
            index = int(index.replace(')', ''))
            isotropic = float(self[start_shield+4].split()[-1])
            df = self.pandas_dataframe(start_shield, end_shield, ncol=3)
            cols = ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']
            df = pd.DataFrame(df.unstack().values.reshape(1,9), columns=cols)
            df['isotropic'] = isotropic
            df['atom'] = index
            df['symbol'] = symbol
            df['label'] = 'nmr shielding'
            df['frame'] = 0
            dfs.append(df)
        shielding = pd.concat(dfs, ignore_index=True)
        self.nmr_shielding = shielding

