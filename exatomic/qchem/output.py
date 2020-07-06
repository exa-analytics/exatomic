# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Q-Chem Ouput Editor
#######################
Editor classes for simple Q-Chem output files
"""
import six
import numpy as np
import pandas as pd

from exa import TypedMeta
from exa.util.units import Length#, Energy
from .editor import Editor
from exatomic.base import sym2z
from exatomic.core.atom import Atom, Frequency
from exatomic.core.frame import Frame#, compute_frame_from_atom
from exatomic.core.basis import (BasisSet, BasisSetOrder, Overlap)#, deduplicate_basis_sets)
from exatomic.core.orbital import Orbital, MOMatrix, Excitation
#from exatomic.algorithms.basis import lmap, lorder

class QMeta(TypedMeta):
    atom = Atom
    basis_set = BasisSet
    orbital = Orbital
    momatrix = MOMatrix
    basis_set_order = BasisSetOrder
    frame = Frame
    excitation = Excitation
    frequency = Frequency
    overlap = Overlap
    multipole = pd.DataFrame

class Output(six.with_metaclass(QMeta, Editor)):
    def parse_atom(self):
        # Atom flags
        _regeom01 = "Standard Nuclear Orientation (Angstroms)"
        _regeom02 = "Coordinates (Angstroms)"
        # Find Data
        found = self.find(_regeom01, keys_only=True)
        starts = np.array(found) + 3
        stop = starts[0]
        while '-------' not in self[stop]: stop += 1
        stops = starts + (stop - starts[0])
        dfs = []
        for i, (start, stop) in enumerate(zip(starts, stops)):
            atom = self.pandas_dataframe(start, stop, 5)
            atom['frame'] = i
            dfs.append(atom)
        atom = pd.concat(dfs).reset_index(drop=True)
        atom.columns = ['set', 'symbol', 'x', 'y', 'z', 'frame']
        atom['set'] -= 1
        atom['x'] *= Length['Angstrom', 'au']
        atom['y'] *= Length['Angstrom', 'au']
        atom['z'] *= Length['Angstrom', 'au']
        atom['Z'] = atom['symbol'].map(sym2z)
        self.atom = atom

    def __init__(self, *args, **kwargs):
        super(Output, self).__init__(*args,**kwargs)

