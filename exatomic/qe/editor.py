# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Quantum ESPRESSO Editor
##########################
"""
from exatomic.editor import Editor as AtomicEditor


class Editor(AtomicEditor):
    """
    """
    pass
#import pandas as pd
#import numpy as np
#from exa.relational.unit import Length
#from exatomic.editor import Editor as AtomicEditor
#from exqe.types import lengths
#
#
#class Editor(AtomicEditor):
#    """
#    Base editor class tailored to Quantum Espresso inputs and outputs.
#    """
#    def parse_atom(self):
#        """
#        Parse the :class:`~atomic.atom.Atom` dataframe.
#        """
#        atom_lines = []
#        frame = -1
#        length = None
#        append = False
#        for line in self:
#            if 'atomic_positions' in line.lower():
#                ls = line.split()
#                if len(ls) > 1:
#                    unit = ls[1].replace('(', '').replace(')', '').lower()
#                    length = lengths[unit]
#                append = True
#                frame += 1
#            elif append:
#                ls = line.split()
#                if len(ls) >= 4:
#                    atom_lines.append(ls[0:4] + [frame])
#                else:
#                    append = False
#        frame += 1
#        df = pd.DataFrame.from_records(atom_lines)
#        df.columns = ('symbol', 'x', 'y', 'z', 'frame')
#        df[['x', 'y', 'z']] = df[['x', 'y', 'z']].astype(np.float64)
#        df['x'] *= Length[length, 'au']
#        df['y'] *= Length[length, 'au']
#        df['z'] *= Length[length, 'au']
#        self.atom = df
#
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)
#        self.meta = {'program': 'qe'}
#
