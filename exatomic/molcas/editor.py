# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
'''
Base editor
##################
'''

import numpy as np
import pandas as pd
from exatomic.container import Universe
from io import StringIO
from exatomic import Editor as AtomicEditor

class Editor(AtomicEditor):

    _to_universe = AtomicEditor.to_universe

    def _basis_map(self, start, stop, seht):
        df = []
        for ln in self[start:stop]:
            ln = ln.strip()
            shell, nprim, nbas, x = ln.split()
            if len(ln) < 30:
                df.append([shell.lower(), int(nprim), int(nbas), seht, False])
            else:
                df.append([shell.lower(), int(nprim), int(nbas), seht, True])
        return pd.DataFrame(df)

    def _find_break(self, start, finds=[]):
        stop = start
        if finds:
            while True:
                stop += 1
                if any((find in self[stop] for find in finds)):
                    return stop
        while True:
            stop += 1
            if not self[stop].strip():
                return stop

    def to_universe(self, *args, **kwargs):
        uni = self._to_universe(self, *args, **kwargs)
        try:
            uni.occupation_vector = self.occupation_vector
        except AttributeError:
            pass
        return uni

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.meta is None:
            self.meta = {'program': 'molcas'}
        else:
            self.meta.update({'program': 'molcas'})
