# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
XYZ File Editor
##################
"""
import csv
import numpy as np
import pandas as pd
from io import StringIO
from exa.math.misc.indexing import starts_counts
from exa.relational.unit import Length
from exa.utility import mkp
from exatomic.editor import Editor
from exatomic.frame import compute_frame_from_atom


class XYZ(Editor):
    """
    An editor for programmatically editing `xyz`_ files.

    .. _xyz: https://en.wikipedia.org/wiki/XYZ_file_format
    """
    _header = '{nat}\n{comment}\n'
    _cols = ['symbol', 'x', 'y', 'z']

    def parse_atom(self, unit='A'):
        """
        Parse the atom table from the current xyz file.

        Args:
            unit (str): Default xyz unit of length is the Angstrom
        """
        df = pd.read_csv(StringIO(str(self)), delim_whitespace=True,
                                  names=('symbol', 'x', 'y', 'z'), header=None,
                                  skip_blank_lines=False)
        # The following algorithm works for both trajectory files and single xyz files
        nats = pd.Series(df[df[['y', 'z']].isnull().all(axis=1)].index)
        nats = nats[nats.diff() != 1].values
        comments = nats + 1
        nats = df.ix[nats, 'symbol']
        comments = df.ix[comments, :].dropna(how='all').index
        initials = nats.index.values.astype(np.int64) + 2
        counts = nats.values.astype(np.int64)
        frame, label, indices = starts_counts(initials, counts)
        df = df[df.index.isin(indices)]
        df[['x', 'y', 'z']] = df[['x', 'y', 'z']].astype(np.float64)
        df['symbol'] = df['symbol'].astype('category')
        df['frame'] = frame
        df['frame'] = df['frame'].astype('category')
        df.reset_index(drop=True, inplace=True)
        df.index.names = ['atom']
        df['x'] *= Length[unit, 'au']
        df['y'] *= Length[unit, 'au']
        df['z'] *= Length[unit, 'au']
        if self.meta is not None:
            self.meta['comments'] = {line: self._lines[line] for line in comments}
        else:
            self.meta = {'comments': {line: self._lines[line] for line in comments}}
        self.atom = df

    def write(self, path, trajectory=True, float_format='%    .8f'):
        """
        Write an xyz file (or files) to disk.

        Args:
            path (str): Directory or file path
            trajectory (bool): Write xyz trajectory file (default) or individual

        Returns:
            path (str): On success, return the directory or file path written
        """
        if trajectory:
            with open(path, 'w') as f:
                f.write(str(self))
        else:
            grps = self.atom.grouped()
            n = len(str(self.frame.index.max()))
            for frame, atom in grps:
                filename = str(frame).zfill(n) + '.xyz'
                with open(mkp(path, filename), 'w') as f:
                    f.write(self._header.format(nat=str(len(atom)),
                                                comment='frame: ' + str(frame)))
                    a = atom[self._cols].copy()
                    a['x'] *= Length['au', 'A']
                    a['y'] *= Length['au', 'A']
                    a['z'] *= Length['au', 'A']
                    a.to_csv(f, header=False, index=False, sep=' ', float_format=float_format,
                             quoting=csv.QUOTE_NONE, escapechar=' ')

    @classmethod
    def from_universe(cls, universe, float_format='%    .8f'):
        """
        Create an xyz file editor from a given universe. If the universe has
        more than one frame, creates an xyz trajectory format editor.
        """
        string = ''
        grps = universe.atom.grouped()
        for frame, atom in grps:
            string += cls._header.format(nat=len(atom), comment='frame: ' + str(frame))
            atom_copy = atom[cls._cols].copy()
            atom_copy['x'] *= Length['au', 'A']
            atom_copy['y'] *= Length['au', 'A']
            atom_copy['z'] *= Length['au', 'A']
            string += atom_copy.to_csv(sep=' ', header=False, quoting=csv.QUOTE_NONE,
                                       index=False, float_format=float_format,
                                       escapechar=' ')
        return cls(string, name=universe.name, description=universe.description,
                   meta=universe.meta)
