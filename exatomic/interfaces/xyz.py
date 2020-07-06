# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
XYZ File Editor
##################
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import six
import csv
import numpy as np
import pandas as pd
from exa import TypedMeta
from exa.util.units import Length
from exa.util.utility import mkp
from exatomic.core.editor import Editor
from exatomic.core.frame import compute_frame_from_atom, Frame
from exatomic.core.atom import Atom
from exatomic.algorithms.indexing import starts_counts


class Meta(TypedMeta):
    atom = Atom
    frame = Frame


class XYZ(six.with_metaclass(Meta, Editor)):
    """
    An editor for programmatically editing `xyz`_ files.

    .. _xyz: https://en.wikipedia.org/wiki/XYZ_file_format
    """
    _header = '{nat}\n{comment}\n'
    _cols = ['symbol', 'x', 'y', 'z']

    def parse_frame(self):
        self.frame = compute_frame_from_atom(self.atom)

    def parse_atom(self, unit='Angstrom', names=('symbol', 'x', 'y', 'z')):
        """
        Parse the atom table from the current xyz file.

        Args:
            unit (str): Default xyz unit of length is the Angstrom
        """
        df = pd.read_csv(six.StringIO(six.u(str(self))), delim_whitespace=True,
                                      names=names, header=None,
                                      skip_blank_lines=False)
        # The following algorithm works for both trajectory files and single xyz files
        nats = pd.Series(df[df[['y', 'z']].isnull().all(axis=1)].index)
        nats = nats[nats.diff() != 1].values
        comments = nats + 1
        nats = df.loc[nats, 'symbol']
        comments = df.loc[comments, :].dropna(how='all').index
        initials = nats.index.values.astype(np.int64) + 2
        counts = nats.values.astype(np.int64)
        frame, _, indices = starts_counts(initials, counts)
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
            grps = self.atom.cardinal_groupby()
            n = len(str(self.frame.index.max()))
            for frame, atom in grps:
                filename = str(frame).zfill(n) + '.xyz'
                with open(mkp(path, filename), 'w') as f:
                    f.write(self._header.format(nat=str(len(atom)),
                                                comment='frame: ' + str(frame)))
                    a = atom[self._cols].copy()
                    a['x'] *= Length['au', 'Angstrom']
                    a['y'] *= Length['au', 'Angstrom']
                    a['z'] *= Length['au', 'Angstrom']
                    a.to_csv(f, header=False, index=False, sep=' ', float_format=float_format,
                             quoting=csv.QUOTE_NONE, escapechar=' ')

    @classmethod
    def from_universe(cls, universe, atom_table='atom', float_format='%    .8f'):
        """
        Create an xyz file editor from a given universe. If the universe has
        more than one frame, creates an xyz trajectory format editor.

        Args:
            universe: The universe
            atom_table (str): One of 'atom', 'unit', or 'visual' corresponding to coordinates
            float_format (str): Floating point format (for writing)
        """
        string = ''
        grps = universe.atom.cardinal_groupby()
        for frame, atom in grps:
            string += cls._header.format(nat=len(atom), comment='frame: ' + str(frame))
            atom_copy = atom[cls._cols].copy()
            if atom_table == 'unit':
                atom_copy.update(universe.unit_atom)
            elif atom_table == 'visual':
                atom_copy.update(universe.visual_atom)
            atom_copy['x'] *= Length['au', 'Angstrom']
            atom_copy['y'] *= Length['au', 'Angstrom']
            atom_copy['z'] *= Length['au', 'Angstrom']
            string += atom_copy.to_csv(sep=' ', header=False, quoting=csv.QUOTE_NONE,
                                       index=False, float_format=float_format,
                                       escapechar=' ')
        return cls(string, name=universe.name, description=universe.description,
                   meta=universe.meta)
