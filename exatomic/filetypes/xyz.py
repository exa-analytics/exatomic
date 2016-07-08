# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
'''
XYZ File Support
##################

'''
import csv
import numpy as np
import pandas as pd
from io import StringIO
from exa.algorithms import arange1, arange2
from exatomic import Editor, Length, Atom, Universe, Editor


header = '{nat}\n{comment}\n'
comment = 'frame: {frame}'


class XYZ(Editor):
    '''
    An editor for programmatically manipulating xyz and xyz-like files.

    Provides convenience methods for transforming an xyz like file on disk into a
    :class:`~exatomic.universe.Universe`.
    '''
    def parse_atom(self):
        '''
        Extract the :class:`~exatomic.atom.Atom` dataframe from the file.

        Note:
            This method will add a key "comments" to the meta attribute.
        '''
        df = pd.read_csv(StringIO(str(self)), delim_whitespace=True, names=('symbol', 'x', 'y', 'z'),
                         header=None, skip_blank_lines=False)
        nats = pd.Series(df[df[['y', 'z']].isnull().all(axis=1)].index)   # Get all nat lines
        nats = nats[nats.diff() != 1].values
        comments = nats + 1                                               # Comment lines
        nats = df.ix[nats, 'symbol']
        comments = df.ix[comments, :].dropna(how='all').index
        initials = nats.index.values.astype(np.int64) + 2
        counts = nats.values.astype(np.int64)
        frame, label, indices = arange1(initials, counts)
        df = df[df.index.isin(indices)]
        df[['x', 'y', 'z']] = df[['x', 'y', 'z']].astype(np.float64)
        df['symbol'] = df['symbol'].astype('category')
        df['label'] = label
        df['label'] = df['label'].astype('category')
        df['frame'] = frame
        df['frame'] = df['frame'].astype('category')
        df.reset_index(drop=True, inplace=True)
        df.index.names = ['atom']
        df['x'] *= Length[self.unit, 'au']
        df['y'] *= Length[self.unit, 'au']
        df['z'] *= Length[self.unit, 'au']
        self._atom = Atom(df)
        self.meta['comments'] = {line: self._lines[line] for line in comments}

    def __init__(self, *args, unit='A', **kwargs):
        super().__init__(*args, **kwargs)
        self.unit = unit


def write_xyz(uni_or_editor, path, unit='angstrom', trj=False, precision=8):
    '''
    Write an xyz file, set of xyz files, or xyz trajectory file.

    Args:
        uni_or_string: One of :class:`~exatomic.universe.Universe`, :class:`~exatomic.editor.Editor`, or string text
        path (str): Full file path or directory path
        unit (str): Output length unit (default angstrom)
        trj (bool): Write a trajectory file (default False)
    '''
    ffmt = r'% .{}f'.format(precision)
    if isinstance(uni_or_editor, Universe):
        grps = uni_or_editor.atom.groupby('frame')
        if trj:
            _write_trj_from_uni(grps, path, unit, ffmt)
        else:
            _write_xyz_from_uni(grps, path, unit, ffmt)
    elif isinstance(uni_or_editor, AtomicEditor):
        if trj:
            _write_trj_from_editor(uni_or_editor, path, unit, ffmt)
        else:
            _write_xyz_from_editor(uni_or_editor, path, unit, ffmt)


def _write_trj_from_uni(grps, path, unit, ffmt):
    '''
    Write an XYZ trajectory file from a given universe.
    '''
    with open(path, 'w') as f:
        for frame, atom in grps:
            n = len(atom)
            c = comment.format(frame=frame)
            f.write(header.format(nat=n, comment=c))
            atom_ = atom[['symbol', 'x', 'y', 'z']].copy()
            atom_['x'] *= Length['au', 'A']
            atom_['y'] *= Length['au', 'A']
            atom_['z'] *= Length['au', 'A']
            atom_.to_csv(f, header=False, index=False, sep=' ', float_format=ffmt,
                         quoting=csv.QUOTE_NONE, escapechar=' ')

def _write_xyz_from_uni(grps, path, unit, ffmt):
    '''
    '''
    if traj:
        _wrtie

def write_xyz_from_atom(atom, path, unit='angstrom', traj=True):
    '''
    Write an xyz file from a universe.

    Args:
        atom (:class:`~exatomic.atom.Atom`): Atom dataframe
        path (str): Directory path (traj=False) or file path (traj=True)
        unit (str): Output length unit (Angstrom default)
        traj (bool): If true, output xyz trajectory file, otherwise write xyz for every frame
    '''
    grps = atom.groupby('frame')
    with open('xyz', 'w') as f:
        for frame, atom in grps:
            n = len(atom)
            c = comment.format(frame=frame)
            f.write(header.format(nat=n, comment=c))
            atom.to_csv(f, header=False, index=False, sep=' ', float_format='%   .8f',
                        columns=('symbol', 'x', 'y', 'z'), quoting=csv.QUOTE_NONE,
                        escapechar=' ')
