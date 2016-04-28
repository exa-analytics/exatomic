# -*- coding: utf-8 -*-
'''
XYZ File Support
====================

'''
import numpy as np
import pandas as pd
from io import StringIO
from exa.algorithms import arange1, arange2
from atomic import Editor, Length, Atom
from atomic.frame import minimal_frame


header = '{nat}\n{comment}\n'
comment = 'frame: {frame}'


class XYZ(Editor):
    '''
    An editor for programmatically manipulating xyz and xyz-like files.

    Provides convenience methods for transforming an xyz like file on disk into a
    :class:`~atomic.universe.Universe`.
    '''
    def parse_atom(self, unit=None):
        '''
        Extract the :class:`~atomic.atom.Atom` dataframe from the file.

        Args:
            unit (str): Can be enforced otherwise inferred from the file data.

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
        unit = unit if unit else 'A'
        df['x'] *= Length[unit, 'au']
        df['y'] *= Length[unit, 'au']
        df['z'] *= Length[unit, 'au']
        self._atom = Atom(df)
        self.meta['comments'] = {line: self._lines[line] for line in comments}

    def parse_frame(self):
        '''
        Create a :class:`~atomic.frame.Frame` from the xyz file lines.

        Wrapper around :func:`~atomic.frame.minimal_frame`; will create an
        :class:`~atomic.atom.Atom` if it doesn't already exist.
        '''
        self._frame = minimal_frame(self.atom)


def write_xyz(uni_or_string, path, unit='angstrom', traj=False):
    '''
    Write an xyz file, set of xyz files, or xyz trajectory file.

    Args:
        uni_or_string: One of :class:`~atomic.universe.Universe`, :class:`~atomic.filetypes.xyz.XYZ`, or string text
        path (str): Full file path or directory path
        unit (str): Output length unit (default angstrom)
        traj (bool): Write a trajectory file (default False)
    '''
    if isinstance(uni_or_string, Universe):
        write_xyz_from_universe(uni_or_string.atom, path, unit, traj)
    else:
        raise NotImplementedError()


def write_xyz_from_atom(atom, path, unit='angstrom', traj=True):
    '''
    Write an xyz file from a universe.

    Args:
        atom (:class:`~atomic.atom.Atom`): Atom dataframe
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
