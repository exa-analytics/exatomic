# -*- coding: utf-8 -*-
'''
Cube File Support
====================

'''
import numpy as np
import pandas as pd
from io import StringIO
from exa import Series
from atomic import Isotope, Universe, Editor, Atom
from atomic.frame import minimal_frame


class Cube(Editor):
    '''
    An editor for handling cube files.

    .. code-block:: python

        cube = Cube('my.cube')
        cube.field               # Displays the scalar field as a 1-D array (column-ordered)
        cube.atom                # Displays the atom dataframe
        uni = cube.to_universe() # Converts the cube file editor to a universe
        uni                      # Renders the cube file
    '''
    @property
    def field(self):
        if not hasattr(self, '_field'):
            self.parse_field()
        return self._field

    def parse_atom(self):
        '''
        '''
        df = pd.read_csv(StringIO('\n'.join(self._lines[6:self._volume_data_start])), delim_whitespace=True,
                         header=None, names=('Z', 'nelectron', 'x', 'y', 'z'))
        df['symbol'] = df['Z'].map(Isotope.Z_to_symbol()).astype('category')
        df['frame'] = 0
        df['frame'] = df['frame'].astype('category')
        df['label'] = range(self._nat)
        df['label'] = df['label'].astype('category')
        del df['Z']
        del df['nelectron']
        df.index.names = ['atom']
        self._atom = Atom(df)

    def parse_field(self):
        '''
        '''
        data = pd.read_csv(StringIO('\n'.join(self._lines[self._volume_data_start:])),
                           delim_whitespace=True, header=None).values.ravel()
        self._field = Series(data[~np.isnan(data)])

    def parse_frame(self):
        '''
        '''
        self._frame = minimal_frame(self.atom)

    def to_universe(self, **kwargs):
        '''
        See Also:
            :func:`~atomic.editor.Editor.to_universe`
        '''
        return Universe(frame=self.frame, atom=self.atom, meta=self.meta,
                        fields=[self.field], **kwargs)

    def _init(self):
        '''
        Perform some preliminary parsing so that future parsing of atoms, etc.
        is easy. Also parse out metadata.
        '''
        nat, ox, oy, oz = self[2].split()
        nx, xi, xj, xk = self[3].split()
        ny, yi, yj, yk = self[4].split()
        nz, zi, zj, zk = self[5].split()
        self._nat = int(nat)
        self._nx = int(nx)
        self._ny = int(ny)
        self._nz = int(nz)
        self._unit = 'au'
        if self._nx < 0 or self._ny < 0 or self._nz < 0:
            self._unit = 'A'
            self._nx = abs(self._nx)
            self._ny = abs(self._ny)
            self._nz = abs(self._nz)
        self._ox = float(ox)
        self._oy = float(oy)
        self._oz = float(oz)
        self._xi = float(xi)
        self._xj = float(xj)
        self._xk = float(xk)
        self._yi = float(yi)
        self._yj = float(yj)
        self._yk = float(yk)
        self._zi = float(zi)
        self._zj = float(zj)
        self._zk = float(zk)
        self._volume_data_start = 6 + self._nat
        self.meta = {'comments': self[0:2]}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init()
