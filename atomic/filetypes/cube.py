# -*- coding: utf-8 -*-
'''
Cube File Support
====================
Cube files contain an atomic geometry and scalar field values corresponding to
a physical quantity.
'''
import numpy as np
import pandas as pd
from io import StringIO
from exa import Series
from atomic import Isotope, Universe, Editor, Atom
from atomic.frame import minimal_frame
from atomic.field import UField3D


class Cube(Editor):
    '''
    An editor for handling cube files.

    .. code-block:: python

        cube = Cube('my.cube')
        cube.atom                # Displays the atom dataframe
        cube.field               # Displays the field dataframe
        cube.field_values        # Displays the list of field values
        uni = cube.to_universe() # Converts the cube file editor to a universe
        uni                      # Renders the cube file
    '''
    @property
    def field(self):
        if self._field is None:
            self.parse_field()
        return self._field

    @property
    def field_values(self):
        return self.field.field_values

    def parse_atom(self):
        '''
        Parse the :class:`~atomic.atom.Atom` object from the cube file in place.
        '''
        df = pd.read_csv(StringIO('\n'.join(self._lines[6:self._volume_data_start])), delim_whitespace=True,
                         header=None, names=('Z', 'nelectron', 'x', 'y', 'z'))
        del df['nelectron']
        df['symbol'] = df['Z'].map(Isotope.Z_to_symbol()).astype('category')
        del df['Z']
        df['frame'] = pd.Series([0] * len(df), dtype='category')
        df['label'] = pd.Series(range(self._nat), dtype='category')
        self._atom = Atom(df)

    def parse_field(self):
        '''
        Parse the scalar field into a trait aware object.

        Note:
            The :class:`~atomic.field.UField3D` object tracks both the
            field data (i.e. information about the discretization and shape of
            the field's spatial points) as well as the field values (at each of
            those points in space).
        '''
        data = pd.read_csv(StringIO('\n'.join(self._lines[self._volume_data_start:])),
                           delim_whitespace=True, header=None).values.ravel()
        df = pd.Series({'ox': self._ox, 'oy': self._oy, 'oz': self._oz,
                        'nx': self._nx, 'ny': self._ny, 'nz': self._nz,
                        'xi': self._xi, 'xj': self._xj, 'xk': self._xk,
                        'yi': self._yi, 'yj': self._yj, 'yk': self._yk,
                        'zi': self._zi, 'zj': self._zj, 'zk': self._zk,
                        'frame': 0, 'label': self.label})
        df = df.to_frame().T
        df['frame'] = df['frame'].astype(np.int64)
        df['frame'] = df['frame'].astype('category')
        fields = [Series(data[~np.isnan(data)])]
        self._field = UField3D(fields, df)

    def parse_frame(self):
        '''
        Create the :class:`~atomic.frame.Frame` from the atom dataframe.
        '''
        self._frame = minimal_frame(self.atom)

    def to_universe(self, **kwargs):
        '''
        See Also:
            See :class:`~atomic.universe.Universe` for additional arguments.
        '''
        return Universe(frame=self.frame, atom=self.atom, meta=self.meta,
                        field=self.field, **kwargs)

    def _init(self, *args, label=None, **kwargs):
        '''
        Perform some preliminary parsing so that future parsing of atoms, etc.
        is easy. Also parse out metadata (comments).
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
        self.label = label
        self.meta = {'comments': self[0:2]}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init()
