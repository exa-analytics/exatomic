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
from exa import Series, Field3D
from atomic import Isotope, Universe, Editor, Atom
from atomic.frame import minimal_frame


class Cube(Editor):
    '''
    An editor for handling cube files.

    .. code-block:: python

        cube = Cube('my.cube')
        cube.atom                # Displays the atom dataframe
        cube.field               # Displays the field dataframe
        cube.field_values()      # Displayss the field values corresponding to field 0
        uni = cube.to_universe() # Converts the cube file editor to a universe
        uni                      # Renders the cube file
    '''
    @property
    def field(self):
        '''
        Display the field dataframe.
        '''
        if self._field is None:
            self.parse_field()
        return self._field

    def field_values(self, key=0):
        '''
        Display the values of the scalar field.
        '''
        return self.field._fields[key]

    def parse_atom(self):
        '''
        Parse the :class:`~atomic.atom.Atom` object from the cube file in place.

        See Also:
            :py:attr:`~atomic.editor.Editor.atom`
        '''
        df = pd.read_csv(StringIO('\n'.join(self._lines[6:self._volume_data_start])), delim_whitespace=True,
                         header=None, names=('Z', 'nelectron', 'x', 'y', 'z'))
        del df['nelectron']
        df['symbol'] = df['Z'].map(Isotope.Z_to_symbol()).astype('category')
        del df['Z']
        df['frame'] = 0
        df['frame'] = df['frame'].astype('category')
        df['label'] = pd.Series(range(self._nat), dtype='category')
        self._atom = Atom(df)

    def parse_field(self):
        '''
        Parse the :class:`~exa.numerical.Field` object from the cube filed in place.

        Note:
            The :class:`~exa.numerical.Field` object tracks both the field meta-
            data (i.e. information about the discretization and shape of the
            field's spatial points) as well as the field values (at each of
            those points in space).

        See Also:
            :class:`~exa.numerical.Field`
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
        self._field = Field3D(fields, df)

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
                        field=self.field, fields=self.field.field_values, **kwargs)

    def _init(self, *args, label=None, **kwargs):
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
        self.label = label
        self.meta = {'comments': self[0:2]}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init()
