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
        df['nx'] = df['nx'].astype(np.int64)
        df['ny'] = df['ny'].astype(np.int64)
        df['nz'] = df['nz'].astype(np.int64)
        df['ox'] = df['ox'].astype(np.float64)
        df['oy'] = df['oy'].astype(np.float64)
        df['oz'] = df['oz'].astype(np.float64)
        df['xi'] = df['xi'].astype(np.float64)
        df['xj'] = df['xj'].astype(np.float64)
        df['xk'] = df['xk'].astype(np.float64)
        df['yi'] = df['yi'].astype(np.float64)
        df['yj'] = df['yj'].astype(np.float64)
        df['yk'] = df['yk'].astype(np.float64)
        df['zi'] = df['zi'].astype(np.float64)
        df['zj'] = df['zj'].astype(np.float64)
        df['zk'] = df['zk'].astype(np.float64)
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
                        field=self.field, fields=self.field.field_values, **kwargs)

    def _init(self):
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
        self.meta = {'comments': self[0:2]}

    def __init__(self, *args, label=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = label
        self._init()


def write_cube(path, universe, frame=None, field=None):
    '''
    Generate a cube file or files from a given unvierse.

    Only the path and universe are required; if no frame or field indices are
    provided, this function will write all fields in the given universe. If
    multiple frames are desired, pass a list of frame indices; the field
    argument should then be a list of the same length with the corresponding
    field indices or none (in which case the first field for each frame will be
    written). If multple fields per frame (multiple) is desired, use a list
    for frames as previously, but use a list of lists for fields where the
    outer list has the same length as the frame argument list and each "sublist"
    now contains the indices of the fields specific to that frame to be written.

    Args:
        path (str): Directory or file path
        universe (:class:`~atomic.universe.Universe`): Universe to pull data from
        frame (list or int): If only a specific frame is needed
        field (list or int): If only a specific field is needed

    Note:
        "Indices" in the above description always refer to dataframe index.
        Also always writes in atomic units.
    '''
    if isinstance(frame, list) and isinstance(field, list):
        if isinstance(field[0], list):
            raise NotImplementedError('frame=[], field=[[]]')
        else:
            raise NotImplementedError('frame=[], field=[]')
    elif isinstance(frame, list) and field is None:
        raise NotImplementedError('frame=[], field=None')
    elif frame is None and field is None:
        _write_first_field_of_each_frame(path, universe)
        return path
    else:
        raise TypeError('Unsupported argument types ({}, {}), see docstring.'.format(type(frame), type(field)))

def _write_first_field_of_each_frame(path, universe):
    '''
    Writes the first field of every frame of the universe in a given directory.
    '''
    raise NotImplementedError('Started but not completed')
    if not os.path.isdir(path):
        raise NotADirectoryError('Path {}, is not a directory or does not exist!'.format(path))
    framelist = universe.frame.index
    padding = len(str(len(framelist)))
    for frame in framelist:
        atom = universe.atom[(universe.atom['frame'] == frame), ['symbol', 'x', 'y', 'z']].copy()
        atom['Z'] = atom['symbol'].map(Isotope.symbol_to_Z())
        atom['electrons'] = 1.0
        del atom['symbol']
        field_data = universe.field[universe.field['frame'] == frame]
        idx = field_data.index[0]
        fdata = field_data.ix[idx, ['ox', 'oy', 'oz', 'nx', 'ny', 'nz',
                                    'xi', 'xj', 'xk', 'yi', 'yj', 'yk',
                                    'zi', 'zj', 'zk']]
        ox, oy, oz, nx, ny, nz, xi, xj, xk, yi, yj, yk, zi, zj, zk = fdata
        nat = universe.frame.ix[frame, 'atom_count']
        field = _reshape_to_cube(universe.field_values[idx], nx, ny, nz)
        name = mkp(path, str(frame).zfill(padding) + '.cube')
        with open(name, 'w') as f:
            f.write(str(universe) + '\n')
            f.write('frame: {}\n'.format(frame))
            f.write('{} {} {} {}\n'.format(nat, ox, oy, oz))
            f.write('{} {} {} {}\n'.format(nx, xi, xj, xk))
            f.write('{} {} {} {}\n'.format(ny, yi, yj, yk))
            f.write('{} {} {} {}\n'.format(nz, zi, zj, zk))
            atom.to_csv(f, sep=' ')
            fiel.to_csv(f, sep=' ')

def _reshape_to_cube(field_values):
    pass
