 #-*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Cube File Support
##########################
Cube files contain an atomic geometry and scalar field values corresponding to
a physical quantity.
"""
import numpy as np
import pandas as pd
from io import StringIO
from exa.numerical import Series
from exa.relational.isotope import z_to_symbol, symbol_to_z
from exatomic import __version__
from exatomic.atom import Atom
from exatomic.editor import Editor
from exatomic.field import AtomicField
z_to_symbol = z_to_symbol()
symbol_to_z = symbol_to_z()

class Cube(Editor):
    """
    An editor for handling cube files.

    .. code-block:: python

        cube = Cube('my.cube')
        cube.atom                # Displays the atom dataframe
        cube.field               # Displays the field dataframe
        cube.field_values        # Displays the list of field values
        uni = cube.to_universe() # Converts the cube file editor to a universe
        uni                      # Renders the cube file

    Warning:
        Be sure your cube is in atomic units.
    """
    def parse_atom(self):
        """
        Parse the :class:`~exatomic.atom.Atom` object from the cube file in place.
        """
        nat = abs(int(self[2].split()[0]))
        ncol = len(self[6].split())
        names = ['Z', 'Zeff', 'x', 'y', 'z']
        df = self.pandas_dataframe(6, nat + 6, names)
        df['symbol'] = df['Z'].map(z_to_symbol).astype('category')
        df['label'] = range(nat)
        df['frame'] = 0
        self.atom = Atom(df)

    def parse_field(self):
        """
        parse the scalar field into a trait aware object.

        note:
            the :class:`~exatomic.field.atomicfield` object tracks both the
            field data (i.e. information about the discretization and shape of
            the field's spatial points) as well as the field values (at each of
            those points in space).
        """
        self.meta = {'comments': self[:2]}
        typs = [int, float, float, float]
        nat, ox, oy, oz = [typ(i) for typ, i in zip(typs, self[2].split())]
        nx, dxi, dxj, dxk = [typ(i) for typ, i in zip(typs, self[3].split())]
        ny, dyi, dyj, dyk = [typ(i) for typ, i in zip(typs, self[4].split())]
        nz, dzi, dzj, dzk = [typ(i) for typ, i in zip(typs, self[5].split())]
        nat, nx, ny, nz = abs(nat), abs(nx), abs(ny), abs(nz)
        volstart = nat + 6
        if len(self[volstart].split()) < 5: volstart += 1
        ncol = len(self[volstart].split())
        data = self.pandas_dataframe(volstart, len(self), ncol).values.ravel()
        df = pd.Series({'ox': ox, 'oy': oy, 'oz': oz,
                        'nx': nx, 'ny': ny, 'nz': nz,
                        'dxi': dxi, 'dxj': dxj, 'dxk': dxk,
                        'dyi': dyi, 'dyj': dyj, 'dyk': dyk,
                        'dzi': dzi, 'dzj': dzj, 'dzk': dzk,
                        'frame': 0, 'label': self.label,
                        'field_type': self.field_type}).to_frame().T
        for col in ['nx', 'ny', 'nz']:
            df[col] = df[col].astype(np.int64)
        for col in ['ox', 'oy', 'oz', 'dxi', 'dxj', 'dxk',
                    'dyi', 'dyj', 'dyk', 'dzi', 'dzj', 'dzk']:
            df[col] = df[col].astype(np.float64)
        fields = [Series(data[~np.isnan(data)])]
        self.field = AtomicField(df, field_values=fields)

    @classmethod
    def from_universe(cls, uni, idx, name=None, frame=None):
        """
        Write a cube file format Editor from a given field in an
        :class:`~exatomic.container.Universe`.

        Args
            uni (Universe): a universe
            idx (int): field index
        """
        name = '' if name is None else name
        frame = uni.atom.nframes - 1 if frame is None else frame
        hdr = '{} -- written by exatomic v{}\n\n'
        ffmt = ' {:> 12.6f}'
        flfmt = ('{:>5}' + ffmt * 3 + '\n').format
        if 'Z' not in uni.atom:
            uni.atom['Z'] = uni.atom['symbol'].map(symbol_to_z)
        if 'Zeff' not in uni.atom:
            uni.atom['Zeff'] = uni.atom['Z'].astype(np.float64)
        frame = uni.atom[uni.atom['frame'] == frame]
        for col in ['nx', 'ny', 'nz']:
            uni.field[col] = uni.field[col].astype(np.int64)
        field = uni.field.loc[idx]
        volum = uni.field.field_values[idx]
        orig = len(frame.index), field.ox, field.oy, field.oz
        nx, ny, nz = field.nx, field.ny, field.nz
        xdim = nx, field.dxi, field.dxj, field.dxk
        ydim = ny, field.dyi, field.dyj, field.dyk
        zdim = nz, field.dzi, field.dzj, field.dzk
        atargs = {'float_format': '%12.6f',
                  'header': None, 'index': None,
                  'columns': ['Z', 'Zeff', 'x', 'y', 'z']}
        chnk = ''.join(['{}' * 6 + '\n' for i in range(nz // 6)])
        if nz % 6: chnk += '{}' * (nz % 6) + '\n'
        return cls(hdr.format(name, __version__)
                  + flfmt(*orig) + flfmt(*xdim)
                  + flfmt(*ydim) + flfmt(*zdim)
                  + uni.atom.to_string(**atargs) + '\n'
                  + (chnk * nx * ny).format(*volum.apply(
                    ffmt.replace('f', 'E').format)))

    def __init__(self, *args, label=None, field_type=None, **kwargs):
        super(Cube, self).__init__(*args, **kwargs)
        self.label = label
        self.field_type = field_type



#
# class Cube(Editor):
#     """
#     An editor for handling cube files.
#
#     .. code-block:: python
#
#         cube = Cube('my.cube')
#         cube.atom                # Displays the atom dataframe
#         cube.field               # Displays the field dataframe
#         cube.field_values        # Displays the list of field values
#         uni = cube.to_universe() # Converts the cube file editor to a universe
#         uni                      # Renders the cube file
#
#     Warning:
#         Be sure your cube is in atomic units.
#     """
#     def parse_atom(self):
#         """
#         Parse the :class:`~exatomic.atom.Atom` object from the cube file in place.
#         """
#         df = pd.read_csv(StringIO('\n'.join(self._lines[6:self._volume_data_start])), delim_whitespace=True,
#                          header=None, names=('Z', 'nelectron', 'x', 'y', 'z'))
#         del df['nelectron']
#         mapper = z_to_symbol()
#         df['symbol'] = df['Z'].map(mapper).astype('category')
#         del df['Z']
#         df['frame'] = pd.Series([0] * len(df), dtype='category')
#         df['label'] = pd.Series(range(self._nat), dtype='category')
#         self._atom = Atom(df)
#
#     def parse_field(self):
#         """
#         parse the scalar field into a trait aware object.
#
#         note:
#             the :class:`~exatomic.field.atomicfield` object tracks both the
#             field data (i.e. information about the discretization and shape of
#             the field's spatial points) as well as the field values (at each of
#             those points in space).
#         """
#         data = pd.read_csv(StringIO('\n'.join(self._lines[self._volume_data_start:])),
#                            delim_whitespace=True, header=None).values.ravel()
#         df = pd.Series({'ox': self._ox, 'oy': self._oy, 'oz': self._oz,
#                         'nx': self._nx, 'ny': self._ny, 'nz': self._nz,
#                         'dxi': self._dxi, 'dxj': self._dxj, 'dxk': self._dxk,
#                         'dyi': self._dyi, 'dyj': self._dyj, 'dyk': self._dyk,
#                         'dzi': self._dzi, 'dzj': self._dzj, 'dzk': self._dzk,
#                         'frame': 0, 'label': self.label, 'field_type': self.field_type})
#         df = df.to_frame().T
#         df['frame'] = df['frame'].astype(np.int64)
#         df['frame'] = df['frame'].astype('category')
#         df['nx'] = df['nx'].astype(np.int64)
#         df['ny'] = df['ny'].astype(np.int64)
#         df['nz'] = df['nz'].astype(np.int64)
#         df['ox'] = df['ox'].astype(np.float64)
#         df['oy'] = df['oy'].astype(np.float64)
#         df['oz'] = df['oz'].astype(np.float64)
#         df['dxi'] = df['dxi'].astype(np.float64)
#         df['dxj'] = df['dxj'].astype(np.float64)
#         df['dxk'] = df['dxk'].astype(np.float64)
#         df['dyi'] = df['dyi'].astype(np.float64)
#         df['dyj'] = df['dyj'].astype(np.float64)
#         df['dyk'] = df['dyk'].astype(np.float64)
#         df['dzi'] = df['dzi'].astype(np.float64)
#         df['dzj'] = df['dzj'].astype(np.float64)
#         df['dzk'] = df['dzk'].astype(np.float64)
#         fields = [Series(data[~np.isnan(data)])]
#         #print(fields)
#         self._field = AtomicField(df, field_values=fields)
#
#     def _init(self):
#         """
#         Perform some preliminary parsing so that future parsing of atoms, etc.
#         is easy. Also parse out metadata (comments).
#         """
#         typs = [int, float, float, float]
#         nat, ox, oy, oz = [typ(i) for typ, i in zip(typs, self[2].split())]
#         nx, dxi, dxj, dxk = [typ(i) for typ, i in zip(typs, self[3].split())]
#         ny, dyi, dyj, dyk = [typ(i) for typ, i in zip(typs, self[4].split())]
#         nz, dzi, dzj, dzk = [typ(i) for typ, i in zip(typs, self[5].split())]
#         self._dxi = dxi
#         self._dxj = dxj
#         self._dxk = dxk
#         self._dyi = dyi
#         self._dyj = dyj
#         self._dyk = dyk
#         self._dzi = dzi
#         self._dzj = dzj
#         self._dzk = dzk
#         self._nat = abs(nat)
#         self._nx = abs(nx)
#         self._ny = abs(ny)
#         self._nz = abs(nz)
#         self._ox = ox
#         self._oy = oy
#         self._oz = oz
#         self._volume_data_start = 6 + self._nat
#         self.meta = {'comments': self[0:2]}
#
#     def __init__(self, *args, label=None, field_type=None, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.label = label
#         self.field_type = field_type
#         self._init()
#
#
# def write_cube(path, universe, frame=None, field=None):
#     """
#     Generate a cube file or files from a given unvierse.
#
#     Only the path and universe are required; if no frame or field indices are
#     provided, this function will write all fields in the given universe. If
#     multiple frames are desired, pass a list of frame indices; the field
#     argument should then be a list of the same length with the corresponding
#     field indices or none (in which case the first field for each frame will be
#     written). If multple fields per frame (multiple) is desired, use a list
#     for frames as previously, but use a list of lists for fields where the
#     outer list has the same length as the frame argument list and each "sublist"
#     now contains the indices of the fields specific to that frame to be written.
#
#     Args:
#         path (str): Directory or file path
#         universe (:class:`~exatomic.universe.Universe`): Universe to pull data from
#         frame (list or int): If only a specific frame is needed
#         field (list or int): If only a specific field is needed
#
#     Note:
#         "Indices" in the above description always refer to dataframe index.
#         Also always writes in atomic units.
#     """
#     if isinstance(frame, list) and isinstance(field, list):
#         if isinstance(field[0], list):
#             raise NotImplementedError('frame=[], field=[[]]')
#         else:
#             raise NotImplementedError('frame=[], field=[]')
#     elif isinstance(frame, list) and field is None:
#         raise NotImplementedError('frame=[], field=None')
#     elif frame is None and field is None:
#         _write_first_field_of_each_frame(path, universe)
#         return path
#     else:
#         raise TypeError('Unsupported argument types ({}, {}), see docstring.'.format(type(frame), type(field)))
#
# def _write_first_field_of_each_frame(path, universe):
#     """
#     Writes the first field of every frame of the universe in a given directory.
#     """
#     raise NotImplementedError('Started but not completed')
#     if not os.path.isdir(path):
#         raise NotADirectoryError('Path {}, is not a directory or does not exist!'.format(path))
#     framelist = universe.frame.index
#     padding = len(str(len(framelist)))
#     for frame in framelist:
#         atom = universe.atom[(universe.atom['frame'] == frame), ['symbol', 'x', 'y', 'z']].copy()
#         mapper = symbol_to_z()
#         atom['Z'] = atom['symbol'].map(mapper)
#         atom['electrons'] = 1.0
#         del atom['symbol']
#         field_data = universe.field[universe.field['frame'] == frame]
#         idx = field_data.index[0]
#         fdata = field_data.ix[idx, ['ox', 'oy', 'oz', 'nx', 'ny', 'nz',
#                                     'xi', 'xj', 'xk', 'yi', 'yj', 'yk',
#                                     'zi', 'zj', 'zk']]
#         ox, oy, oz, nx, ny, nz, xi, xj, xk, yi, yj, yk, zi, zj, zk = fdata
#         nat = universe.frame.ix[frame, 'atom_count']
#         field = _reshape_to_cube(universe.field_values[idx], nx, ny, nz)
#         name = mkp(path, str(frame).zfill(padding) + '.cube')
#         with open(name, 'w') as f:
#             f.write(str(universe) + '\n')
#             f.write('frame: {}\n'.format(frame))
#             f.write('{} {} {} {}\n'.format(nat, ox, oy, oz))
#             f.write('{} {} {} {}\n'.format(nx, xi, xj, xk))
#             f.write('{} {} {} {}\n'.format(ny, yi, yj, yk))
#             f.write('{} {} {} {}\n'.format(nz, zi, zj, zk))
#             atom.to_csv(f, sep=' ')
#             fiel.to_csv(f, sep=' ')
#
# def _reshape_to_cube(field_values):
#     pass
