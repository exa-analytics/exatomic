 #-*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Cube File Support
##########################
Cube files contain an atomic geometry and scalar field values corresponding to
a physical quantity.
"""
import os
import six
import numpy as np
import pandas as pd
from glob import glob
from exa import Series, TypedMeta
from exatomic import __version__, Atom, Editor, AtomicField, Frame, Universe
from exatomic.base import z2sym, sym2z

class Meta(TypedMeta):
    atom = Atom
    frame = Frame
    field = AtomicField


class Cube(six.with_metaclass(Meta, Editor)):
    """
    An editor for handling cube files. Assumes scalar field values are arranged
    with the x axis as the outer loop and the z axis as the inner loop.

    .. code-block:: python

        cube = Cube('my.cube')
        cube.atom                # Displays the atom dataframe
        cube.field               # Displays the field dataframe
        cube.field.field_values  # Displays the list of field values
        uni = cube.to_universe() # Converts the cube file editor to a universe
        UniverseWidget(uni)      # Renders the cube file

    Warning:
        Be sure your cube is in atomic units.
    """
    def parse_atom(self):
        """
        Parse the :class:`~exatomic.atom.Atom` object from the cube file in place.
        """
        nat = abs(int(self[2].split()[0]))
        names = ['Z', 'Zeff', 'x', 'y', 'z']
        df = self.pandas_dataframe(6, nat + 6, names)
        df['symbol'] = df['Z'].map(z2sym).astype('category')
        df['label'] = range(nat)
        df['frame'] = 0
        self.atom = Atom(df)

    def parse_field(self):
        """
        Parse the scalar field into an :class:`~exatomic.core.field.AtomicField`.

        Note:
            The :class:`~exatomic.core.field.AtomicField` tracks both the
            field parameters (i.e. information about the discretization and shape of
            the field's spatial points) as well as the field values (at each of
            those points in space). See :meth:`~exatomic.algorithms.orbital_util.make_fps`
            for more details.
        """
        self.meta = {'comments': self[:2]}
        typs = [int, float, float, float]
        nat, ox, oy, oz = [typ(i) for typ, i in zip(typs, self[2].split())]
        nx, dxi, dxj, dxk = [typ(i) for typ, i in zip(typs, self[3].split())]
        ny, dyi, dyj, dyk = [typ(i) for typ, i in zip(typs, self[4].split())]
        nz, dzi, dzj, dzk = [typ(i) for typ, i in zip(typs, self[5].split())]
        nat, nx, ny, nz = abs(nat), abs(nx), abs(ny), abs(nz)
        volstart = nat + 6
        if len(self[volstart].split()) < 5:
            if not len(self[volstart + 1].split()) < 5:
                volstart += 1
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
        Make a cube file format Editor from a given field in a
        :class:`~exatomic.core.universe.Universe`.

        Args:
            uni (:class:`~exatomic.core.universe.Universe`): a universe
            idx (int): field index in :class:`~exatomic.core.field.AtomicField`
            name (str): description for comment line
            frame (int): frame index in :class:`~exatomic.core.atom.Atom`
        """
        name = '' if name is None else name
        frame = uni.atom.nframes - 1 if frame is None else frame
        hdr = '{} -- written by exatomic v{}\n\n'
        ffmt = ' {:> 12.6f}'
        flfmt = ('{:>5}' + ffmt * 3 + '\n').format
        if 'Z' not in uni.atom:
            uni.atom['Z'] = uni.atom['symbol'].map(sym2z)
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

    def __init__(self, *args, **kwargs):
        label = kwargs.pop("label", None)
        field_type = kwargs.pop("field_type", None)
        super(Cube, self).__init__(*args, **kwargs)
        self.label = label
        self.field_type = field_type



def uni_from_cubes(adir, verbose=False, ncubes=None, ext='cube'):
    """Put a bunch of cubes into a universe.

    .. code-block:: python

        uni = uni_from_cubes('/path/to/files/')       # Parse all cubes matching 'files/*cube'
        uni = uni_from_cubes('files/', ext='cub')     # Parse all cubes matching 'files/*cub'
        uni = uni_from_cubes('files/', verbose=True)  # Print file names when parsing
        uni = uni_from_cubes('files/', ncubes=5)      # Only parse the first 5 cubes
                                                      # sorted lexicographically by file name

    Args:
        verbose (bool): print file names when reading cubes
        ncubes (int): get only the first ncubes
        ext (str): file extension of cube files

    Returns:
        uni (:class:`exatomic.core.universe.Universe`)

    """
    if not adir.endswith(os.sep): adir += os.sep
    cubes = sorted(glob(adir + '*' + ext))
    if ncubes is not None:
        cubes = cubes[:ncubes]
    if verbose:
        for cub in cubes: print(cub)
    uni = Universe(atom=Cube(cubes[0]).atom)
    flds = [Cube(cub).field for cub in cubes]
    uni.add_field(flds)
    return uni
