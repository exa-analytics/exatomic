# -*- coding: utf-8 -*-
'''
XYZ File I/O
====================
'''
from linecache import getline
from exa import _pd as pd
from exa import _np as np
from exa import _os as os
from exa.config import Config
if Config.numba:
    from exa.jitted.indexing import idxs_from_starts_and_counts
    from exa.jitted.deduplication import array1d_with_offset
else:
    from exa.algortihms.deduplication import array1d_with_offset
    from exa.algorithms.indexing import idxs_from_starts_and_counts
from atomic import Length, Universe, Isotope
from atomic.frame import minimal_frame


def read_xyz(path, unit='A', label=True, **kwargs):
    '''
    Reads any type of XYZ or XYZ like file.

    Units will be converted from their source (Angstrom default) to atomic
    units (Bohr: "au") as used by the atomic package. Note the `official`_
    XYZ specification below (example for benzene):

    .. code-block:: bash

        12
        benzene example
        C        0.00000        1.40272        0.00000
        H        0.00000        2.49029        0.00000
        C       -1.21479        0.70136        0.00000
        H       -2.15666        1.24515        0.00000
        C       -1.21479       -0.70136        0.00000
        H       -2.15666       -1.24515        0.00000
        C        0.00000       -1.40272        0.00000
        H        0.00000       -2.49029        0.00000
        C        1.21479       -0.70136        0.00000
        H        2.15666       -1.24515        0.00000
        C        1.21479        0.70136        0.00000
        H        2.15666        1.24515        0.00000

    Args:
        path (str): String file path of xyz like file
        unit (str): Unit of length used (default: Angstrom)
        label (bool): If True, include a numeric label for each atom (default False)

    Returns:
        universe (:class:`~atomic.universe.Universe`): Containing One and comments

    .. _official: http://openbabel.org/wiki/XYZ_%28format%29
    '''
    df = pd.read_csv(path, header=None, skip_blank_lines=False, delim_whitespace=True,
                     names=['symbol', 'x', 'y', 'z'])                   # Parse in the data
    natdf = df.loc[df[['y', 'z']].isnull().all(1)].dropna(how='all')    # Figure out frames & nats
    nat_indices = array1d_with_offset(natdf.index.values)
    natdf = natdf[natdf.index.isin(nat_indices)]
    comments = natdf[natdf.index.isin(natdf.index.values + 1) &
                     ~natdf['symbol'].isnull().all()].index.values
    starts = natdf.index.values + 2
    counts = natdf['symbol'].values.astype(np.int64)
    frame, lbl, indices = idxs_from_starts_and_counts(starts, counts)
    df = df[df.index.isin(indices)]
    df[['x', 'y', 'z']] = df[['x', 'y', 'z']].astype(np.float64)
    df['frame'] = frame
    if label:
        df['label'] = lbl
    df.reset_index(drop=True, inplace=True)
    comments = {num: getline(path, num) for num in comments}
    df[['x', 'y', 'z']] *= Length[unit, 'au']
    if all(df['symbol'].str.isdigit()):
        df.columns = ['Z', 'x', 'y', 'z']
        df['Z'] = df['Z'].astype(np.int64)
        df['symbol'] = df['Z'].map(Isotope.lookup_symbol_by_Z)
    meta = {'file': path, 'comments': comments}
    basename = os.path.basename(path)
    name = os.path.splitext(basename)[0] if 'name' not in kwargs else kwargs['name']
    description = path if 'description' not in kwargs else kwargs['description']
    return Universe(name=name, description=description, atom=df, meta=meta, **kwargs)


def write_xyz(universe, path, unit='A', trajectory=True):
    '''
    Args:
        universe (:class:`~atomic.universe.Universe`): Atomic universe containing atom table
        path (str): Directory or file path
        unit (str): Output unit of length
        trajectory (bool): Generate a single XYZ file for each frame or one trajectory XYZ file (default)
    '''
    symbol = universe.atom['symbol'].values
    x = (universe.atom['x'] * Length['au', unit]).values
    y = (universe.atom['y'] * Length['au', unit]).values
    z = (universe.atom['z'] * Length['au', unit]).values
    atom_count = universe.frame['atom_count']
    if as_trajectory:
        write_xyz_trajectory(path, symbol, x, y, z)
    else:
        write_xyz_files(path, symbol, x, y, z, atom_count)


def write_xyz_trajectory(path, symbol, x, y, z, atom_count):
    '''
    '''
    raise NotImplementedError()


def write_xyz_files(path, symbol, x, y, z, atom_count):
    '''
    '''
    raise NotImplementedError()
