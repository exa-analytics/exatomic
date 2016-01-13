# -*- coding: utf-8 -*-
'''
XYZ File I/O
====================
'''
from linecache import getline
from exa import Config
from exa import _pd as pd
from exa import _np as np
from exa import _os as os
from exa.relational.isotopes import Z_to_symbol_map
from exa.algorithms.deduplication import deduplicate_with_prev_offset
if Config.numba:
    from exa.jitted.indexing import idxs_from_starts_and_counts
else:
    from exa.algorithms.indexing import idxs_from_starts_and_counts
from atomic import Length, Universe


def read_xyz(path, unit='A'):
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

    Returns:
        universe (:class:`~atomic.universe.Universe`): Containing One and comments

    .. _official: http://openbabel.org/wiki/XYZ_%28format%29
    '''
    df = pd.read_csv(path, header=None, skip_blank_lines=False, delim_whitespace=True, names=['symbol', 'x', 'y', 'z'])
    nat_lines = df.loc[df[['y', 'z']].isnull().all(1)].dropna(how='all')[['symbol', 'x']]       # Filter nat rows
    nats = deduplicate_with_prev_offset(nat_lines.index)
    nat_lines = nat_lines.loc[nat_lines.index.isin(nats)]
    comment_lines = df.loc[df.index.isin(nat_lines.index + 1) & ~df.isnull().all(1)].index + 1  # Filter comment rows
    starts = nat_lines.index.values.astype(np.int) + 2                                          # Use this to generate
    counts = nat_lines['symbol'].values.astype(np.int)                                          # indexes
    frame, atom, indexes = idxs_from_starts_and_counts(starts, counts)
    df = df.loc[df.index.isin(indexes)]                                                         # Create one df and set index
    df[['x', 'y', 'z']] = df[['x', 'y', 'z']].astype(np.float)
    df.index = pd.MultiIndex.from_arrays((frame, atom), names=['frame', 'atom'])
    comments = {num: getline(path, num) for num in comment_lines}                               # Get comments
    df[['x', 'y', 'z']] *= Length[unit, 'au']
    if all(df['symbol'].str.isdigit()):
        df.columns = ['Z', 'x', 'y', 'z']
        df['Z'] = df['Z'].astype(np.int)
        df.loc[:, 'symbol'] = df['Z'].map(Z_to_symbol_map)
    meta = {'file': path, 'comments': comments}                                                 # Generate metadata
    name = os.path.basename(path)
    return Universe(name=name, description=path, atoms=df, meta=meta)


def write_xyz():
    raise NotImplementedError()
