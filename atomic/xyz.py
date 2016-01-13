# -*- coding: utf-8 -*-
'''
XYZ File I/O
====================
'''
from linecache import getline
from exa import Config
from exa import _pd as pd
if Config.numba:
    from exa.jitted.indexing import idxs_from_starts_and_counts
else:
    from exa.algorithms.indexing import idxs_from_starts_and_counts
from atomic import Length, Universe
from atomic.one import One


def read_xyz(path):
    '''
    Reads any type of XYZ or XYZ like file.

    Units will be converted from their source (Angstrom default) to atomic
    units (Bohr: "au") as used by the atomic package.

    Args:
        path (str): String file path of xyz like file

    Returns:
        universe (:class:`~atomic.universe.Universe`): Containing One and comments
    '''
    df = pd.read_csv(path, header=None, skip_blank_lines=False, delim_whitespace=True, names=['symbol', 'x', 'y', 'z'])
    nat_lines = df.loc[df[['y', 'z']].isnull().all(1)].dropna(how='all')[['symbol', 'x']]       # Filter nat rows
    comment_lines = df.loc[df.index.isin(nat_lines.index + 1) & ~df.isnull().all(1)].index + 1  # Filter comment rows
    starts = nat_lines.index.values.astype(np.int) + 2                                          # Use this to generate
    counts = nat_lines['symbol'].values.astype(np.int)                                          # indexes
    frame, atom, indexes = _idxs_from_starts_and_counts(starts, counts)
    df = One(df.loc[df.index.isin(indexes)])                                                    # Create one df and set index
    df.index = pd.MultiIndex.from_arrays((frame, atom), names=['frame', 'atom'])
    comments = {num: getline(path, num) for num in comment_lines}                               # Get comments
    if nat_lines['x'].isnull().all():                                                           # Finally handle unit conversions
        df[['x', 'y', 'z']] *= Length['A', 'au']
    else:
        for i, unit in enumerate(nat_lines['x'].tolist()):
            df.loc[i, ['x', 'y', 'z']] *= Length.from_alias(unit, 'au')
    metadata = {'file': path, 'comments': comments}                                             # Generate metadata
    return Universe(one=df, metadata=metadata)
