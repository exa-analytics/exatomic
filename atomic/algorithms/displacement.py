# -*- coding: utf-8 -*-
'''
Computation of Displacement
=============================
'''
from exa import _np as np
from exa import _pd as pd
from exa import DataFrame


def absolute_msd(universe, ref_frame=None):
    '''
    Compute the mean squared displacement per atom per time with respect to the
    referenced position.

    Args:
        universe (:class:`~atomic.Universe`): The universe containing atomic positions
        ref_frame (int): Which frame to use as the reference (default first frame)

    Returns
        df (:class:`~pandas.DataFrame`): Time dependent displacement per atom
    '''
    index = 0
    if ref_frame is None:
        ref_frame = universe.frame.index[index]
    else:
        frames = universe.frame.index.values
        ref_frame = np.where(frames == ref_frame)
    coldata = universe.atom.ix[universe.atom['frame'] == ref_frame, ['label', 'symbol']]
    coldata = (coldata['label'].astype(str) + '_' + coldata['symbol']).values
    groups = universe.atom.groupby('label')
    msd = np.empty((groups.ngroups, ), dtype='O')
    for i, (label, group) in enumerate(groups):
        xyz = group[['x', 'y', 'z']].values
        msd[i] = ((xyz - xyz[0])**2).sum(axis=1)
    df = DataFrame.from_records(msd).T
    df.index = universe.frame.index.copy()
    df.columns = coldata
    return df
