# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Universe trait functions
#########################
"""
##########
# traits #
##########
import numpy as np
import pandas as pd
from exatomic.base import sym2radius, sym2color


def atom_traits(df, atomcolors=None, atomradii=None, atomlabels=None):
    """
    Get atom table traits. Atomic size (using the covalent radius) and atom
    colors (using the common `Jmol`_ color scheme) are packed as dicts and
    obtained from the static data in exa.

    .. _Jmol: http://jmol.sourceforge.net/jscolors/
    """
    atomlabels = pd.Series(dtype=str) if atomlabels is None else pd.Series(atomlabels, dtype=str)
    atomcolors = pd.Series(dtype=str) if atomcolors is None else pd.Series(atomcolors, dtype=str)
    atomradii = pd.Series(dtype=float) if atomradii is None else pd.Series(atomradii, dtype=float)
    traits = {}
    cols = ['x', 'y', 'z']
    grps = df.groupby('frame')
    for col in cols:
        ncol = 'atom_' + col
        traits[ncol] = grps.apply(
            lambda y: y[col].to_json(
            orient='values', double_precision=3)
            ).to_json(orient="values").replace('"', '')
    syms = grps.apply(lambda g: g['symbol'].cat.codes.values)
    symmap = {i: v for i, v in enumerate(df['symbol'].cat.categories)
              if v in df.unique_atoms}
    unq = df['symbol'].astype(str).unique()
    cov_radii = {k: sym2radius[k][0] for k in unq}
    van_radii = {k: sym2radius[k][1] for k in unq}
    colors = {k: sym2color[k] for k in unq}
    labels = symmap
    colors.update(atomcolors)
    cov_radii.update(atomradii)
    van_radii.update(atomradii)
    labels.update(atomlabels)
    traits['atom_s'] = syms.to_json(orient='values')
    traits['atom_cr'] = {i: cov_radii[v] for i, v in symmap.items()}
    traits['atom_vr'] = {i: van_radii[v] for i, v in symmap.items()}
    traits['atom_c'] = {i: colors[v] for i, v in symmap.items()}
    traits['atom_l'] = labels
    return traits


def field_traits(df):
    """Get field table traits."""
    df['frame'] = df['frame'].astype(int)
    df['nx'] = df['nx'].astype(int)
    df['ny'] = df['ny'].astype(int)
    df['nz'] = df['nz'].astype(int)
    grps = df.groupby('frame')
    fps = grps.apply(lambda x: x[['ox', 'oy', 'oz',
                                  'nx', 'ny', 'nz',
                                  'fx', 'fy', 'fz']].T.to_dict()).to_dict()
    try: idxs = list(map(list, grps.groups.values()))
    except: idxs = [list(grp.index) for i, grp in grps]
    return {'field_v': [f.to_json(orient='values',
                        double_precision=5) for f in df.field_values],
            'field_i': idxs,
            'field_p': fps}


def two_traits(uni):
    """Get two table traitlets."""
    if not hasattr(uni, "atom_two"):
        raise AttributeError("for the catcher")
    if "frame" not in uni.atom_two.columns:
        uni.atom_two['frame'] = uni.atom_two['atom0'].map(uni.atom['frame'])
    lbls = uni.atom.get_atom_labels().astype(int)
    df = uni.atom_two
    bonded = df.loc[df['bond'] == True, ['atom0', 'atom1', 'frame']]
    lbl0 = bonded['atom0'].map(lbls)
    lbl1 = bonded['atom1'].map(lbls)
    lbl = pd.concat((lbl0, lbl1), axis=1)
    lbl['frame'] = bonded['frame']
    bond_grps = lbl.groupby('frame')
    frames = df['frame'].unique().astype(np.int64)
    b0 = np.empty((len(frames), ), dtype='O')
    b1 = b0.copy()
    for i, frame in enumerate(frames):
        try:
            b0[i] = bond_grps.get_group(frame)['atom0'].astype(np.int64).values
            b1[i] = bond_grps.get_group(frame)['atom1'].astype(np.int64).values
        except Exception:
            b0[i] = []
            b1[i] = []
    b0 = pd.Series(b0).to_json(orient='values')
    b1 = pd.Series(b1).to_json(orient='values')
    del uni.atom_two['frame']
    return {'two_b0': b0, 'two_b1': b1}


def frame_traits(uni):
    """Get frame table traits."""
    # ASSUME SIMPLE CUBIC CELL this is a hack for now.
    if 'xi' in uni.frame.columns:
        return {'frame__a': uni.frame['xi'].max()}
    return {}


def tensor_traits(uni):
    grps = uni.tensor.groupby('frame')
    try: idxs = list(map(list, grps.groups.values()))
    except: idxs = [list(grp.index) for i, grp in grps]
    return {'tensor_d': grps.apply(lambda x: x.T.to_dict()).to_dict(), 'tensor_i': idxs}

def freq_traits(uni):
    grps = uni.frequency.groupby('freqdx')
    try: idxs = list(grps.groups.keys())
    except: idxs = [list(grp.index) for i, grp in grps]
    atoms = uni.atom.last_frame
    cols = [['x', 'y', 'z'],
            ['dx', 'dy', 'dz']]
    # use a max of 50% of the normal mode
    max_mult = 5
    freq_x = []
    freq_y = []
    freq_z = []
    freq_s = []
    for fdx, data in grps:
        # generate a list that goes from 0% to 50% to -50% to 0% of the normal mode
        # movement this will generate a 'smooth' movement
        mults = np.concatenate([range(0, max_mult), range(max_mult, -max_mult, -1),
                                range(-max_mult, 0)])
        dfs = []
        for idx, mult in enumerate(mults):
            df = atoms.copy()
            # distort the molecule by a percentage of the normal modes
            df[cols[0]] += mult/10 * data[cols[1]].values
            df['frame'] = idx
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        grps = df.groupby('frame')
        traits = {}
        for col in cols[0]:
            traits['freq_'+col] = grps.apply(
                    lambda y: y[col].to_json(
                    orient='values', double_precision=3)
                    ).to_json(orient="values").replace('"', '')
        freq_x.append(traits['freq_x'])
        freq_y.append(traits['freq_y'])
        freq_z.append(traits['freq_z'])
        syms = grps.apply(lambda g: g['symbol'].cat.codes.values)
        freq_s.append(syms.to_json(orient='values'))
    traits = {}
    # put together the vibrational normal modes
    traits['freq_d'] = grps.apply(lambda x: x.T.to_dict()).to_dict()
    # put together the vibrational frequency indexes
    traits['freq_i'] = idxs
    # put together the atomic positions for animating the normal modes
    # we use the replace function as when the values are appended together
    # they are strings and not lists so you cannot use the
    # np.concatenate function
    traits['freq_x'] = ''.join(freq_x).replace(']][[', '],[')
    traits['freq_y'] = ''.join(freq_y).replace(']][[', '],[')
    traits['freq_z'] = ''.join(freq_z).replace(']][[', '],[')
    # the atomic symbols
    traits['freq_s'] = ''.join(freq_s).replace(']][[', '],[')
    return traits


def uni_traits(uni, atomcolors=None, atomradii=None, atomlabels=None):
    """Get Universe traits."""
    unargs = {}
    fields, tensors, freq = [], None, None
    if hasattr(uni, 'frame'):
        unargs.update(frame_traits(uni))
    if hasattr(uni, 'atom'):
        unargs.update(atom_traits(uni.atom, atomcolors, atomradii, atomlabels))
    if hasattr(uni, 'atom_two'):
        unargs.update(two_traits(uni))
    if hasattr(uni, 'field'):
        unargs.update(field_traits(uni.field))
        fields = ['null'] + unargs['field_i'][0]
    if hasattr(uni, 'tensor'):
        unargs.update(tensor_traits(uni))
        tensors = unargs['tensor_i'][0]
    if hasattr(uni, 'frequency'):
        unargs.update(freq_traits(uni))
        freq = unargs['freq_i']
    return unargs, fields, tensors, freq

