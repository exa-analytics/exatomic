
# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Universe trait functions
#########################
"""
##########
# traits #
##########
import re
import numpy as np
import pandas as pd

from exatomic.base import sym2radius, sym2color



def atom_traits(df, atomcolors=None, atomradii=None):
    """
    Get atom table traits. Atomic size (using the covalent radius) and atom
    colors (using the common `Jmol`_ color scheme) are packed as dicts and
    obtained from the static data in exa.

    .. _Jmol: http://jmol.sourceforge.net/jscolors/
    """
    # Implement logic to automatically choose
    # whether or not to create labels
    labels = True
    atomcolors = pd.Series() if atomcolors is None else pd.Series(atomcolors)
    atomradii = pd.Series() if atomradii is None else pd.Series(atomradii)
    traits = {}
    cols = ['x', 'y', 'z']
    if labels:
        cols.append('l')
        if 'tag' in df.columns: df['l'] = df['tag']
        else: df['l'] = df['symbol'] + df.index.astype(str)
    grps = df.groupby('frame')
    for col in cols:
        ncol = 'atom_' + col
        if col == 'l':
            labels = grps.apply(lambda y: y[col].to_json(orient='values')
                ).to_json(orient="values")
            repl = {r'\\': '', '"\[': '[', '\]"': ']'}
            replpat = re.compile('|'.join(repl.keys()))
            repl = {'\\': '', '"[': '[', ']"': ']'}
            traits['atom_l'] = replpat.sub(lambda m: repl[m.group(0)],
                                           labels)
            del df['l']
        else:
            traits[ncol] = grps.apply(
                lambda y: y[col].to_json(
                orient='values', double_precision=3)
                ).to_json(orient="values").replace('"', '')
    syms = grps.apply(lambda g: g['symbol'].cat.codes.values)
    symmap = {i: v for i, v in enumerate(df['symbol'].cat.categories)
              if v in df.unique_atoms}
    unq = df['symbol'].astype(str).unique()
    radii = {k: sym2radius[k] for k in unq}
    colors = {k: sym2color[k] for k in unq}
    colors.update(atomcolors)
    radii.update(atomradii)
    traits['atom_s'] = syms.to_json(orient='values')
    # TODO : This multiplication by 0.5 is in a bad place
    traits['atom_r'] = {i: 0.5 * radii[v] for i, v in symmap.items()}
    traits['atom_c'] = {i: colors[v] for i, v in symmap.items()}
    return traits

def field_traits(df):
    """Get field table traits."""
    df['frame'] = df['frame'].astype(int)
    df['nx'] = df['nx'].astype(int)
    df['ny'] = df['ny'].astype(int)
    df['nz'] = df['nz'].astype(int)
    if not all((col in df.columns for col in ['fx', 'fy', 'fz'])):
        for d, l in [('x', 'i'), ('y', 'j'), ('z', 'k')]:
            df['f'+d] = df['o'+d] + (df['n'+d] - 1) * df['d'+d+l]
    grps = df.groupby('frame')
    fps = grps.apply(lambda x: x[['ox', 'oy', 'oz',
                                  'nx', 'ny', 'nz',
                                  'fx', 'fy', 'fz']].T.to_dict()).to_dict()
    try: idxs = list(map(list, grps.groups.values()))
    except: idxs = [list(grp.index) for i, grp in grps]
    #vals = [f.tolist() for f in df.field_values]
    # shape0 = len(df.field_values)
    # shape1 = len(df.field_values[0])
    # vals = np.empty((shape0, shape1), dtype=np.float32)
    # for i in range(shape0):
    #     vals[i] = df.field_values[i].values
    # '[' + ','.join(listcomp) + ']'
    vals = [f.to_json(orient='values',
                      double_precision=5) for f in df.field_values]
    return {'field_v': vals, 'field_i': idxs, 'field_p': fps}

#def two_traits(df, lbls):
def two_traits(uni):
    """Get two table traitlets."""
    if not hasattr(uni, "atom_two"):
        raise AttributeError("for the catcher")
    if "frame" not in uni.atom_two.columns:
        uni.atom_two['frame'] = uni.atom_two['atom0'].map(uni.atom['frame'])
    lbls = uni.atom.get_atom_labels()
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
    if not hasattr(uni, 'frame'): return {}
    # TODO :: Implement me!!
    return {}




def uni_traits(uni, atomcolors=None, atomradii=None):
    """Get Universe traits."""
    unargs = {}
    fields = []
    if hasattr(uni, 'atom'):
        unargs.update(atom_traits(uni.atom, atomcolors, atomradii))
    if hasattr(uni, 'atom_two'):
        unargs.update(two_traits(uni))
    if hasattr(uni, 'field'):
        unargs.update(field_traits(uni.field))
        fields = ['null'] + unargs['field_i'][0]
    if hasattr(uni, 'tensor'):
        unargs.update({'tensor_d': uni.tensor.groupby('frame').apply(
            lambda x: x.T.to_dict()).to_dict()})
    return unargs, fields
