# -*- coding: utf-8 -*-
'''
Cube File Parsing and Composing
=============================================

'''
from exa import _pd as pd

from atomic import Universe
from atomic import Isotope
#from atomic.algorithms.nonjitted import generate_minimal_framedf_from_onedf as _gen_fdf
from atomic.frame import _min_frame_from_atom

def read_cubes(paths, frames=None, fldidxs=None, labels=None,
               universe=None, metadata={}, **kwargs):
    '''
    Args
        paths (str or list): str to path or list of paths to cube files
        frames (int or list): frame of interest or list of same length as paths
        fldidxs (int or list): starting fiidx or list of same length as paths
        labels (list): List of same length as paths
        universe (:lclass:`~atomic.container.Universe`): universe to which field belongs (see Note)

    Returns
        unikws: New universe or modified universe

    Note
        In order for the cube files to be attached correctly to an existing universe, all
        indices (frame, fldidx) must be specified and the original universe
        must be provided. Otherwise it will create a universe with the given information.

    Warning
        If attaching to an existing universe only the cubedata and cube tables will
        be updated! The frame and one body tables will not be modified!
    '''
    if type(paths) != list:
        paths = [paths]
    fldlist = []
    fldmlist = []
    unikws = {}

    cfidx = None
    if frames is None:
        if universe is None:
            frame = 0
        else:
            frame = universe.framelist[0]
    elif type(frames) is int:
        frame = frames
    elif type(frames) is list:
        cfidx = True
        frame = frames[0]

    cvidx = None
    if fldidxs is None:
        if universe is None:
            fldidx = 0
        else:
            fldidx = universe.fieldm.loc[frame].index.get_level_values('fldidx')[-1] + 1
    elif type(fldidxs) is int:
        fldidx = fldidxs
    elif type(fldidxs) is list:
        cfldidx = True
    clidx = None
    if labels:
        clidx = True
        label = labels[0]
    else:
        label = None

    onelist = []
    framelist = []

    for i, fl in enumerate(paths):
        df = pd.read_csv(fl, delim_whitespace=True, header=None,
                         skiprows=[0, 1], names=range(6), dtype=float)

        nat = int(df.iloc[0, 0])   # Always needed

        if cfidx:
            frame = frames[i]
        else:
            fldidx += 1
        if cvidx:
            fldidx = fldidxs[i]
        if clidx:
            label = labels[i]

        index = (frame, fldidx)    # Generate field data
        flddat, convert_xyz = _gen_flddat(df.iloc[0:4],frame, fldidx, label)

                                   # Generate the cube entry
        field = _gen_field(df.iloc[nat + 4:], frame, fldidx)
        fldlist.append(field)
        fldmlist.append(flddat)

        if cfidx:
            onelist.append(_gen_atom(df.iloc[4:nat + 4], convert_xyz, frame))
            framelist.append(_min_frame_from_atom(onelist[-1]))
        else:
            if i == 0:             # Only need one of these dfs each
                unikws['one'] = _gen_atom(df.iloc[4:nat + 4], convert_xyz, frame)
                mnikws['frame'] = _gen_fdf(unikws['one'])

    if universe is None:
        if cfidx:
            unikws['atom'] = pd.concat(onelist)
            unikws['frame'] = pd.concat(framelist)
        unikws['fieldm'] = pd.concat(flddatlist)
        unikws['field'] = pd.concat(fldlist)
        ##unikws['metadata'].update({'paths': paths})
        unikws['metadata'] = {'paths': paths}
        unikws.update(kwargs)
        uni = Universe(**unikws)
#        uni.compute_two_body(inplace=True)
        return uni
    else:
        o = ['frame', 'volidx']
        if universe.volume is None:
            universe.volume = volume
        else:
            if index in universe.volume.index:
                universe.volume.set_value(index, 'mag', volume.values)
            else:
                universe.volume = universe.volume.append(volume).reset_index(o).sort_values(o).set_index(o)
        if universe.voldat is None:
            universe.voldat = voldat
        else:
            mymap = voldat.to_dict()
            if index in universe.voldat.index:
                universe.voldat.iloc[index].map(mymap)
            else:
                universe.voldat = universe.voldat.append(voldat).reset_index(o).sort_values(o).set_index(o)
        return universe


def _gen_flddat(data, frame, fldidx, label):
    '''
    Generates 'fieldm' dataframe (field metadata)
    '''
    convert_xyz = False
    origin = data.iloc[0, 1:4].values
    v = data.iloc[1:].unstack().dropna().values
    flddatdict = {'ox': origin[0], 'oy': origin[1], 'oz': origin[2],
                  'nx': v[0], 'dxi': v[3], 'dxj': v[4], 'dxk': v[5],
                  'ny': v[1], 'dyi': v[6], 'dyj': v[7], 'dyk': v[8],
                  'nz': v[2], 'dzi': v[9], 'dzj': v[10], 'dzk': v[11],
                  'label': label}
    df = pd.DataFrame(flddatdict, index=[0])
    print(df)
    # Check units
    for i in ['x', 'y', 'z']:
        if any(df['n' + i]) < 0:
            convert_xyz = True
            df['n' + i] *= -1
            df['d' + i + 'i'] *= Length['A', 'a0']
            df['d' + i + 'j'] *= Length['A', 'a0']
            df['d' + i + 'k'] *= Length['A', 'a0']
    df['frame'] = frame
    df['fldidx'] = fldidx
    df.set_index(['frame', 'fldidx'], inplace=True)
    return df, convert_xyz


def _gen_atom(onedf, convert_xyz, frame):
    '''
    Generates 'one' body dataframe
    '''
    df = onedf.loc[:, [0, 2, 3, 4]].reset_index(drop=True)
    df.index.names = ['one']
    df.columns = ['symbol', 'x', 'y', 'z']
    if convert_xyz:
        df[['x', 'y', 'z']] *= Length['A', 'a0']
    i = Isotope.Z_to_symbol_map
    df['symbol'] = df['symbol'].map(i.groupby(i.index).first())
    df['frame'] = frame
    df.set_index('frame', append=True, inplace=True)
    df = df.reorder_levels(['frame', 'one'])
    return df


def _gen_volume(data, frame, fldidx):
    '''
    Generates 'volume' dataframe
    '''
    df = data.stack().dropna().reset_index(drop=True).to_frame()
    df.columns = ['mag']
    df.index.names = ['data']
    df['frame'] = frame
    df['fldidx'] = fldidx
    # Next line of code is the slow step (but not due to inplace=True)
    df.set_index(['frame', 'fldidx'], append=True, inplace=True)
    df = df.reorder_levels(['frame', 'fldidx', 'data'])
    return df
