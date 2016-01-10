# -*- coding: utf-8 -*-
'''
Cube File Parsing and Composing
=============================================

'''
# Hacky imports
import sys
sys.path.insert(0, '/home/tjd/Programs/analytics-exa/exa')

from exa import Isotope
from atomic import _pd as pd
from atomic.algorithms.nonjitted import generate_minimal_framedf_from_onedf as _gen_fdf


def read_cubes(paths, frames=None, volidxs=None, labels=None,
               universe=None, metadata={}, **kwargs):
    '''
    Args
        paths (str or list): str to path or list of paths to cube files
        frames (int or list): frame of interest or list of same length as paths
        volidxs (int or list): starting volidx or list of same length as paths
        labels (list): List of same length as paths
        universe (:class:`~atomic.container.Universe`): universe to which field belongs (see Note)

    Returns
        unikws: New universe or modified universe

    Note
        See parse_cube() for the singular case.
        In order for the cube files to be attached correctly to a universe, all
        indices (frame, volidx) must be specified and the original universe
        must be provided. Otherwise it will create a universe with the given information.

    Warning
        If attaching to an existing universe only the cubedata and cube tables will
        be updated! The frame and one body tables will not be modified!
    '''
    if type(paths) != list:
        paths = [paths]
    vollist = []
    voldatlist = []
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
    if volidxs is None:
        if universe is None:
            volidx = 0
        else:
            volidx = universe.voldat.loc[frame].index.get_level_values('volidx')[-1] + 1
    elif type(volidxs) is int:
        volidx = volidxs
    elif type(volidxs) is list:
        cvidx = True
    clidx = None
    if labels:
        clidx = True
        label = labels[0]
    else:
        label = None

    print(frame)
    print(label)
    print(volidx)

    onelist = []
    framelist = []

    for i, fl in enumerate(paths):
        df = pd.read_csv(fl, delim_whitespace=True, header=None,
                         skiprows=[0, 1], names=range(6), dtype=float)

        nat = int(df.iloc[0, 0])   # Always needed

        if cfidx:
            frame = frames[i]
        else:
            volidx += 1
        if cvidx:
            volidx = volidxs[i]
        if clidx:
            label = labels[i]

        index = (frame, volidx)    # Generate voldata
        voldat, convert_xyz = _gen_voldat(df.iloc[0:4],frame, volidx, label)

                                   # Generate the cube entry
        volume = _gen_volume(df.iloc[nat + 4:], frame, volidx)
        vollist.append(volume)
        voldatlist.append(voldat)

        if cfidx:
            onelist.append(_gen_one(df.iloc[4:nat + 4], convert_xyz, frame))
            framelist.append(_gen_fdf(onelist[-1]))
        else:
            if i == 0:             # Only need one of these dfs each
                unikws['one'] = _gen_one(df.iloc[4:nat + 4], convert_xyz, frame)
                unikws['frame'] = _gen_fdf(unikws['one'])

    if universe is None:
        if cfidx:
            unikws['one'] = pd.concat(onelist)
            unikws['frame'] = pd.concat(framelist)
        unikws['voldat'] = pd.concat(voldatlist)
        unikws['volume'] = pd.concat(vollist)
        unikws['metadata'] = {'paths': paths}
        unikws.update(kwargs)
        #uni = Universe(**unikws)
        #uni.get_two_body_properties()
        return unikws
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


def _gen_voldat(data, frame, volidx, label):
    '''
    Generates 'voldat' dataframe (volume metadata)
    '''
    convert_xyz = False
    origin = data.iloc[0, 1:4].values
    v = data.iloc[1:].unstack().dropna().values
    voldatdict = {'ox': origin[0], 'oy': origin[1], 'oz': origin[2],
                  'nx': v[0], 'dxi': v[3], 'dxj': v[4], 'dxk': v[5],
                  'ny': v[1], 'dyi': v[6], 'dyj': v[7], 'dyk': v[8],
                  'nz': v[2], 'dzi': v[9], 'dzj': v[10], 'dzk': v[11],
                  'label': label}
    df = pd.DataFrame(voldatdict, index=[0])
    # Check units
    for i in ['x', 'y', 'z']:
        if any(df['n' + i]) < 0:
            convert_xyz = True
            df['n' + i] *= -1
            df['d' + i + 'i'] *= Length['A', 'a0']
            df['d' + i + 'j'] *= Length['A', 'a0']
            df['d' + i + 'k'] *= Length['A', 'a0']
    df['frame'] = frame
    df['volidx'] = volidx
    df.set_index(['frame', 'volidx'], inplace=True)
    return df, convert_xyz


def _gen_one(onedf, convert_xyz, frame):
    '''
    Generates 'one' body dataframe
    '''
    distinct = onedf[0].astype(int).unique()
    nisos = [Isotope.get_element(int(znum), by='znum') for znum in distinct]
    isos = {iso.Z: iso.symbol for iso in nisos}
    df = onedf.loc[:, [0, 2, 3, 4]].reset_index(drop=True)
    df.index.names = ['one']
    df.columns = ['symbol', 'x', 'y', 'z']
    if convert_xyz:
        df[['x', 'y', 'z']] *= Length['A', 'a0']
    df['symbol'] = df['symbol'].map(isos)
    df['frame'] = frame
    df.set_index('frame', append=True, inplace=True)
    df = df.reorder_levels(['frame', 'one'])
    return df


def _gen_volume(data, frame, volidx):
    '''
    Generates 'volume' dataframe
    '''
    df = data.stack().dropna().reset_index(drop=True).to_frame()
    df.columns = ['mag']
    df.index.names = ['data']
    df['frame'] = frame
    df['volidx'] = volidx
    # Next line of code is the slow step (but not due to inplace=True)
    df.set_index(['frame', 'volidx'], append=True, inplace=True)
    df = df.reorder_levels(['frame', 'volidx', 'data'])
    return df
