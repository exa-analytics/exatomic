# -*- coding: utf-8 -*-

from atomic import _pd as pd
from atomic import _os as os
try:
    from atomic.algorithms.jitted import expand
except ImportError:
    from atomic.algorithms.nonjitted import expand

columns = ['symbol', 'x', 'y', 'z']

def read_xyz(fp, unit='A', metadata={}, **kwargs):
    '''
    Reads an xyz or xyz trajectory file

    Args
        path (str): file path
        unit (str): unit of atomic positions (optional - default 'A')
        metadata (dict): metadata as key, value pairs
        **kwargs: only if using with exa content management system

    Return
        unikws (dict): dataframes containing 'frame' and 'one' body data

    See Also
        :class:`atomic.container.Universe`
    '''
    df = pd.read_csv(fp, names=columns, delim_whitespace=True, skip_blank_lines=False)
    unikws = _parse_xyz(df, unit)
    #unikws['metadata'].update(metadata)
    return df

def _parse_xyz(df, unit):
    nats = df.loc[df[['x', 'y', 'z']].isnull().all(axis=1)]
    ## This method for getting number of atoms support variable nat frames
    #nats = df.loc[df[['x', 'y', 'z']].isnull().all(axis=1) & df.symbol.str.isdigit(), 'symbol'].astype(int)
    ## Get the xyz and symbol data
    #framedx, onedx, indices = expand(nats.index.values + 2, nats.values)
    #one = df.loc[df.index.isin(indices), ['symbol', 'x', 'y', 'z']]
    #one.loc[:, ['x', 'y', 'z']] = one.loc[:, ['x', 'y', 'z']].astype(float)
    #one.loc[:, ['x', 'y', 'z']] *= Length[unit, 'a0']
    #one['frame'] = framedx
    #one['one'] = onedx
    #one.set_index(['frame', 'one'], inplace=True)
    ## Get metadata
    #comment_idx = nats.index + 1
    #metadata = {'comments': df.loc[comment_idx].dropna().apply(join_row, on=' ', axis=1).to_dict()}
    ## Generate the framedf
    #frame = generate_minimal_framedf_from_onedf(one)
    #return {'frame': frame, 'one': one, 'metadata': metadata}

def write_xyz(fp):
    pass
