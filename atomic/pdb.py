# -*- coding: utf-8 -*-

from atomic import _os as os
from atomic import _pd as pd
from atomic import _np as np
from atomic import sys
from requests import get as _get
import json as _json
from warnings import warn as _warn

try:
    from atomic.algorithms.jitted import expand as _expand
except ImportError:
    from atomic.algorithms.nonjitted import expand as _expand

from atomic.algorithms.nonjitted import generate_minimal_framedf_from_onedf as _gen_fdf
#Hacky imports
sys.path.insert(0, '/home/tjd/Programs/analytics-exa/exa')
from exa.utils import mkpath

_selfpath = os.path.abspath(__file__).replace('pdb.py', '')
_recpath = mkpath(_selfpath, 'static', 'pdb-min.json')

with open(_recpath, 'r') as f:
    records = _json.load(f)

def read_pdb(path, metadata={}, **kwargs):
    '''
    Reads a PDB file

    Args
        path (str): file path (local) or PDB identifier (remote)
        metadata (dict): metadata as key, value pairs
        **kwargs: only if using with exa content management system

    Return
        unikws (dict): dataframes containing 'frame', 'one' body and 'meta' data

    See Also
        :class:`atomic.container.Universe`
    '''
    flins = _path_handler(path)
    rds = _pre_process_pdb(flins)
    frdx, odx, ptls = _expand(
        np.arange(0, rds['nrf']),
        np.array([rds['nat'] for i in range(rds['nrf'])])
    )
    del ptls
    sepdict = _parse_pdb(flins, rds)
    one = _restruct_one(sepdict['ATOM'], frdx, odx)
    frame = _gen_fdf(one)
    unikws = {'frame': frame, 'one': one, 'metadata': {'text': sepdict['TEXT']}}
    unikws['metadata'].update(metadata)
    #try:
    #    unikws['aniso'] = _restruct_one(sepdict['ANISOU'], frdx, odx)
    #except KeyError:
    #    pass
    return unikws

def _path_handler(path):
    if os.sep in path:
        return _local_path(path)
    else:
        return _remote_path(path)

def _local_path(path):
    with open(path, 'r') as f:
        flins = f.read().splitlines()
    return flins


def _remote_path(path):
    url = 'http://www.rcsb.org/pdb/files/{}.pdb'.format(path.lower())
    flins = _get(url).text.splitlines()
    if 'FileNotFoundException' in flins[0]:
        raise FileNotFoundError('{}:{}'.format(url, ' may not exist.'))
    return flins


def _pre_process_pdb(flins):
    rds = {}
    recs = [line[:6].strip() for line in flins]
    for record in records:
        rds[record] = [sum([1 for rec in recs if record == rec]), 0]
    if rds['ENDMDL'][0]:
        nrf = rds['ENDMDL'][0]
    else:
        nrf = 1
    nat = rds['ATOM'][0] // nrf
    rds['nrf'] = nrf
    rds['nat'] = nat
    return rds

def _restruct_one(atomdf, frdx, odx):
    one = atomdf.loc[:, ['symbol', 'x', 'y', 'z', 'res.seq.']]
    one['frame'] = frdx
    one['one'] = odx
    one.set_index(['frame', 'one'], inplace=True)
    return one

def _parse_pdb(flins, rds):
    sepdict = {}
    for i, line in enumerate(flins):
        rec = line[:6].strip()
        if rec == 'ATOM' or rec == 'ANISOU' and records[rec]:
            if records[rec]:
                sepdict.setdefault(
                    rec, {i[2]: np.zeros(
                        (rds[rec][0], ), dtype=i[3]
                    ) for i in records[rec]}
                )
                sepdict[rec].setdefault('line', np.zeros((rds[rec][0],), dtype='i8'))
                slices = [(i[2], i[0], i[1]) for i in records[rec]]
                idx = rds[rec][1]
                for sub, i, j in slices:
                    if line[i:j].strip():
                        sepdict[rec][sub][idx] = line[i:j]
                sepdict[rec]['line'][idx] = i
                rds[rec][1] += 1
        sepdict.setdefault('TEXT', [])
        sepdict['TEXT'].append(line)
    for rec in sepdict:
        if rec == 'TEXT':
            continue
        sepdict[rec] = pd.DataFrame.from_dict(sepdict[rec])
    return sepdict


def write_pdb(pdbuni, path):
    '''
    Writes a PDB file if it was parsed with :method:`atomic.read_csv`

    Args
        pdbuni (:class:`atomic.container.Universe`): atomic universe from PDB
        path (str): file path to be written to

    Return
        None - writes to file
    '''
    with open(path, 'w') as f:
        for line in pdbuni['metadata']['text']:
            f.write(line + '\n')
