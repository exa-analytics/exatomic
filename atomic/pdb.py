# -*- coding: utf-8 -*-

from atomic import _os as os
from atomic import _pd as pd
from atomic import _np as np
from atomic import _sys as sys
import requests as _req
import json as _json
from io import StringIO
from warnings import warn as _warn

#Hacky imports
from atomic.algorithms.nonjitted import generate_minimal_framedf_from_onedf as _gen_fdf
sys.path.insert(0, '/home/tjd/Programs/analytics-exa/exa')
from exa.utils import mkpath

_selfpath = os.path.abspath(__file__).replace('pdb.py', '')
_recpath = mkpath(_selfpath, 'static', 'pdb-small.json')

with open(_recpath, 'r') as f:
    records = _json.load(f)

def read_pdb(path):
    '''
    Reads a PDB file

    Args
        path (str): file path (local) or PDB identifier (remote)

    Return
        unikws (dict): dataframes containing 'frame' and 'one' body data

    See Also
        :class:`atomic.container.Universe`
    '''
    if os.sep in path:
        with open(path, 'r') as f:
            flins = f.read().splitlines()
    else:
        try:
            url = 'http://www.rcsb.org/pdb/files/{}.pdb'.format(path.lower())
            flins = _req.get(url).text.splitlines()
        except:
            ConnectionError('{}: check that url exists.'.format(url))
    rds = _pre_process_pdb(flins)
    frdx = [i for i in range(rds['nrf']) for j in range(rds['nat'])]
    odx = [i for j in range(rds['nrf']) for i in range(rds['nat'])]
    sepdict = _parse_pdb(flins, rds)
    one = sepdict['ATOM'].loc[:, ['symbol', 'x', 'y', 'z']]
    one['frame'] = frdx
    one['one'] = odx
    one.set_index(['frame', 'one'], inplace=True)
    frame = _gen_fdf(one)
    return {'frame': frame, 'one': one}

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

def _parse_pdb(flins, rds):
    sepdict = {}
    for i, line in enumerate(flins):
        rec = line[:6].strip()
        #if 'ORIGX' in rec or 'SCALE' in rec \
        #or 'MATRX' in rec or 'DBREF' in rec:
        #    rec = rec[:5]
        if rec == 'ATOM' or rec == 'HETATM' or rec == 'ANISOU':
            if records[rec]['rec']:
                sepdict.setdefault(
                    rec, {i[2]: np.zeros(
                        (rds[rec][0], ), dtype=i[3]
                    ) for i in records[rec]['rec']}
                )
                slices = [(i[2], i[0], i[1]) for i in records[rec]['rec']]
                for sub, i, j in slices:
                    if line[i:j].strip():
                        sepdict[rec][sub][rds[rec][1]] = line[i:j]
                rds[rec][1] += 1
    for rec in sepdict:
        sepdict[rec] = pd.DataFrame.from_dict(sepdict[rec])
    return sepdict


def write_pdb(fp):
    raise NotImplementedError(
        "Writing PDB files is not supported."
    )