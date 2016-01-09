# -*- coding: utf-8 -*-

from atomic import _os as os
from atomic import _pd as pd
from atomic import _np as np
from atomic import _sys as sys
import requests as _req
import json as _json
from warnings import warn as _warn

#Hacky imports
sys.path.insert(0, '/home/tjd/Programs/analytics-exa/exa')
from exa.utils import mkpath

_selfpath = os.path.abspath(__file__).replace('pdb.py', '')
_recpath = mkpath(_selfpath, 'static', 'pdb.json')

with open(_recpath, 'r') as f:
    records = _json.load(f)

def read_pdb(path):
    '''
    Reads a PDB file

    Args
        path (str): file path (local) or PDB code (remote)

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
    rds = _pre_process_pdb(flins)       # Get number of occurances of records
    sepdict, errors = _parse_pdb(flins, rds)    # dictionary separated into records
    return sepdict, errors


def _pre_process_pdb(flins):
    recdims = {}
    reccnt = [line[:6].strip() for line in flins]
    for record in records:
        if 'ORIGX' in record or 'SCALE' in record \
        or 'MATRX' in record or 'DBREF' in record:
            recdims[record[:5]] = [sum([1 for rec in reccnt if record == rec]), 0]
        else:
            recdims[record] = [sum([1 for rec in reccnt if record == rec]), 0]
    print(recdims['ATOM'])
    print(recdims['HETATM'])
    print(recdims['ANISOU'])
    return recdims

def _parse_pdb(flins, rds):
    sepdict = {}
    errors = []
    for key, val in records.items():
        trec = val['rec']
        if trec:
            dtyp=[(i[2], i[3]) for i in trec]
            sepdict[key] = np.empty((rds[key][0], ), dtype=dtyp)
    for i, line in enumerate(flins):
        rec = line[:6].strip()
        if 'ORIGX' in rec or 'SCALE' in rec \
        or 'MATRX' in rec or 'DBREF' in rec:
            rec = rec[:5]
        if records[rec]['rec']:
            slices = [(i[0], i[1]) for i in records[rec]['rec']]
            dtyps = [(i[2], i[3]) for i in records[rec]['rec']]
            if rec in sepdict:
                cidx = rds[rec][1]
                print(rec)
                print(line)
                print(slices)
                print(dtyps)
                #try:
                sepdict[rec][cidx] = tuple([line[i:j] for i, j in slices])
                #except (ValueError, TypeError) as e:
                #    errors.append('{}:{}'.format(rec, e))
                #    pass
                rds[rec][1] += 1
    return sepdict, errors



def write_pdb(fp):
    raise NotImplementedError(
        "Writing PDB files is not supported."
    )
