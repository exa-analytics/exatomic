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
    reccnt = _pre_process_pdb(flins)

def _pre_process_pdb(flins):
    recdims = {}
    for record in records:
        recdims[key] = sum([1 for rec in recs if record == rec])



def _parse_pdb(flins):

    pass



def write_pdb(fp):
    raise NotImplementedError(
        "Writing PDB files is not supported."
    )
