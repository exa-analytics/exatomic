## -*- coding: utf-8 -*-
## Copyright (c) 2015-2020, Exa Analytics Development Team
## Distributed under the terms of the Apache License 2.0
#"""
#PDB File I/O
#=======================
#"""
#import pandas as pd
#from json import load
#from io import StringIO
#from requests import get as _get
#
#from exa._config import config
#from exatomic import Editor, Length
#
#base = config['dynamic']['exatomic_pkgdir']
#pdbr = '{}/{}/{}'.format(base, '_static', 'pdb-min.json')
#with open(pdbr) as f: records = load(f)
#
#class PDB(Editor):
#
#    def _pandas_fwf(self, start=None, stop=None, linenos=None, **kwargs):
#        if linenos is not None:
#            piece = StringIO('\n'.join([self[lno] for lno in linenos]))
#        elif start is not None and stop is not None:
#            piece = StringIO('\n'.join(self[start:stop]))
#        else:
#            raise Exception('Must pass start and stop or linenos')
#        return pd.read_fwf(piece, **kwargs)
#
#    def parse_atom(self):
#        print('No deduplication of atoms occurs at the moment.')
#        found = self.regex('^ATOM', keys_only=True)
#        widths = [(i[0], i[1]) for i in records['ATOM']]
#        names = [i[2] for i in records['ATOM']]
#        df = self._pandas_fwf(linenos=found, colspecs=widths, names=names)
#        df['x'] *= Length['A', 'au']
#        df['y'] *= Length['A', 'au']
#        df['z'] *= Length['A', 'au']
#        df['frame'] = 0
#        self.atom = df
#
#
##from warnings import warn as _warn
##
##from exa.utils import mkpath
##from exatomic import _os as os
##from exatomic import _pd as pd
##from exatomic import _np as np
##from exatomic import _sys as sys
##from exatomic.algorithms.nonjitted import generate_minimal_framedf_from_onedf as _gen_fdf
##try:
##    from exatomic.algorithms.jitted import expand as _expand
##except ImportError:
##    from exatomic.algorithms.nonjitted import expand as _expand
##
##
##_selfpath = os.path.abspath(__file__).replace('pdb.py', '')
##_recpath = mkpath(_selfpath, 'static', 'pdb-min.json')
##
##
##records = None
##with open(_recpath, 'r') as f:
##    records = load(f)
##
##
##def read_pdb(path, metadata={}, **kwargs):
##    """
##    Reads a PDB file
##
##    Args
##        path (str): file path (local) or PDB identifier (remote)
##        metadata (dict): metadata as key, value pairs
##        **kwargs: only if using with exa content management system
##
##    Return
##        unikws (dict): dataframes containing 'frame', 'one' body and 'meta' data
##
##    See Also
##        :class:`exatomic.universe.Universe`
##    """
##    flins = _path_handler(path)
##    rds = _pre_process_pdb(flins)
##    frdx, odx, ptls = _expand(
##        np.arange(0, rds['nrf']),
##        np.array([rds['nat'] for i in range(rds['nrf'])])
##    )
##    del ptls
##    sepdict = _parse_pdb(flins, rds)
##    one = _restruct_one(sepdict['ATOM'], frdx, odx)
##    frame = _gen_fdf(one)
##    unikws = {'frame': frame, 'one': one, 'metadata': {'text': sepdict['TEXT']}}
##    unikws['metadata'].update(metadata)
##    return unikws
##
##
##def _path_handler(path):
##    if os.sep in path:
##        return _local_path(path)
##    else:
##        return _remote_path(path)
##
##
##def _local_path(path):
##    with open(path, 'r') as f:
##        flins = f.read().splitlines()
##    return flins
##
##
##def _remote_path(path):
##    url = 'http://www.rcsb.org/pdb/files/{}.pdb'.format(path.lower())
##    flins = _get(url).text.splitlines()
##    if 'FileNotFoundException' in flins[0]:
##        raise FileNotFoundError('{}:{}'.format(url, ' may not exist.'))
##    return flins
##
##
##def _pre_process_pdb(flins):
##    rds = {}
##    recs = [line[:6].strip() for line in flins]
##    for record in records:
##        rds[record] = [sum([1 for rec in recs if record == rec]), 0]
##    if rds['ENDMDL'][0]:
##        nrf = rds['ENDMDL'][0]
##    else:
##        nrf = 1
##    nat = rds['ATOM'][0] // nrf
##    rds['nrf'] = nrf
##    rds['nat'] = nat
##    return rds
##
##
##def _restruct_one(atomdf, frdx, odx):
##    one = atomdf.loc[:, ['symbol', 'x', 'y', 'z', 'res.seq.']]
##    one['frame'] = frdx
##    one['one'] = odx
##    one.set_index(['frame', 'one'], inplace=True)
##    return one
##
##
##def _parse_pdb(flins, rds):
##    sepdict = {}
##    for i, line in enumerate(flins):
##        rec = line[:6].strip()
##        if rec == 'ATOM' or rec == 'ANISOU' and records[rec]:
##            if records[rec]:
##                sepdict.setdefault(
##                    rec, {i[2]: np.zeros(
##                        (rds[rec][0], ), dtype=i[3]
##                    ) for i in records[rec]}
##                )
##                sepdict[rec].setdefault('line', np.zeros((rds[rec][0],), dtype='i8'))
##                slices = [(i[2], i[0], i[1]) for i in records[rec]]
##                idx = rds[rec][1]
##                for sub, i, j in slices:
##                    if line[i:j].strip():
##                        sepdict[rec][sub][idx] = line[i:j]
##                sepdict[rec]['line'][idx] = i
##                rds[rec][1] += 1
##        sepdict.setdefault('TEXT', [])
##        sepdict['TEXT'].append(line)
##    for rec in sepdict:
##        if rec == 'TEXT':
##            continue
##        sepdict[rec] = pd.DataFrame.from_dict(sepdict[rec])
##    return sepdict
##
##
##def write_pdb(pdbuni, path):
##    """
##    Writes a PDB file if it was parsed with :method:`exatomic.read_csv`
##
##    Args
##        pdbuni (:class:`exatomic.universe.Universe`): atomic universe from PDB
##        path (str): file path to be written to
##
##    Return
##        None - writes to file
##    """
##    with open(path, 'w') as f:
##        for line in pdbuni['metadata']['text']:
##            f.write(line + '\n')
##
