# -*- coding: utf-8 -*-
'''
Input Parseing and Composing for cp.x
=======================================
'''
#from io import StringIO
#from exa import _pd as pd
#from exa import _np as np
#from atomic import Length
#from atomic.atom import Atom
#
#
#cp_blocks = ['control', 'system', 'electrons', 'ions', 'cell', 'atomic_species', 'atomic_positions', 'k_points', 'cell_parameters', 'occupations', 'constraints', 'atomic_forces']
#py_to_qe_types = {False: '.false.', True: '.true.'}
#qe_to_py_types = {v: k for k, v in py_to_qe_types.items()}
#qe_unit_names = {'angstrom': 'A', 'bohr': 'au', 'alat': None, 'crystal': None, 'crystal_sg': None}
#
#
#class CPInput:
#    '''
#    '''
#
#    @classmethod
#    def from_file(cls, path):
#        '''
#        '''
#        return cls(**read_input(path))
#
#    def __init__(self, control={}, system={}, electrons={}, ions={},
#                 cell={}, atomic_species=[], k_points=[], cell_paramemters=[],
#                 occupations=[], atomic_forces=[]):
#        for k, v in kwargs.items():
#            setattr(self, k, v)
#
#
#def atom_df(line_list, unit):
#    '''
#    '''
#    unit = 'au' if unit is None else unit
#    inp = StringIO(''.join(line_list))
#    df = pd.read_csv(inp, delim_whitespace=True, header=None, names=['symbol', 'x', 'y', 'z'])
#    df[['x', 'y', 'z']] = df[['x', 'y', 'z']].astype(float) * Length[unit, 'au']
#    df['frame'] = 0
#    df.index.names = ['atom']
#    #df.set_index('frame', append=True, inplace=True)
#    #df = df.reorder_levels(['frame', 'atom'])
#    return Atom(df)
#
#
#def read_input(path):
#    '''
#    Parser for "cp.x" input files.
#    '''
#    lines = None
#    indexes = {}
#    current_block = None
#    kwargs = {'control': {}, 'system': {}, 'electrons': {}, 'ions': {},
#              'cell': {}, 'atomic_species': [], 'atomic_positions': [],
#              'k_points': [], 'cell_parameters': [], 'occupations': [],
#              'constraints': [], 'atomic_forces': []}
#    with open(path) as f:
#        lines = f.readlines()
#    for line in lines:
#        ll = line.lower()
#        ls = ll.split()
#        name = ls[0].replace('&', '')
#        if name in cp_blocks and '=' not in line:
#            current_block = name
#            if len(ls) > 1:
#                kwargs[current_block + '_unit'] = ls[1]
#        if '=' in ll:
#            k, v = ll.split('=')
#            k = k.strip()
#            v = v.strip()
#            if v in qe_to_py_types.keys():
#                v = qe_to_py_types[v]
#                kwargs[current_block][k] = v
#            elif v[0].isdigit():
#                try:
#                    v = int(v)
#                except:
#                    try:
#                        v = float(v)
#                    except:
#                        pass
#                kwargs[current_block][k] = v
#            elif k != '' and v != '':
#                kwargs[current_block][k] = v
#        elif '&' not in ll and '/' not in ll and current_block not in ll:
#            kwargs[current_block].append(line)
#    unit = 'au'
#    if 'atomic_positions_unit' in kwargs.keys():
#        arg = kwargs['atomic_positions_unit'].replace('(', '').replace(')', '')
#        unit = qe_unit_names[arg]
#    kwargs['atomic_positions'] = atom_df(kwargs['atomic_positions'], unit)
#    kwargs['atomic_species'] = [line.strip().split() for line in kwargs['atomic_species']]
#    return kwargs
#
