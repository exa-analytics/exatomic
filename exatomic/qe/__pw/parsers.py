# -*- coding: utf-8 -*-
#'''
#QE pw.x Parsers
#=======================================
#Parsers for pw.x inputs and outputs
#'''
#from io import StringIO
#from exa import _pd as pd
#from exa import _np as np
#from exa.config import Config
#if Config.numba:
#    from exa.jitted.indexing import idxs_from_starts_and_counts
#else:
#    from exa.algorithms.indexing import idxs_from_starts_and_counts
#from atomic import Length
#from atomic.atom import Atom
#from atomic.frame import _min_frame_from_atom, Frame
#from qe.types import to_py_type, get_length
#from qe.classes import classes, Timings
#from qe.pw.classes import PWInput, PWOutput
#
#
#def parse_pw_input(path):
#    '''
#    '''
#    kwargs = {}
#    current = None
#    with open(path) as f:
#        for line in f:
#            low = line.lower()
#            split = low.split()
#            block = split[0].replace('&', '')
#            if block in classes.keys() and '=' not in line:
#                current = block
#                kwargs[current] = classes[block]()
#                if len(split) > 1:
#                    kwargs[current]._info = ' '.join(split[1:])
#            if '=' in line:
#                k, v  = line.split('=')
#                k = k.strip()
#                v = to_py_type(v)
#                kwargs[current][k] = v
#            elif '&' not in line and '/' not in line and current not in low:
#                kwargs[current] = kwargs[current]._append(line)
#    return PWInput(**kwargs)
#
#
#def parse_pw_output(path):
#    '''
#    Args:
#        path (str): Output file path
#
#    Returns:
#        obj ()
#    '''
#    header = []
#    footer = []
#    scf = []
#    eigs = []
#    frames = []
#    positions = []
#    block = 'header'
#    kwargs = {'header': [], 'footer': [], 'scf': [], 'eigs': [],
#              'frames': [], 'atom': [], 'forces': [], 'meta': []}
#    with open(path) as f:
#        for line in f:
#            if 'Self-consistent Calculation' in line:
#                block = 'scf'
#            elif 'End of self-consistent calculation' in line:
#                block = 'eigs'
#            elif 'highest occupied level' in line or 'Fermi energy' in line:
#                block = 'frames'
#            elif 'Forces acting on atoms' in line:
#                block = 'forces'
#            elif 'ATOMIC_POSITIONS' in line:
#                block = 'atom'
#            elif 'Writing output' in line:
#                block = 'meta'
#            elif 'init_run' in line:
#                block = 'footer'
#            kwargs[block].append(line)
#    timings = _parse_footer(kwargs['footer'])
#    atom = _parse_atom(''.join(kwargs['atom']), ''.join(kwargs['forces']))
#    frame = _parse_frame(kwargs['frames'], atom)
#    out = PWOutput(timings=timings, atom=atom, frame=frame)
#    return out, kwargs
#
#
#def _parse_footer(footer):
#    '''
#    '''
#    data = {'catagory': [], 'called_by': [], 'name': [], 'cpu': [], 'wall': [], 'ncalls': []}
#    called_by = ''
#    catagory = 'summary'
#    for line in footer:
#        if 'Called by' in line:
#            called_by = line.replace('Called by ', '').replace(':\n', '')
#            if called_by != 'init_run':
#                catagory = 'individual'
#        elif 'routines' in line:
#            called_by = line.split()[0].lower()
#            catagory = 'individual'
#        elif 'calls' in line:
#            ls = line.split()
#            name = ls[0]
#            cpu = float(ls[2].replace('s', ''))
#            wall = float(ls[4].replace('s', ''))
#            ncalls = int(ls[7])
#            data['catagory'].append(catagory)
#            data['called_by'].append(called_by)
#            data['name'].append(name)
#            data['cpu'].append(cpu)
#            data['wall'].append(wall)
#            data['ncalls'].append(ncalls)
#    df = Timings.from_dict(data)
#    df.set_index(['catagory', 'name'], inplace=True)
#    df['cpu'] = df['cpu'].map(lambda x: pd.Timedelta(seconds=x.astype(np.float64)))
#    df['wall'] = df['wall'].map(lambda x: pd.Timedelta(seconds=x.astype(np.float64)))
#    df.sort_index(inplace=True)
#    return df
#
#
#def _parse_atom(atoms, forces, label=True):
#    '''
#    '''
#    forces = StringIO(forces)
#    forces = pd.read_csv(forces, delim_whitespace=True, header=None,
#                         names=['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'fx', 'fy', 'fz'])
#    forces = forces.ix[(forces['n0'] == 'atom'), ['fx', 'fy', 'fz']].reset_index(drop=True)
#    atom = StringIO(atoms)
#    atom = pd.read_csv(atom, delim_whitespace=True, header=None,
#                       names=['symbol', 'x', 'y', 'z'])
#    frames = atom[atom['symbol'] == 'ATOMIC_POSITIONS']
#    unit = get_length(frames.iloc[0, 1])
#    starts = frames.index.values + 1
#    counts = (frames.index[1:] - frames.index[:-1]).values - 1
#    counts = np.append(counts, atom.index.values[-1] - frames.index[-1] - 1)
#    atom.dropna(inplace=True)
#    atom.reset_index(inplace=True, drop=True)
#    frame, lbl, indices = idxs_from_starts_and_counts(starts, counts)
#    atom['symbol'] = atom['symbol'].astype('category')
#    atom['frame'] = frame
#    atom['label'] = lbl
#    atom['fx'] = forces['fx']
#    atom['fy'] = forces['fy']
#    atom['fz'] = forces['fz']
#    atom[['x', 'y', 'z', 'fx', 'fy', 'fz']] = atom[['x', 'y', 'z', 'fx', 'fy', 'fz']].astype(np.float64)
#    conv = Length[unit, 'au']
#    atom['x'] *= conv
#    atom['y'] *= conv
#    atom['z'] *= conv
#    atom['fx'] *= conv
#    atom['fy'] *= conv
#    atom['fz'] *= conv
#    atom.index.names = ['atom']
#    return Atom(atom)
#
#
#def _parse_frame(data, atom):
#    '''
#    '''
#    df = _min_frame_from_atom(atom)
#    rows = {'energy': [], 'one_electron_energy': [], 'hartree_energy': [], 'xc_energy': [],
#        'ewald': [], 'smearing': []}
#    for line in data:
#        split = line.split()
#        if 'total energy' in line and '!' in line:
#            rows['energy'].append(split[4])
#        elif 'one-electron' in line:
#            rows['one_electron_energy'].append(split[3])
#        elif 'hartree contribution' in line:
#            rows['hartree_energy'].append(split[3])
#        elif 'xc contribution' in line:
#            rows['xc_energy'].append(split[3])
#        elif 'ewald contribution' in line:
#            rows['ewald'].append(split[3])
#        elif 'smearing contrib' in line:
#            rows['smearing'].append(split[4])
#    frame = pd.DataFrame.from_dict(rows)
#    frame = frame.astype(np.float64)
#    for col in df.columns:
#        frame[col] = df[col]
#    return Frame(frame)
#
