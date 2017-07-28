# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
QE cp.x Molecular Dynamics
###############################
Functionality related to parsing inputs and outputs generated when running QE's
cp.x module.
"""
import os
from xml.etree import ElementTree
import numpy as np
import pandas as pd
from linecache import getline
from operator import itemgetter
from exa.utility import mkp
from exa.math.misc.indexing import starts_count
from exa.relational.unit import Time
from exatomic.atom import Atom
from exatomic.frame import Frame
from exatomic.container import Universe
from exatomic.qe.cp.error import CPException


# Default columns
evp_cols = ('step', 'electronic_kinetic_energy', 'cell_temperature', 'atom_temperature',
            'energy', 'enthalpy', 'conserved_energy', 'constant_of_motion', 'cell_volume',
            'cell_pressure', 'time')
nos_cols = ('step', 'nose_e1', 'nose_e2', 'nose_n1', 'nose_n2', 'time')
eig_cols = range(10)


def parse_dynamics_dir(path):
    """
    Parse a molecular dynamics simulation given the "outdir".

    Args:
        path (str): Location of QE's output directory ("outdir")

    Returns:
        universe (:class:`~atomic.universe.Universe`): An atomic universe
    """
    save_dirs = get_save_dirs(path)
    if len(save_dirs) == 0:
        raise CPException('No .save directory in path {}.'.format(path))
    prefix = save_dirs[0].split('_')[0]
    data = parse_datafile(mkp(path, save_dirs[0]))
    symbols = data['symbols']
    nat = len(symbols)
    atom = create_atom(path, prefix, symbols)
    frame = create_frame(path, prefix, nat)
    meta = {'path': path, 'prefix': prefix, 'program': 'qe'}
    return Universe(name=prefix, description=path, atom=atom, frame=frame,
                    meta=meta)


def get_save_dirs(path):
    """
    Find the save directories in a given path.

    Args:
        path (str): Output directory path

    Returns:
        dirs (list): List of save directories
    """
    return [p for p in os.listdir(path) if os.path.isdir(mkp(path, p)) and 'save' in p]


def parse_datafile(save_dir):
    """
    Parses the standand data-file.xml output file from cp.x calculations.

    Args:
        save_dir (str): Save directory path

    Returns:
        d (dict): Dictionary of parsed data key, value pairs
    """
    od = None
    xmlpath = mkp(save_dir, "data-file.xml")
    ions = ElementTree(xmlpath).getroot().getchildren()[4]
    symbols = [atom.get("SPECIES") for atom in ions.getchildren() if 'ATOM.' in str(atom)]
    order = dict(enumerate(set(symbols)))
    #ntype = len(set(symbols))
    #with open(mkp(save_dir, 'data-file.xml'), 'rb') as f:
    #    od = xmltodict.parse(f)
    #symbols = [od['Root']['IONS'][sym]['@SPECIES'].strip() for sym in od['Root']['IONS'].keys() if 'ATOM.' in sym ]
    #ntype = int(od['Root']['IONS']['NUMBER_OF_SPECIES']['#text'])
    #order = {}
    #for i, j in enumerate(range(1, ntype + 1)):
    #    order[od['Root']['IONS']['SPECIE.' + str(j)]['ATOM_TYPE']['#text']] = i
    #symbols = [(sym, order[sym]) for sym in symbols]
    #symbols = [sym[0] for sym in sorted(symbols, key=itemgetter(1))]
    return {'symbols': symbols, 'order': order};


def parse_ijk(path, columns, include_frame=False):
    """
    Generic parser for pos, for, and vel files.

    Args:
        path (str): File path
        columns (list): Header
        include_frame (bool): Include the frame index (default: false)

    Returns:
        df (:pandas:`~pandas.DataFrame`): Atom-like dataframe
    """
    fcol = columns[0]
    lcol = columns[-1]
    df = pd.read_csv(path, delim_whitespace=True, names=columns)
    step = df[df[lcol].isnull()]
    starts = step.drop_duplicates(fcol, keep='last').index.values + 1
    nframes = len(step)
    nat = np.min(starts[1:] - starts[:-1]) - 1
    frame, label, indices = starts_count(starts, nat)
    frame = pd.Series(frame, dtype='category')
    df = df[df.index.isin(indices)].reset_index(drop=True)
    if include_frame:
        df['frame'] = frame
    return df


def parse_pos(*args, **kwargs):
    """
    Parse the positions output file into a dataframe
    """
    return parse_ijk(*args, columns=('x', 'y', 'z'), **kwargs)

def parse_vel(*args, **kwargs):
    """
    Parse the positions output file into a dataframe
    """
    return parse_ijk(*args, columns=('vx', 'vy', 'vz'), **kwargs)


def parse_evp(path, columns=evp_cols):
    """
    Parse in electronic, volume, and pressure data.

    Args:
        columns (array-list): Array of string column names

    Returns:
        evpdf (:class:`~pandas.DataFrame`): Pandas DataFrame containing evp file data
    """
    header = getline(path, 1)
    skiprows = []
    if '#' in header:
        columns = header[1:].split()
        skiprows = [0]
    df = pd.read_csv(path, delim_whitespace=True, names=columns, skiprows=skiprows)
    df.drop_duplicates(df.columns[0], keep='last', inplace=True)
    df.dropna(how='all', axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def parse_nos(path, columns=nos_cols):
    """
    Read in thermostat information.
    """
    df = pd.read_csv(path, delim_whitespace=True, names=columns)
    df.drop_duplicates(df.columns[0], inplace=True, keep='last')
    df.reset_index(drop=True, inplace=True)
    return df


def parse_cel(path):
    """
    Parse cell dimensions.
    """
    df = pd.read_csv(path, delim_whitespace=True, names=('i', 'j', 'k'))
    starts = df[::4].drop_duplicates(df.columns[0], keep='last').index.values + 1
    frame, label, indices = starts_count(starts, 3)
    df = df[df.index.isin(indices)].reset_index(drop=True)
    df['frame'] = pd.Series(frame, dtype='category')
    df['label'] = pd.Series(label, dtype='category')
    df = df.pivot('frame', 'label')
    df.columns = ['xi', 'xj', 'xk', 'yi', 'yj', 'yk', 'zi', 'zj', 'zk']
    df['ox'] = 0.0
    df['oy'] = 0.0
    df['oz'] = 0.0
    df.index.names = [None]
    return df


def create_atom(path, prefix, symbols):
    """
    Create a :class:`~atomic.atom.Atom`.
    """
    df1 = parse_pos(mkp(path, prefix + '.pos'), include_frame=True)
    df2 = parse_vel(mkp(path, prefix + '.vel'), include_frame=False)
    n = len(df1) // len(symbols)
    df1['symbol'] = pd.Series(symbols * n, dtype='category')
    df1['vx'] = df2['vx']
    df1['vy'] = df2['vy']
    df1['vz'] = df2['vz']
    df1.dropna(axis=0, how='any', inplace=True)
    return Atom(df1)


def create_frame(path, prefix, nat, evp_cols=evp_cols, nos_cols=nos_cols):
    """
    Create a :class:`~atomic.frame.Frame` from dynamics data.

    Args:
        path (str): Directory path
        prefix (str): Output file prefix
        nat (int): Number of atoms per frame
        evp_cols (list): Header for the evp file
        nos_cols (list): Header for the nos file
    """
    df = parse_evp(mkp(path, prefix + '.evp'))
    df1 = parse_cel(mkp(path, prefix + '.cel'))
    for col in df1.columns:
        df[col] = df1[col]
    df1 = parse_nos(mkp(path, prefix + '.nos'))
    for col in df1.columns:
        df[col] = df1[col]
    df['atom_count'] = nat
    df['periodic'] = True
    df.dropna(axis=0, how='any', inplace=True)
    factor = Time['ps', 'au']
    df['time'] *= factor
    return Frame(df)
