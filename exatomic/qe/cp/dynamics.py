# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
QE cp.x Molecular Dynamics
###############################
Functionality related to parsing inputs and outputs generated when running QE's
cp.x module. Data files that come from dynamics include evp, for, pos, etc.
Since these files are CSV-like files, and typically large, CSV reading functions,
rather than :class:`~exa.core.editor.Editor` objects are used to parse in the
data.
"""
import re, bz2
from six import StringIO
import pandas as pd
import numpy as np
import numba as nb
from exa import Editor
from exa.util.units import Length
from exatomic.base import nbpll


@nb.jit(nopython=True, nogil=True, parallel=nbpll)
def construct_fdx(fdxs, size):
    n = len(fdxs)*size
    frame = np.empty((n, ), dtype=np.int64)
    k = 0
    for fdx in fdxs:
        for _ in range(size):
            frame[k] = fdx
            k += 1
    return frame


def parse_symbols_from_input(path):
    """
    The only way to get the symbols, in the correct order, is
    by parsing them from an input file.
    """
    inp = Editor(path)
    found = inp.regex("atomic_positions", "nat", flags=re.I)
    nat = int(found['nat'][0][1].split("=")[-1])
    start = found['atomic_positions'][0][0]
    length = "Angstrom"
    if "bohr" in inp[start].lower():
        length = "au"
    xyz = pd.read_csv(StringIO("\n".join(inp[start+1:start+1+nat])), names=("symbol", "x", "y", "z"),
                      delim_whitespace=True)
    for q in ("x", "y", "z"):
        xyz[q] = Length[length, 'au']*xyz[q].astype(float)
    return xyz


def parse_evp(path, symbols, columns=None, **kwargs):
    """
    Parse in the EVP file using pandas.

    If the ``columns`` argument is None, this function attempts to
    determine if the column names are present otherwise uses defaults.
    """
    dw = kwargs.pop("delim_whitespace", True)
    skiprows = kwargs.pop("skiprows", [])
    def parser(columns, skiprows):
        df = pd.read_csv(path, names=columns, skiprows=skiprows,
                         delim_whitespace=dw, **kwargs)
        df.drop_duplicates(columns[0], keep="last", inplace=True)
        df.dropna(how="all", axis=1, inplace=True)
        return df

    if path.endswith("bz2"):
        opener = bz2.open
    else:
        opener = open
    if columns is None:
        with opener(path) as f:
            try:
                first = f.readline().decode("utf-8")
            except:
                first = f.readline()
        if first.strip().startswith("#"):
            columns = first.split()[1:]
            skiprows = [0]
        else:
            columns = list(range(13))
            skiprows = []
    return parser(columns, skiprows)


def parse_xyz(path, symbols, columns=("x", "y", "z"), **kwargs):
    """
    Parse XYZ-like files, pos, vel, for, using pandas.

    Warning:
        In certain cases using ``pandas.read_fwf`` may work better.
    """
    # Parse in the data using pandas
    dw = kwargs.pop("delim_whitespace", True)
    df = pd.read_csv(path, delim_whitespace=dw, names=columns, **kwargs)
    # The first line contains the frame number, isolate it
    fdxs = df.loc[df[columns[-1]].isnull(), columns[0]]
    nat = int(fdxs.index[1] - fdxs.index[0] - 1)
    # And remove those lines from the xyz-like data
    df.dropna(how="any", inplace=True)
    # Construct the frame index
    df['frame'] = construct_fdx(fdxs.values.astype(int), nat)
    df['frame'] = df['frame'].astype("category")
    # and label so that we can deduplicate the data
    df['label'] = list(range(len(symbols)))*len(fdxs)
    df['label'] = df['label'].astype("category")
    # Drop duplicated data (typically due to simulation errors)
    df.drop_duplicates(["frame", "label"], keep="last", inplace=True)
    # Cleanup and add symbols
    del df['label']
    df['symbol'] = symbols*len(df['frame'].unique())
    df['symbol'] = df['symbol'].astype("category")
    df.reset_index(drop=True, inplace=True)
    return df
