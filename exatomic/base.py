# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Base Functionality
############################
"""
import os
from exa.util import isotopes
from platform import system
from IPython.display import display_html
from pandas import DataFrame, concat

# For numba compiled functions
sysname= system().lower()
#nbpll = "linux" in sysname
nbpll = False
nbtgt = "parallel" if nbpll else "cpu"
nbche = not nbtgt

isotopedf = isotopes.as_df().dropna(how='any', axis=0)
sym2z = isotopedf.drop_duplicates("symbol").set_index("symbol")["Z"].to_dict()
z2sym = {v: k for k, v in sym2z.items()}
sym2mass = {}
sym2radius = {}
sym2color = {}
for k, v in vars(isotopes).items():
    if isinstance(v, isotopes.Element):
        sym2mass[k] = v.mass
        sym2radius[k] = [v.cov_radius, v.van_radius]
        sym2color[k] = '#' + v.color[-2:] + v.color[3:5] + v.color[1:3]

def staticdir():
    """Return the location of the static data directory."""
    root = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(root, "static")


def resource(name):
    """
    Return the full path of a named resource in the static directory.

    If multiple files with the same name exist, **name** should contain
    the first directory as well.

    .. code-block:: python

        resource("myfile")
        resource("test01/test.txt")
        resource("test02/test.txt")
    """
    for path, _, files in os.walk(staticdir()):
        if name in files:
            return os.path.abspath(os.path.join(path, name))


def list_resources():
    """
    Helper to return a list of all available resources (test input/output/etc)
    files.

    .. code-block:: python

        list_resources()    # Displays a list of available input/output/etc files
        path = resource("qe.inp")    # Get full path of an example file

    Returns:
        resources (list): List of file names
    """
    files = []
    for path, _, files_ in os.walk(staticdir()):
        files.extend(files_)
    return files


def display_side_by_side(*args):
    """Simple function to display 2 dataframes side by side in a notebook."""
    html_str = ''.join([df.to_html() for df in args])
    display_html(html_str.replace('table','table style=\"display:inline\"'),
                 raw=True)

def sym2isomass(symbol, isotope=None):
    """
    Function to get a mapper dictionary to get isotopic masses rather than
    isotopically weigthed atomic masses.

    .. code-block:: python

        >>> sym2isomass('Ni', None)
        {'Ni': 57.9353429}
        >>> sym2isomass(['Ni', 'H', 'C'], None)
        {'C': 12.0, 'H': 1.0078250321, 'Ni': 57.9353429}
        >>> sym2isomass(['Ni', 'H', 'C'], [64, 2, 13])
        {'Ni': 63.927966, 'H': 2.0141017778, 'C': 13.0033548378}


    Args:
        symbol (list or iterable): Elements of interest
        isotope (list or iterable): Isotopes of interest

    Returns:
        masses (dict): Dictionary that can be used inplace of sym2mass

    Raises:
        NotImplementedError: Do not currently support multiple isotopes
                             with same element label
        KeyError: When the given element does not have the isotope index
        TypeError: When either the symbol or isotope values have changed
                   from the expected types of `str` or `int`, respectively
    """
    # take care of right type but not iterable object
    if isinstance(symbol, str): symbol = [symbol]
    # take care of duplicates
    # TODO: if isotopes is passed we need a way to make sure that
    #       we determine if the isotopes are the same and differentiate
    #       between them
    if isotope is not None:
        if isinstance(isotope, int) or isinstance(isotope, float):
            isotope = [int(isotope)]
    if isotope is None:
        symbol = sorted(symbol)
        # TODO: maybe condense to one line
        filtered = isomass.groupby('symbol').filter(lambda x: x['symbol'].unique() in symbol)
        isotope = filtered.groupby('symbol').apply(lambda x:
                                                        x.loc[x['abundance'].idxmax(), 'isotope']).values
    # make a dataframe to handle data better
    # we use the upper case of symbols to avoid possible duplicates different
    # by the case
    df = DataFrame.from_dict({'symbol': list(map(lambda x: x.upper(), symbol)), 'isotope': isotope})
    df.reset_index(drop=True, inplace=True)
    # check that duplicates have the same isotope passed
    for sym, dat in df.groupby('symbol'):
        if not all(list(map(lambda x: abs(x - dat.iloc[0]['isotope']) < 1e-6, dat['isotope']))):
            raise NotImplementedError("We do not currently support getting multiple isotopic " \
                                      + "masses for the same element.")
    # drop the duplicates
    df = df.loc[df['symbol'].drop_duplicates().index]
    # change the case of the symbols to the chemical names
    tmp = []
    for symb in df['symbol']:
        if len(symb) > 1: tmp.append(symb[0]+symb[1:].lower())
        else: tmp.append(symb)
    df['symbol'] = tmp
    # we will return a mapping dictionary
    masses = {}
    for sym, iso in zip(df['symbol'], df['isotope']):
        # just checking the types
        if not isinstance(sym, str):
            raise TypeError("Symbols were have changed type somehow currently, " \
                            +"{}, expected, {}".format(type(sym), 'str'))
        if not isinstance(iso, int):
            raise TypeError("Symbols were have changed type somehow currently, " \
                            +"{}, expected, {}".format(type(iso), "'int'"))
        try:
            mass = isomass.groupby(['symbol', 'isotope']).get_group((sym, iso))['mass'].values[0]
            masses[sym] = mass
        except KeyError:
            raise KeyError("An invalid symbol or isotope was given " \
                           "currently {}, {}".format(sym, iso))
    return masses

