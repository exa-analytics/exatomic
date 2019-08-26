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
nbpll = "linux" in sysname
nbtgt = "parallel" if nbpll else "cpu"
nbche = not nbtgt

isotopedf = isotopes.as_df()
sym2z = isotopedf.drop_duplicates("symbol").set_index("symbol")["Z"].to_dict()
z2sym = {v: k for k, v in sym2z.items()}
sym2mass = {}
sym2radius = {}
sym2color = {}
dfs = []
for k, v in vars(isotopes).items():
    if isinstance(v, isotopes.Element):
        sym2mass[k] = v.mass
        sym2radius[k] = [v.cov_radius, v.van_radius]
        sym2color[k] = '#' + v.color[-2:] + v.color[3:5] + v.color[1:3]
        abds = list(map(lambda x: x.af, v.isotopes))
        masses = list(map(lambda x: x.mass, v.isotopes))
        isos = list(map(lambda x: x.A, v.isotopes))
        z = list(map(lambda x: x.Z, v.isotopes))
        symbol = list(map(lambda x: x.symbol, v.isotopes))
        df = DataFrame.from_dict({'abundance': abds, 'mass': masses, 'isotope': isos, 'Z': z,
                                  'symbol': symbol})
        dfs.append(df)
isomass = concat(dfs)
isomass.dropna(how='any', inplace=True)
isomass.reset_index(drop=True, inplace=True)


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
    Function to get a mapper dictionary based on isotopes

    Args:
        symbol (list or iterable): Elements of interest
        isotope (list or iterable): Isotopes of interest

    Returns:
        masses (dict): Dictionary that can be used inplace of sym2mass
    """
    # take care of right type but not iterable object
    if isinstance(symbol, str): symbol = [symbol]
    # take care of duplicates
    # TODO: if isotopes is passed we need a way to make sure that
    #       we determine if the isotopes are the same and differentiate
    #       between them
    symbol = list(set(symbol))
    if isotope is not None:
        if isinstance(isotope, int) or isinstance(isotope, float):
            isotope = [int(isotope)]
        else:
            isotope = list(set(symbol)) # since we do it for the symbols
    # determine the most abundant isotopes
    if isotope is None:
        # needs to be sorted because of the groupby method
        symbol = sorted(symbol)
        # TODO: maybe condense to one line
        filtered = isomass.groupby('symbol').filter(lambda x: x['symbol'].unique() in symbol)
        isotope = filtered.groupby('symbol').apply(lambda x:
                                                        x.loc[x['abundance'].idxmax(), 'isotope'])
    # this is old but may be a good idea to consider
    #elif len(symbol) != len(isotope):
    #    raise AttributeError("Length mismatch between symbol input " \
    #                         + "{} and isotope input {}".format(len(symbol), len(isotope)))
    # we will return a mapping dictionary
    masses = {}
    for i, (sym, iso) in enumerate(zip(symbol, isotope)):
        # TODO: make sure that we do not change the isotope and symbols relationship when
        #       sorting/eliminating duplicates
        try:
            mass = isomass.groupby(['symbol', 'isotope']).get_group((sym, iso))['mass'].values[0]
            masses[sym] = mass
        except KeyError:
            raise KeyError("An invalid symbol or isotope was given " \
                           "currently {}, {}".format(sym, iso))
    return masses
