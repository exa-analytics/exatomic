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
for k, v in vars(isotopes).items():
    if isinstance(v, isotopes.Element):
        sym2mass[k] = v.mass
        sym2radius[k] = v.radius
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
