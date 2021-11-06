# -*- coding: utf-8 -*-
# Copyright (c) 2015-2022, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Periodic Table of Elements and Isotopes
########################################
This module provides a database of the atomic elements and their isotopes.
Visualization parameters are also provided. Data is provided and maintained
by `NIST`_. The full api is given in the code example below. Note that not
all attributes that are present on isotopes are present on elements (and vice
versa).

.. code-block:: python

    from exatomic.exa.util import isotopes
    isotopes.H            # Hydrogen element
    isotopes.H[2]         # Hydrogen isotopes 2 (deuterium)
    isotopes.H.cov_radius # Empirical covalent radius (a.u. - Bohr)
    isotopes.H.van_radius # Empirical Van der Waals radius (a.u. - Bohr)
    isotopes.H.af         # Abundance fraction
    isotopes.H.afu        # Abundance fraction uncertainty
    isotopes.H.mass       # Atomic mass (g/mol)
    isotopes.H.massu      # Atomic mass uncertainty (g/mol)
    isotopes.H.eneg       # Electronegativity (Pauling scale)
    isotopes.H[2].quad    # Nuclear quadrupole moment (eb - electron-barn - e10^(-28)m^2)
    isotopes.H[2].A       # Atomic mass number (g/mol)
    isotopes.H.Z          # Proton number
    isotopes.H[2].g       # Nuclear g-factor (dimensionless magnetic moment)
    isotopes.H[2].spin    # Nuclear spin
    isotopes.H.color      # Traditional atomic color (HTML)
    isotopes.H.name       # Full element name

Warning:
    Isotopes are provided as part of the static data directory.

.. _NIST: https://www.nist.gov/
"""
import os as _os
import sys as _sys
import bz2 as _bz2
from pandas import read_json as _rj
from exatomic.exa.static import resource as _resource
if not hasattr(_bz2, "open"):
    _bz2.open = _bz2.BZ2File


class Element(object):
    """
    An element from Mendeleev's periodic table.

    .. code-block:: python

        from exatomic.exa.util import isotopes
        H = isotopes.H         # Hydrogen element (isotope averaged)
        D = isotopes.H['2']    # Deuterium (2H, a specific isotope)
        isotopes.H.isotopes    # List of available isotopes
    """
    @property
    def isotopes(self):
        return [v for k, v in vars(self).items() if k.startswith("_")]

    @property
    def radius(self):
        return self.cov_radius

    def __init__(self, symbol, name, mass, znum, cov_radius, van_radius, color):
        self.symbol = symbol
        self.name = name
        self.mass = mass
        self.Z = znum
        self.cov_radius = cov_radius
        self.van_radius = van_radius
        self.color = color

    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, "_"+key)
        return getattr(self, key)

    def __repr__(self):
        return self.symbol


class Isotope(object):
    """
    A specific atomic isotope (the physical manifestation of an element).

    .. code-block:: python

        from exatomic.exa.util import isotopes
        isotopes.U['235'].mass    # Mass of 235-U
    """
    @property
    def radius(self):
        return self.cov_radius

    def __init__(self, anum, znum, af, afu, cov_radius, van_radius, g, mass, massu, name, eneg, quad, spin, symbol, color):
        self.A = anum
        self.Z = znum
        self.af = af
        self.afu = afu
        self.cov_radius = cov_radius
        self.van_radius = van_radius
        self.g = g
        self.mass = mass
        self.massu = massu
        self.name = name
        self.eneg = eneg
        self.quad = quad
        self.spin = spin
        self.symbol = symbol
        self.color = color

    @property
    def radius(self):
        return self.cov_radius

    def __repr__(self):
        return str(self.A) + self.symbol


def _create():
    """Globally called function for creating the isotope/element API."""
    def creator(group):
        """Helper function applied to each symbol group of the raw isotope table."""
        symbol = group['symbol'].values[0]
        try:    # Ghosts and custom atoms don't necessarily have an abundance fraction
            mass = (group['mass']*group['af']).sum()
            afm = group['af'].sum()
            if afm > 0.0:
                mass /= afm
        except ZeroDivisionError:
            mass = group['mass'].mean()
        znum = group['Z'].max()
        cov_radius = group['cov_radius'].mean()
        van_radius = group['van_radius'].mean()
        try:
            color = group.loc[group['af'].idxmax(), 'color']
        except (TypeError, KeyError):
            color = group['color'].values[0]
        name = group['name'].values[0]
        ele = Element(symbol, name, mass, znum, cov_radius, van_radius, color)
        # Attached isotopes
        for tope in group.apply(lambda s: Isotope(*s.tolist()), axis=1):
            setattr(ele, "_"+str(tope.A), tope)
        return ele

    iso = _rj(_path)
    iso.columns = _columns
    setattr(_this, "iso", iso)
    for element in iso.groupby("symbol").apply(creator):
        setattr(_this, element.symbol, element)


def as_df():
    """Return a dataframe of isotopes."""
    return _this.iso


# Data order of isotopic (nuclear) properties:
_resource = _resource("isotopes.json")
_columns = ("A", "Z", "af", "afu", "cov_radius", "van_radius", "g", "mass", "massu", "name",
            "eneg", "quad", "spin", "symbol", "color")
_this = _sys.modules[__name__]         # Reference to this module
_path = _os.path.abspath(_os.path.join(_os.path.abspath(__file__), _resource))
if not hasattr(_this, "H"):
    _create()
