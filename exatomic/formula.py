# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Simple Formula
##################
"""
import numpy as np
import pandas as pd
from .core.error import StringFormulaError
from exatomic.base import isotopes, sym2mass


class SimpleFormula(pd.Series):
    """
    A simple way of storing a chemical formula that contains no structural
    information. Element symbols are in alphabetical order (e.g. 'B', 'C', 'Cl', 'Uuo')

        >>> water = SimpleFormula('H(2)O(1)')
        >>> naoh = SimpleFormula('Na(1)O(1)H(1)')
        >>> naoh
        SimpleFormula('H(1)Na(1)O(1)')
    """
    @property
    def mass(self):
        """
        Returns:
            mass (float): Mass (in atomic units) of the associated formula
        """
        df = self.to_frame()
        df['mass'] = df.index.map(sym2mass)
        return (df['mass'] * df['count']).sum()

    def as_string(self):
        """
        Returns:
            formula (str): String representation of the chemical formula.
        """
        return ''.join(('{0}({1})'.format(key.title(), self[key]) for key in sorted(self.index)))

    def __init__(self, data):
        if isinstance(data, str):
            data = string_to_dict(data)
        super().__init__(data=data, dtype=np.int64, name='count')
        self.index.names = ['symbol']

    def __repr__(self):
        return "{}('{}')".format(self.__class__.__name__, self.as_string())

    def __str__(self):
        return self.__repr__()


def string_to_dict(formula):
    """
    Convert string formula to a dictionary.

    Args:
        formula (str): String formula representation

    Returns:
        fdict (dict): Dictionary formula representation
    """
    obj = []
    if ')' not in formula and len(formula) <= 3 and all((not char.isdigit() for char in formula)):
        return {formula: 1}
    elif ')' not in formula:
        raise StringFormulaError(formula)
    for s in formula.split(')'):
        if s != '':
            symbol, count = s.split('(')
            obj.append((symbol, np.int64(count)))
    return dict(obj)


def dict_to_string(formula):
    """
    Convert a dictionary formula to a string.

    Args:
        formula (dict): Dictionary formula representation

    Returns:
        fstr (str): String formula representation
    """
    return ''.join(('{0}({1})'.format(key.title(), formula[key]) for key in sorted(formula.keys()) if formula[key] > 0))
