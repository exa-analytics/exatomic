# -*- coding: utf-8 -*-
'''
Simple Formula
==========================
'''
from exa import _np as np
from exa import _pd as pd
from atomic import Isotope


class SimpleFormula(pd.Series):
    '''
    A simple way of storing a chemical formula that contains no structural
    information.

    Element symbols are in alphabetical order (e.g. 'B', 'C', 'Cl', 'Uuo')

    .. code-block: bash

        'H(2)O(1)'
        'C(6)H(6)'
    '''
    def get_string(self):
        '''
        Returns:
            formula (str): String representation of the chemical formula.
        '''
        return ''.join(('{0}({1})'.format(key.title(), self[key]) for key in sorted(self.index)))

    def get_mass(self):
        '''
        Returns:
            mass (float): Mass (in atomic units) of the associated formula
        '''
        df = self.to_frame()
        df['mass'] = df.index.map(Isotope.symbol_to_mass())
        return (df['mass'] * df['count']).sum()

    @classmethod
    def from_string(cls, formula):
        '''
        '''
        return cls(string_to_dict(formula))

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = np.int64
        super().__init__(*args, **kwargs)
        self.index.names = ['symbol']
        self.name = 'count'


def string_to_dict(formula):
    '''
    Convert string formula to a dictionary.

    Args:
        formula (str): String formula representation

    Returns:
        fdict (dict): Dictionary formula representation
    '''
    obj = []
    for s in formula.split(')'):
        if s != '':
            symbol, count = s.split('(')
            obj.append((symbol, np.int64(count)))
    return dict(obj)


def dict_to_string(formula):
    '''
    Convert a dictionary formula to a string.

    Args:
        formula (dict): Dictionary formula representation

    Returns:
        fstr (str): String formula representation
    '''
    return ''.join(('{0}({1})'.format(key.title(), formula[key]) for key in sorted(formula.keys())))
