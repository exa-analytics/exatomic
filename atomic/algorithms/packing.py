# -*- coding: utf-8 -*-
'''
Cell Packing Utilities
========================
'''
from exa.relational import Mass, Length


def cubic_cell_dimension(mass, density):
    '''
    Args:
        mass (float): Total mass of all atoms (in atomic units)
        density (float): Desired density (in g/cm^3)

    Returns:
        a (float): Cubic unit cell dimension (in atomic units)
    '''
    mass *= Mass['u', 'g']
    density *= Length['cm', 'au']**3
    return (mass / density)**(1/3)
