# -*- coding: utf-8 -*-
'''
Orbital DataFrame
==========================

'''
from exa import DataFrame
from exa.frames import Updater, DataFrame


class Orbital(DataFrame):
    '''
    '''
    __pk__ = ['atom']
    __fk__ = ['frame']
    __traits__ = ['x', 'y', 'z', 'radius', 'color']
    __groupby__ = 'frame'


class OrbitalMeta(Updater):
