# -*- coding: utf-8 -*-
'''
Configuration for exatomic
====================================
This module adds two attributes to exa's global configuration (regardless of
whether exa is persistent or dynamic).

Attributes
'''
import os
from exa import global_config
from exa.utility import mkp


pkg = os.path.dirname(__file__)


def update_config():
    '''
    Update the exa's global config.
    '''
    global global_config
    global_config['exatomic_nbext_localdir'] = mkp(pkg, 'nbextensions')
    global_config['exatomic_nbext_sysdir'] = mkp(global_config['nbext_sysdir'], 'exatomic')


update_config()
