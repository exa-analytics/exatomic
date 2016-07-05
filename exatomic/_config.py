# -*- coding: utf-8 -*-
'''
Configuration for exatomic
====================================
This module adds two attributes to exa's global configuration (regardless of
whether exa is persistent or dynamic).

Attributes
'''
import os
from exa._config import config
from exa.utility import mkp


pkg = os.path.dirname(__file__)


def update_config():
    '''
    Update the "global" configuration.
    '''
    global config
    config['exatomic_nbext_localdir'] = mkp(pkg, '_nbextensions')
    config['exatomic_nbext_sysdir'] = mkp(config['nbext_sysdir'], 'exatomic', mk=True)
