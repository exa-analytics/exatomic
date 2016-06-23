# -*- coding: utf-8 -*-
'''
Installer
====================
'''
from exa.relational.base import _create_all
from exa._install import _install_notebook_widgets
from exatomic import _conf


def install(persistent=False, verbose=False):
    '''
    '''
    if persistent:
        raise NotImplementedError('Persistent state exatomic not yet working...')
    else:
        _create_all()
        _install_notebook_widgets(_conf['exatomic_nbext_localdir'],
                                  _conf['exatomic_nbext_sysdir'], verbose=verbose)
