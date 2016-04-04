# -*- coding: utf-8 -*-
'''
Installer
====================
'''
from exa.utility import _install_notebook_widgets
from exa.relational.base import _create_all
from atomic import _conf


def install(persistent=False):
    '''
    '''
    if persistent:
        raise NotImplementedError('Persistent state atomic not yet working...')
    else:
        _create_all()
        _install_notebook_widgets(_conf['atomic_nbext_localdir'],
                                  _conf['atomic_nbext_sysdir'])
