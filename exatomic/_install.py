# -*- coding: utf-8 -*-
'''
Installer
====================
This module updates the exa installation (regardless of whether it is a
persistent or dynamic installation) to support exatomic's relational tables.
'''
from exa.relational.base import create_tables
from exa._install import install_notebook_widgets
from exa._install import install as exa_install
from exatomic import global_config


def install(persist=False, verbose=False):
    '''
    Persistent installation.
    '''
    if persist:
        exa_install(True)
    update(verbose)


def update(verbose=False):
    '''
    Update the exa installation (persistent or dynamic) to include atomic
    tables and notebook extensions.
    '''
    create_tables()
    install_notebook_widgets(global_config['exatomic_nbext_localdir'],
                             global_config['exatomic_nbext_sysdir'], verbose)
