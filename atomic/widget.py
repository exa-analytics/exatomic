# -*- coding: utf-8 -*-
'''
Universe Notebook Widget
=============================
'''
from traitlets import Unicode
from exa.widget import ContainerWidget


class UniverseWidget(ContainerWidget):
    '''
    Custom widget for the :class:`~atomic.universe.Universe` data container.
    '''
    _view_module = Unicode('nbextensions/exa/atomic/universe').tag(sync=True)
    _view_name = Unicode('UniverseView').tag(sync=True)
