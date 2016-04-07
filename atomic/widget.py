# -*- coding: utf-8 -*-
'''
Widget (Universe)
======================
'''
from traitlets import Integer, Unicode
from exa.widget import ContainerWidget


class UniverseWidget(ContainerWidget):
    '''
    Custom widget for the :class:`~atomic.universe.Universe` data container.
    '''
    _view_module = Unicode('nbextensions/exa/atomic/universe').tag(sync=True)
    _view_name = Unicode('UniverseView').tag(sync=True)
    gui_width = Integer(250).tag(sync=True)
