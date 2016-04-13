# -*- coding: utf-8 -*-
'''
Universe Notebook Widget
=============================
'''
import pandas as pd
from traitlets import Unicode
from exa.widget import ContainerWidget


class UniverseWidget(ContainerWidget):
    '''
    Custom widget for the :class:`~atomic.universe.Universe` data container.
    '''
    _view_module = Unicode('nbextensions/exa/atomic/universe').tag(sync=True)
    _view_name = Unicode('UniverseView').tag(sync=True)

    def _handle_field(self, data):
        values = pd.read_json(data.pop('values'), typ='series')
        values.sort_index(inplace=True)
        field = pd.DataFrame.from_dict({key: [val] for key, val in data.items()})
        field['xj'] = field['xk'] = field['yi'] = field['yk'] = field['zi'] = field['zj'] = field['frame'] = 0
        self.container.add_field(field, None, [values])
