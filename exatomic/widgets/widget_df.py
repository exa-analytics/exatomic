# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
A dataframe widget
#########################
A generic pandas dataframe widget.
"""
import pandas as pd
from ipywidgets import Box, Dropdown, IntRangeSlider, SelectMultiple, Button, Output, IntSlider
from traitlets import Integer, List, Tuple, Int
from IPython.display import display_html
from exatomic.widgets.widget_utils import _glo, _flo, _wlo, _hboxlo, _vboxlo, _bboxlo, _ListDict, Folder, GUIBox


class DFBox(Box):
    cur_frame = Integer(-1).tag(sync=True)
    columns = List([]).tag(sync=True)
    indexes = List([]).tag(sync=True)
    start = Int(0).tag(sync=True)

    def _update_output(self):
        self._output.clear_output()
        with self._output:
            df = self._df
            idxs = self.indexes
            if self.cur_frame >= 0:
                df = self._df.groupby('frame').get_group(self.cur_frame)
            #idxs = [max(idxs[0], df.index[0]), min(idxs[1], df.index[-1])]
            display_html(df[self.columns].loc[range(self.start, self.start + 50)].to_html(),
                         raw=True)

    def _init_gui(self):

        close = Button(description=' Close', icon='trash', layout=_wlo)
        def _close(b): self.close()
        close.on_click(_close)

        frame = IntSlider(min=self._df.frame.astype(int).min(),
                          max=self._df.frame.astype(int).max(),
                          value=-1, description='Frame', layout=_wlo)

        cbut = Button(description=' Columns', icon='barcode')
        cols = self._df.columns.tolist()
        cols = SelectMultiple(options=cols, value=cols)

        def _cols(c):
            self.columns = c.new
            self._update_output()
        cols.observe(_cols, names='value')
        cfol = Folder(cbut, _ListDict([('cols', cols)]))

        rbut = Button(description=' Rows', icon='bars')
        rows = IntRangeSlider(min=self.indexes[0],
                              max=self.indexes[1],
                              value=[0, 50])
        def _rows(c):
            self.indexes = c.new
            print(self.indexes)
            self._update_output()
        rows.observe(_rows, names='value')

        rfol = Folder(rbut, _ListDict([('rows', rows)]))

        return _ListDict([('close', close),
                          ('frame', frame),
                          ('cols', cfol),
                          ('rows', rfol)])

    def __init__(self, df, *args, **kwargs):

        self._df = df
        self.cur_frame = 0 if df.frame.astype(int).max() > 1 else -1
        self.indexes = (df.index[0], df.index[-1])
        self.start = df.index[0]
        self.columns = df.columns.tolist()

        self._controls = self._init_gui()

        gui = GUIBox(tuple(self._controls.values()))

        _ = kwargs.pop('layout', None)

        self._output = Output()
        children = [gui, self._output]
        self._update_output()

        super(DFBox, self).__init__(children, layout=_bboxlo, **kwargs)
