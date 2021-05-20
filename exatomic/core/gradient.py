# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0

from exa import DataFrame
import numpy as np
import pandas as pd

class Gradient(DataFrame):
    """
    The gradient dataframe
    """
    # simple function that will have to be seen if it can have any other functions
    _index = 'gradient'
    _columns = ['Z', 'atom', 'fx', 'fy', 'fz', 'symbol', 'frame']
    _categories = {'frame': np.int64, 'atom': np.int64, 'symbol': str}

    @property
    def stats(self):
        cols = ['fx', 'fy', 'fz']
        srs = []
        for _, data in self.groupby('frame'):
            norms = np.linalg.norm(data[cols].values, axis=1)
            rms = np.sqrt(np.mean(np.square(norms)))
            avg = np.mean(norms)
            max = norms.max()
            min = norms.min()
            sr = pd.Series([rms, avg, max, min], index=['rms', 'mean',
                                                        'max', 'min'])
            srs.append(sr)
        df = pd.concat(srs, axis=1, ignore_index=True).T
        return df

    @property
    def rms_grad(self):
        cols = ['fx', 'fy', 'fz']
        arr = []
        for _, data in self.groupby('frame'):
            val = np.sqrt(np.mean(np.square(data[cols].values)))
            arr.append(val)
        rms = pd.Series(arr)
        return rms

    @property
    def avg_grad(self):
        cols = ['fx', 'fy', 'fz']
        arr = []
        for _, data in self.groupby('frame'):
            val = np.mean(data[cols].values)
            arr.append(val)
        avg = pd.Series(arr)
        return avg

