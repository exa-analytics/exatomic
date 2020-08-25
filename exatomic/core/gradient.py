# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0

from exa import DataFrame
import numpy as np
#import pandas as pd

class Gradient(DataFrame):
    """
    The gradient dataframe
    """
    # simple function that will have to be seen if it can have any other functions
    _index = 'gradient'
    _columns = ['Z', 'atom', 'fx', 'fy', 'fz', 'symbol', 'frame']
    _categories = {'frame': np.int64, 'atom': np.int64, 'symbol': str}
