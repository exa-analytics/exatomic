# -*- coding: utf-8 -*-

#from atomic import _np as np
import numpy as np

def expand(fridxs, natcnts):
    n = np.sum(natcnts)
    expanded = np.empty((n, ), dtype='i8')
    frame = np.empty((n, ), dtype='i8')
    one = np.empty((n, ), dtype='i8')
    k = 0
    for i, low in enumerate(fridxs):
        miter = range(low, low + natcnts[i])
        for j, index in enumerate(miter):
            frame[k] = i
            one[k] = j
            expanded[k] = index
            k += 1
    return frame, one, expanded
