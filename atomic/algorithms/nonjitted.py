# -*- coding: utf-8 -*-

#from atomic import _np as np
from atomic import _np as np
from atomic import _pd as pd

def generate_minimal_framedf_from_onedf(onedf):
    '''
    Generates a minimal frame dataframe from a given one body object dataframe.
    '''
    def count(frame):
        '''
        Get the count of objects in the frame.
        '''
        return len(frame.index)
    frame = onedf.groupby(level='frame').apply(count).to_frame()
    frame.columns = ['count']
    return frame

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
