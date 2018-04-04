# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0

from exa import DataFrame
import numpy as np
import pandas as pd

class Tensor(DataFrame):
    """
    The tensor dataframe.

    +---------------+----------+-----------------------------------------+
    | Column        | Type     | Description                             |
    +===============+==========+=========================================+
    | xx            | float    | 0,0 position in tensor                  |
    +---------------+----------+-----------------------------------------+
    | xy            | float    | 0,1 position in tensor                  |
    +---------------+----------+-----------------------------------------+
    | xz            | float    | 0,2 position in tensor                  |
    +---------------+----------+-----------------------------------------+
    | yx            | float    | 1,0 position in tensor                  |
    +---------------+----------+-----------------------------------------+
    | yy            | float    | 1,1 position in tensor                  |
    +---------------+----------+-----------------------------------------+
    | yz            | float    | 1,2 position in tensor                  |
    +---------------+----------+-----------------------------------------+
    | zx            | float    | 3,0 position in tensor                  |
    +---------------+----------+-----------------------------------------+
    | zy            | float    | 3,1 position in tensor                  |
    +---------------+----------+-----------------------------------------+
    | zz            | float    | 3,2 position in tensor                  |
    +---------------+----------+-----------------------------------------+
    | frame         | category | frame value to which atach tensor       |
    +---------------+----------+-----------------------------------------+
    | atom          | int      | atom index of molecule to place tensor  |
    +---------------+----------+-----------------------------------------+
    | label         | category | label of the type of tensor             |
    +---------------+----------+-----------------------------------------+
    """
    _index = 'tensor'
    _columns = ['xx','xy','xz','yx','yy','yz','zx','zy','zz',
                'frame','atom','label']
    _categories = {'frame': np.int64, 'label': str}

    #@property
    #def _constructor(self):
    #    return Tensor

    @classmethod
    def from_file(cls, filename):
        """
        A file reader that will take a tensor file and extract all 
        necessary information. There is a specific file format in place 
        and is as follows.

        frame label atom
        xx xy xz
        yx yy yz
        zx zy zz

        For multiple tensors just append the same format as above without 
        whitespace unless leaving the frame, label, atom attributes as empty.
        
        Args:
            filename (str): file pathname
        
        Returns:
            tens (:class:`~exatomic.tensor.Tensor`): Tensor table with the tensor attributes
        """
        df = pd.read_csv(filename, delim_whitespace=True, header=None,
                         skip_blank_lines=False)
        meta = df[::4]
        idxs = meta.index.values
        n = len(idxs)
        df = df[~df.index.isin(idxs)]
        df[1] = df[1].astype(np.float64)
        df['grp'] = [i for i in range(n) for j in range(3)]
        df = pd.DataFrame(df.groupby('grp').apply(lambda x: 
                     x.unstack().values[:-3]).values.tolist(),
                     columns=['xx','xy','xz','yx','yy','yz','zx','zy','zz'])
#        scale = []
#        for i in df.index.values:
#            scale.append(5./abs(df.loc[i,:]).max().astype(np.float64))
        meta.reset_index(drop=True, inplace=True)
        meta.rename(columns={0: 'frame', 1: 'label', 2: 'atom'}, inplace=True)
        df = pd.concat([meta, df], axis=1)
        df['atom'] = df['atom'].astype(np.int64)
        df['frame'] = df['frame'].astype(np.int64)
#        df['scale'] = scale
#        print(df)
        return cls(df)

def add_tensor(uni, fp):
    """
    Simple function to add a tensor object to the universe.
    
    Args:
        uni (universe): Universe object
        fp (str): file pathname
    """
    uni.tensor = Tensor.from_file(fp)
