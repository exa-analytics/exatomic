# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0

from exa import DataFrame
import numpy as np
import pandas as pd

class Tensor(DataFrame):
    _index = 'tensor'
    _columns = ['xx','xy','xz','yx','yy','yz','zx','zy','zz',
                                                    'frame','atom','label']
    _categories = {'frame': np.int64, 'label': str}

#    def _get_eigen(self, n, df):
#        value = np.zeros((n,3))
#        vector = np.zeros((n,3,3))
#        for i in range(n):
#            A = df.loc[df.groupby([0,1,2])['grp'].filter(lambda x: 
#                                            x.sum()==i).index]
#            eigen = np.linalg.eig(A.values[:,[0,1,2]])
#            value[i] = eigen[0]
#            vector[i] = eigen[1]
#        valuedf = pd.DataFrame(data = np.transpose(value))
#        vectordf = pd.DataFrame(data = vector[0])
#        print(valuedf,vectordf)
#        return valuedf


    @classmethod
    def from_file(cls, filename, frame=0, atom_index=0):
        df = pd.read_csv(filename, delim_whitespace=True, header=None,
                         skip_blank_lines=False)
        meta = df[::4]
        idxs = meta.index.values
        n = len(idxs)
        df = df[~df.index.isin(idxs)]
        df[1] = df[1].astype(np.float64)
        df['grp'] = [i for i in range(n) for j in range(3)]
#        eigen = cls._get_eigen(cls, n, df)
        df = pd.DataFrame(df.groupby('grp').apply(lambda x: 
                          x.unstack().values[:-3]).values.tolist(),
                          columns=['xx','xy','xz','yx','yy','yz','zx','zy','zz'])
        meta.reset_index(drop=True, inplace=True)
        meta.rename(columns={0: 'frame', 1: 'label', 2: 'atom'}, inplace=True)
        df = pd.concat([meta, df], axis=1)
        df['atom'] = df['atom'].astype(np.int64)
        df['frame'] = df['frame'].astype(np.int64)
        return cls(df)

def add_tensor(uni, fp):
    uni.tensor = Tensor.from_file(fp)
