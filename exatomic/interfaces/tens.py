# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
import six
#import numpy as np
import pandas as pd
from io import StringIO
#from exa import Series, TypedMeta
from exa import TypedMeta
from exatomic.core import Editor, Tensor


class Meta(TypedMeta):
    tensor = Tensor


class RTensor(six.with_metaclass(Meta, Editor)):
    """
    This is a simple script to read a rank-2 tensor file with frame, label and atom index
    labels. The format for such a file is,

    0: f=** l=** a=**
    1: xx   xy   xz
    2: yx   yy   yz
    3: zx   zy   zz
    4:
    5: Same as above for a second tensor

    """
## Must make this into a class that looks like the XYZ and Cube
#           classes. Must have something like parse_tensor.
#           Then on the Tensor class there should be something that can
#           be activated to find the eigenvalues and eigenvectors of the
#           matrix to plot the basis vectors.
#           Look at untitled1.ipynb for more info.
    _to_universe = Editor.to_universe

    def to_universe(self):
        raise NotImplementedError("Tensor file format has no atom table")

    def parse_tensor(self):
        df = pd.read_csv(StringIO(str(self)), delim_whitespace=True, header=None,
                            skip_blank_lines=False)
        #print(df)
        try:
            i=0
            data = ''
            while True:
                a = df.loc[[i*5],:].values[0]
                labels = []
                for lbl in a:
                    d = lbl.split('=')
                    labels.append(d[1])
                cols = ['xx','xy','xz','yx','yy','yz','zx','zy','zz']
                af = pd.DataFrame([df.loc[[i*5+1,i*5+2,i*5+3],:].unstack().values], \
                                            columns=cols)
                af['frame'] = labels[0] if labels[0] != '' else 0
                af['label'] = labels[1] if labels[1] != '' else None
                af['atom'] = labels[2] if labels[0] != '' else 0
                if i >= 1:
                    data = pd.concat([data,af],keys=[o for o in range(i+1)])
                    #data = data.append(af)
                    print('tens.py--------')
                    print(data)
                    print('---------------')
                else:
                    data = af
                i+=1
        except:
            print('tens.py--------')
            print("Reached EOF reading {} tensor".format(i))
            print(data)
            print('---------------')
        self.tensor = data

#    @classmethod
#    def from_universe(cls):
#        pass
