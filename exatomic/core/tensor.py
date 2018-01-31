import pandas as pd
import numpy as np
from exa import DataFrame

class Tensor(DataFrame):
    _index = 'tensor'
    _columns = ['xx','xy','xz','yx','yy','yz','zx','zy','zz',
                                                    'frame','atom','label']
    _categories = {'frame': np.int64, 'label': str}

#    def get_eigenvalues(self, 

    @classmethod
    def from_file(cls, filename, frame=0, atom_index=0):
#        df = pd.read_csv(filename, delim_whitespace=True, header=None)
#        print(df.unstack().values)
#        cols = ['xx','xy','xz','yx','yy','yz','zx','zy','zz']
#        df = pd.DataFrame([df.unstack().values], columns=cols)
#        df['frame'] = frame
#        df['label'] = 'efg'
#        df['atom'] = atom_index
        return cls(df)
