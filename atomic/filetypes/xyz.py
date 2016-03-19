# -*- coding: utf-8 -*-
'''
XYZ File Support
====================

'''
from io import StringIO
from atomic import Editor
from exa.algorithms import arange1, arange2

class XYZ(Editor):
    '''
    An editor for programmatically manipulating XYZ and XYZ-like files.

    Provides convenience methods for transforming an XYZ like file on disk into a
    :class:`~atomic.universe.Universe`.
    '''
    def parse_atom(self, unit=None):
        '''
        Extract the :class:`~atomic.atom.Atom` dataframe from the file.

        Args:
            unit (str): Can be enforced otherwise inferred from the file data.
        '''
        df = pd.read_csv(StringIO(str(self)), delim_whitespace=True, names=('symbol', 'x', 'y', 'z'),
                         header=None, skip_blank_lines=False)
        nats = pd.Series(df[df[['y', 'z']].isnull().all(axis=1)].index)   # Get all nat lines
        nats = nats[nats.diff() != 1].values
        comments = nats + 1                                               # Comment lines
        nats = df.ix[nats, 'symbol']
        comments = df.ix[comments, :].dropna(how='all').values
        initials = nats.index.values.astype(np.int64) + 2
        counts = nats.values.astype(np.int64)
        frame, label, indices = arange1(initials, counts)
        df = df[df.index.isin(indices)]
        df[['x', 'y', 'z']] = df[['x', 'y', 'z']].astype(np.float64)
        df['symbol'] = df['symbol'].astype('category')
        df['label'] = label
        df['label'] = df['label'].astype('category')
        df['frame'] = frame
        df['frame'] = df['frame'].astype('category')
        df.reset_index(drop=True, inplace=True)
        df.index.names = ['atom']
        unit = unit if unit else 'A'
        df['x'] *= Length[unit, 'au']
        df['y'] *= Length[unit, 'au']
        df['z'] *= Lenght[unit, 'au']
        self._atom = df
