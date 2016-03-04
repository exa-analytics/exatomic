# -*- coding: utf-8 -*-
'''
Cube File Parsing and Composing
=============================================

'''
from exa import _pd as pd
from exa import _np as np

from atomic import Universe
from atomic.atom import Atom
from atomic.field import FieldMeta
from atomic import Isotope, Length
from atomic.frame import _min_frame_from_atom

def read_cubes(paths, frames=None, fieldxs=None,
               labels=None, universe=None, metadata={},
               **kwargs):
    '''
    Args
        paths (str or list): path(s) to cube files
        frames (int or list): frame(s) to which cubes belong
        fieldxs (int or list): field(s) indices or starting index
        labels (list): descriptions of cubes (12 character limit per label)
        universe (:class:`~atomic.container.Universe`): Universe

    Returns
        unikws: New or modified Universe

    Note
        In order to correctly attach cubes to an existing
        Universe, frame and fieldxs must be specified

    Warning
        If attaching to an existing universe only the field
        data will be updated!
    '''

    # Initialization and type declaration for fieldmeta
    if type(paths) is not list: paths = [paths]
    nrcubes = len(paths)
    unikws = {}
    fields = []
    atoms = []
    fmeta = np.zeros((nrcubes,), dtype=[
        ('ox', 'f8'), ('oy', 'f8'), ('oz', 'f8'),
        ('nx', 'f8'), ('dxi', 'f8'), ('dxj', 'f8'), ('dxk', 'f8'),
        ('ny', 'f8'), ('dyi', 'f8'), ('dyj', 'f8'), ('dyk', 'f8'),
        ('nz', 'f8'), ('dzi', 'f8'), ('dzj', 'f8'), ('dzk', 'f8'),
        ('label', 'U12'), ('frame', 'i8'), ('field', 'i8'),
    ])

    # Logic to handle function Args
    trkfrm, trkfld, trklbl = False, False, False
    if universe is None:

        if frames is None: frame = 0
        elif type(frames) is int: frame = frames
        elif len(frames) != nrcubes:
            print('Incorrect length of frames, proceeding with frame = 0.')
        else: trkfrm = True

        if fieldxs is None: fieldx = -1
        elif type(fieldxs) is int: fieldx = fieldxs
        elif len(fieldxs) != nrcubes:
            print('Incorrect length of fieldxs, proceeding with initial field = 0.')
        else: trkfld = True

        if labels is None: label = 0
        elif len(labels) != nrcubes:
            print('Incorrect length of labels, proceeding without them.')
            label = None
        else: trklbl = True

    for i, fl in enumerate(paths):
        # Logic to increment function Args
        if trkfrm: frame = frames[i]
        if trkfld: fieldx = fieldxs[i]
        else: fieldx += 1
        if trklbl: label = labels[i]

        # Read the cube file
        df = pd.read_csv(fl, delim_whitespace=True,
                         header=None, skiprows=[0, 1],
                         names=range(6), dtype=float)
        nat = int(df.iloc[0, 0])
        convert_xyz = False if nat > 0 else True
        nat = abs(nat)

        # FieldMeta table data
        brkmeta = df.iloc[0:4].values[:,:4].flatten()
        fmeta[i] = tuple(list(brkmeta[1:]) + [label, frame, fieldx])

        # Atom table data
        atoms.append(_gen_atom(df.iloc[4:nat + 4], convert_xyz, frame))

        # Field data
        fields.append(df[nat + 4:].stack().dropna().reset_index(drop=True).tolist())

    if not trkfrm:
        frameidxs = [frame]
        for i, atom in enumerate(atoms):
            if i == 0:
                continue
            cur = atoms[i - 1]
            try:
                if np.all(np.allclose(cur[['x', 'y', 'z']], atom[['x', 'y', 'z']])):
                    frameidxs.append(frame)
                else:
                    frame += 1
                    frameidxs.append(frame)
            except ValueError:
                frame += 1
                frameidxs.append(frame)
        tmp = [i['label'] for i in atoms]
        unikws['atom'] = pd.concat(atoms).reset_index(drop=True)
        unikws['atom']['frame'] = [fidx for i, fidx in enumerate(frameidxs) for j in tmp[i]]
        fmeta['frame'] = frameidxs
    else:
        unikws['atom'] = pd.concat(atoms).reset_index(drop=True)
    unikws['fieldmeta'] = FieldMeta(fmeta)
    unikws['fields'] = fields

    return Universe(**unikws)

def _gen_atom(smdf, convert, fdx):
    atomdf = smdf.loc[:, [0, 2, 3, 4]].reset_index(drop=True)
    atomdf.columns = ['symbol', 'x', 'y', 'z']
    if convert:
        atomdf['x'] *= Length['au', 'A']
        atomdf['y'] *= Length['au', 'A']
        atomdf['z'] *= Length['au', 'A']
    i = Isotope.Z_to_symbol_map
    atomdf['symbol'] = atomdf['symbol'].map(i.groupby(i.index).first())
    atomdf['frame'] = fdx
    atomdf['label'] = atomdf.index.values
    print(atomdf)
    return Atom(atomdf)
