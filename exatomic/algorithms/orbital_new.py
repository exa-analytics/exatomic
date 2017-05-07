# -*- coding: utf-8 -*-
'''
Evaluate Molecular Orbitals
##############################
With enough information, molecular orbitals may be
evaluated on a numerical grid and added to a universe.
'''
import numpy as np
from exatomic._config import config
from exatomic.algorithms.basis import cart_lml_count, spher_lml_count



def add_molecular_orbitals(universe, params=None, vectors=None):
    """
    Provided momatrix, basis_set and basis_set_order attributes,
    molecular orbitals of a universe are evaluated on a numerical grid
    and added to the field attribute. By default, the momatrix data
    evaluated is in the 'coefficient' column. vectors can be defined
    as a list of tuples corresponding to ('column', vector_number),
    eg. vectors=[('coefficient', 39), ('coefficient_beta', 39)] (provided
    'coefficient' and 'coefficient_beta' exist in the momatrix attribute).

    Args
        universe (exatomic.container.Universe): an exatomic universe
        params (iterable(len==3 or len==9)): rmin, rmax, nr or cartesian sets of those
        vectors (int, list, range): the MO vectors to evaluate

    Warning
        Removes any fields already attached to the universe
    """
    if universe.atom['frame'].cat.as_ordered().max() > 0:
        raise NotImplementedError('Molecular orbitals only works on single frame universes')
    if hasattr(universe, '_field'):
        del universe.__dict__['_field']

    x, y, z = _determine_params(universe, params)
    vectors = _determine_vectors(universe, vectors)

    atoms = universe.atom[['x', 'y', 'z']].values
    momatrix = universe.momatrix.square().values.T
    if universe.basis_set.spherical:
        centers, lvalues, mlvalues, compnt, nprims, npntrs, expnts, ds = _obtain_arrays(universe)
        #print(centers, lvalues, mlvalues, compnt, nprims, npntrs, expnts, sep='\n\n')
        c2s = [spherical(l, ml) for l, ml in zip(lvalues, mlvalues)]
    else:
        centers, lvalues, ivalues, jvalues, kvalues, compnt, nprims, npntrs, expnts, ds = _obtain_arrays(universe)
        #c2s = [cartesian(l) for l in ]
        #print(centers, lvalues, ivalues, jvalues, kvalues, compnt, nprims, npntrs, expnts, ds, sep='\n\n')
    nx = len(x)
    nv = len(vectors)
    fields = np.empty((nx, nv), dtype=np.float64)

    # It is hard getting the variable sized arrays working in a numbafied function
    # So just stack them all together with a component and pointer array to keep track
    c2sptrs = np.empty(len(c2s), dtype=np.int64)
    c2scmps = np.empty(len(c2s), dtype=np.int64)
    tot = 0
    for i, mat in enumerate(c2s):
        c2sptrs[i] = tot
        c2scmps[i] = len(mat)
        tot += len(mat)
    c2s = np.concatenate(c2s)

    print(expnts)
    print(expnts[-1])

    tmp = _test(compnt, nprims, npntrs, centers, atoms, x, y, z, expnts, ds, lvalues)
    print(tmp)
    tmp = _evaluate_basis(fields, c2s, c2sptrs, c2scmps, centers, compnt,
                          lvalues, nprims, npntrs, x, y, z, expnts, ds, atoms)
    print(tmp)

    return c2s, c2sptrs, c2scmps
    #tmp = _evaluate_mos(fields, x, y, z, momatrix, vectors, atoms, centers,
    #                    car2sph, car2sphptrs, car2sphcmps, compnt, nprims, npntrs, expnts, ds)
    #print(tmp)

def _test(compnt, nprims, npntrs, centers, atoms, x, y, z, expnts, ds, lvalues):
    cnt = 0
    xa = np.empty(len(x), dtype=np.float64)
    ya = np.empty(len(x), dtype=np.float64)
    za = np.empty(len(x), dtype=np.float64)
    r2 = np.empty(len(x), dtype=np.float64)
    ex = np.empty(len(x), dtype=np.float64)
    t1, t2, t3 = 0., 0., 0.
    a = 0.
    d = 0.
    for shl, mldegen in enumerate(compnt):
        npntr = npntrs[shl]
        nprim = nprims[shl]
        for ml in range(mldegen):
            l = lvalues[cnt]
            t1 = atoms[centers[cnt],0]
            t2 = atoms[centers[cnt],1]
            t3 = atoms[centers[cnt],2]
            xa = x - t1
            ya = y - t2
            za = z - t3
            r2 = xa * xa + ya * ya + za * za
            for k in range(nprim):
                a = expnts[npntr+k]
                d = ds[l,npntr+k]
                ex = a * r2
            cnt += 1
    return a, d

def _evaluate_basis(fields, c2s, c2sptrs, c2scmps, centers, compnt, lvalues,
                    nprims, npntrs, x, y, z, expnts, ds, atoms):
    cnt = 0
    #d, px, py, pz = 0., 0, 0, 0
    xa = np.empty(len(x), dtype=np.float64)
    ya = np.empty(len(x), dtype=np.float64)
    za = np.empty(len(x), dtype=np.float64)
    r2 = np.empty(len(x), dtype=np.float64)
    pe = np.empty(len(x), dtype=np.float64)
    ex = np.empty(len(x), dtype=np.float64)
    to = np.empty(len(x), dtype=np.float64)
    a0, a1, a2, a, d = 0., 0., 0., 0., 0.
    e = 2.718281828459045
    for shl, mldegen in enumerate(compnt):
        npntr = npntrs[shl]
        nprim = nprims[shl]
        for ml in range(mldegen):
            ptr = c2sptrs[cnt]
            l = lvalues[cnt]
            a0 = atoms[centers[cnt],0]
            a1 = atoms[centers[cnt],1]
            a2 = atoms[centers[cnt],2]
            xa = x - a0
            ya = y - a1
            za = z - a2
            r2 = xa * xa + ya * ya + za * za
            for j in range(c2scmps[cnt]):
                c = c2s[ptr+j,0]
                px = c2s[ptr+j,1]
                py = c2s[ptr+j,2]
                pz = c2s[ptr+j,3]
                pe = c * xa ** px * ya ** py * za ** pz
                for k in range(nprim):
                    a = expnts[npntr+k]
                    d = ds[l,npntr+k]
                    ex = d * e ** (-a * r2)
                #fields[:,cnt] = x
                #tot2 += c2s[ptr+j,0]
                #tot2 += c2s[ptr+j,1]
                #tot2 += c2s[ptr+j,2]
                #tot2 += c2s[ptr+j,3]
            cnt += 1
    #return xa, ya, za
    return xa, ya, za
#    return fields

def _evaluate_mos(fields, x, y, z, momatrix, vectors, atoms, centers,
                  car2sph, car2sphptrs, car2sphcmps, compnt, nprims, npntrs, expnts, ds):
    cnt = 0
    tot = 0.
    nbas = momatrix.shape[0]
    bfns = np.zeros((len(x), nbas), dtype=np.float64)
    nshell = len(compnt)
    for shl, mldegen in enumerate(compnt):
        nprim = nprims[shl]
        pntr = npntrs[shl]
        for ml in range(mldegen):
            carptr = car2sphptrs[cnt]
            carcmp = car2sphcmps[cnt]
            xa = x - atoms[centers[shl], 0]
            ya = y - atoms[centers[shl], 1]
            za = z - atoms[centers[shl], 2]
            r2 = xa * xa + ya * ya + za * za
            for car in range(carcmp):
                tot += car2sph[carptr+car,0]
                tot += car2sph[carptr+car,1]
                tot += car2sph[carptr+car,2]
                tot += car2sph[carptr+car,3]
            cnt += 1
    return tot
    #  subroutine driver(mos,xlo,xhi,nx,ylo,yhi,ny,zlo,zhi,nz,
    # &                  nvectors,vectors,nbas,momat,
    # &                  natoms,atompos,centers,labels,
    # &                  nshell,ncomp,nprim,nptr,
    # &                  nexp,ntypes,alphas,coefs)

def _obtain_arrays(universe):
    shells = universe.basis_set.functions_by_shell()
    shells = shells.groupby(shells.index.get_level_values(0)).apply(lambda x: x.sum())
    nshell = universe.atom['set'].map(shells).sum()
    nexpnt = (universe.basis_set[np.abs(universe.basis_set['d']) > 0].groupby('set').apply(
                                 lambda x: x.shape[0]) * universe.atom['set'].value_counts()).sum()
    centers = universe.basis_set_order['center'].values
    lvalues = np.array(universe.basis_set_order['L'].values)
    if universe.basis_set.spherical:
        mlvalues = universe.basis_set_order['ml'].values
        lml_count = spher_lml_count
    else:
        ivalues = universe.basis_set_order['l'].values
        jvalues = universe.basis_set_order['m'].values
        kvalues = universe.basis_set_order['n'].values
        lml_count = cart_lml_count
    compnt = np.empty(nshell, dtype=np.int64)
    nprims = np.empty(nshell, dtype=np.int64)
    npntrs = np.empty(nshell, dtype=np.int64)
    expnts = np.empty(nexpnt, dtype=np.float64)
    lmax = universe.basis_set['L'].cat.as_ordered().max()
    ds = np.empty((lmax + 1, nexpnt), dtype=np.float64)
    bases = universe.basis_set[np.abs(universe.basis_set['d']) > 0].groupby('set')
    cnt, ptr, xpc = 0, 1, 0
    for seht, center in zip(universe.atom['set'], universe.atom.index):
        b = bases.get_group(seht)
        for sh, grp in b.groupby('shell'):
            if len(grp) == 0: continue
            compnt[cnt] = lml_count[grp['L'].values[0]]
            nprims[cnt] = grp.shape[0]
            npntrs[cnt] = ptr
            ptr += nprims[cnt]
            cnt += 1
        for l, d, exp in zip(b['L'], b['d'], b['alpha']):
            expnts[xpc] = exp
            for i, ang in enumerate(ds):
                ds[i][xpc] = d if i == l else 0.
            xpc += 1
    if universe.basis_set.spherical:
        return centers, lvalues, mlvalues, compnt, nprims, npntrs, expnts, ds
    else:
        return centers, lvalues, ivalues, jvalues, kvalues, compnt, nprims, npntrs, expnts, ds

def _determine_params(universe, params):
    if params is None:
        dr = 51
        rmin = universe.atom[['x', 'y', 'z']].min().min()
        rmax = universe.atom[['x', 'y', 'z']].max().max()
        return num_grid(rmin, rmax, dr)
    if len(params) % 3 == 0:
        return num_grid(*params)
    raise Exception('params must have len divisible by 3')

def _determine_mocoefs(universe, vectors):
    raise NotImplementedError('I will get to it.')

def _determine_vectors(universe, vectors):
    if isinstance(vectors, int):
        return np.array([vectors])
    if isinstance(vectors, range):
        return np.array(vectors)
    if isinstance(vectors, (list, tuple)):
        try:
            t = len(vectors[0])
            cols = [i[0] for i in vectors]
            if not all(col in universe.momatrix.columns for col in cols):
                raise Exception('vectors column must be in universe.momatrix')
            return _determine_mocoefs(universe, vectors)
        except:
            raise Exception('vectors must contain (column, vector) pairs')
    if vectors is None:
        total = universe.momatrix['orbital'].max()
        if total < 20:
            return np.array(range(total))
        try:
            try:
                na = universe.frame['alpha_electrons'].ix[0]
                nb = universe.frame['beta_electrons'].ix[0]
                nclosed = (na + nb) // 2
            except KeyError:
                nclosed = universe.frame['total_electrons'].ix[0] // 2
        except KeyError:
            try:
                nclosed = universe.atom['Zeff'].sum() // 2
            except KeyError:
                nclosed = universe.atom['Z'].sum() // 2
        if nclosed - 15 < 0:
            return np.array(range(nclosed + 15))
        else:
            return np.array(range(nclosed - 15, nclosed + 5))


def num_grid(xmin, xmax, nx, ymin=0., ymax=0., ny=0, zmin=0., zmax=0., nz=0):
    ny = nx if ny == 0 else ny
    ymin = xmin if ymin == 0. else ymin
    ymax = xmax if ymax == 0. else ymax
    nz = nx if nz == 0 else nz
    zmin = xmin if zmin == 0. else zmin
    zmax = xmax if zmax == 0. else zmax
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    dz = (zmax - zmin) / nz
    xt = np.empty(nx, dtype=np.float64)
    yt = np.empty(ny, dtype=np.float64)
    zt = np.empty(nz, dtype=np.float64)
    for i in range(nx):
        xt[i] = xmin + i * dx
    for i in range(ny):
        yt[i] = ymin + i * dy
    for i in range(nz):
        zt[i] = zmin + i * dz
    tot = nx * ny * nz
    x = np.empty(tot, dtype=np.float64)
    y = np.empty(tot, dtype=np.float64)
    z = np.empty(tot, dtype=np.float64)
    c = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x[c] = xt[i]
                y[c] = yt[j]
                z[c] = zt[k]
                c += 1
    return x, y, z

def cartesian(l, ncart):
    ret = np.zeros((ncart, 4), dtype=np.float64)
    if l == 0:
        ret[0,:] = (1.0, 0.0, 0.0, 0.0)
        return ret
    elif l == 1:
        ret[0 ,:] = (1., 1.0, 0.0, 0.0)
        ret[1 ,:] = (1., 0.0, 1.0, 0.0)
        ret[2 ,:] = (1., 0.0, 0.0, 1.0)
        return ret
    elif l == 2:
        ret[0 ,:] = (1., 2.0, 0.0, 0.0)
        ret[1 ,:] = (1., 1.0, 1.0, 0.0)
        ret[2 ,:] = (1., 1.0, 0.0, 1.0)
        ret[3 ,:] = (1., 0.0, 2.0, 0.0)
        ret[4 ,:] = (1., 0.0, 1.0, 1.0)
        ret[5 ,:] = (1., 0.0, 0.0, 2.0)
        return ret
    elif l == 3:
        ret[0 ,:] = (1., 3.0, 0.0, 0.0)
        ret[1 ,:] = (1., 2.0, 1.0, 0.0)
        ret[2 ,:] = (1., 2.0, 0.0, 1.0)
        ret[3 ,:] = (1., 1.0, 2.0, 0.0)
        ret[4 ,:] = (1., 1.0, 1.0, 1.0)
        ret[5 ,:] = (1., 1.0, 0.0, 2.0)
        ret[6 ,:] = (1., 0.0, 3.0, 0.0)
        ret[7 ,:] = (1., 0.0, 2.0, 1.0)
        ret[8 ,:] = (1., 0.0, 1.0, 2.0)
        ret[9 ,:] = (1., 0.0, 0.0, 3.0)
        return ret
    elif l == 4:
        ret[0 ,:] = (1., 4.0, 0.0, 0.0)
        ret[1 ,:] = (1., 3.0, 1.0, 0.0)
        ret[2 ,:] = (1., 3.0, 0.0, 1.0)
        ret[3 ,:] = (1., 2.0, 2.0, 0.0)
        ret[4 ,:] = (1., 2.0, 1.0, 1.0)
        ret[5 ,:] = (1., 2.0, 0.0, 2.0)
        ret[6 ,:] = (1., 1.0, 3.0, 0.0)
        ret[7 ,:] = (1., 1.0, 2.0, 1.0)
        ret[8 ,:] = (1., 1.0, 1.0, 2.0)
        ret[9 ,:] = (1., 1.0, 0.0, 3.0)
        ret[10,:] = (1., 0.0, 4.0, 0.0)
        ret[11,:] = (1., 0.0, 3.0, 1.0)
        ret[12,:] = (1., 0.0, 2.0, 2.0)
        ret[13,:] = (1., 0.0, 1.0, 3.0)
        ret[14,:] = (1., 0.0, 0.0, 4.0)
        return ret
    elif l == 5:
        ret[0 ,:] = (1., 5.0, 0.0, 0.0)
        ret[1 ,:] = (1., 4.0, 1.0, 0.0)
        ret[2 ,:] = (1., 4.0, 0.0, 1.0)
        ret[3 ,:] = (1., 3.0, 2.0, 0.0)
        ret[4 ,:] = (1., 3.0, 1.0, 1.0)
        ret[5 ,:] = (1., 3.0, 0.0, 2.0)
        ret[6 ,:] = (1., 2.0, 3.0, 0.0)
        ret[7 ,:] = (1., 2.0, 2.0, 1.0)
        ret[8 ,:] = (1., 2.0, 1.0, 2.0)
        ret[9 ,:] = (1., 2.0, 0.0, 3.0)
        ret[10,:] = (1., 1.0, 4.0, 0.0)
        ret[11,:] = (1., 1.0, 3.0, 1.0)
        ret[12,:] = (1., 1.0, 2.0, 2.0)
        ret[13,:] = (1., 1.0, 1.0, 3.0)
        ret[14,:] = (1., 1.0, 0.0, 4.0)
        ret[15,:] = (1., 0.0, 5.0, 0.0)
        ret[16,:] = (1., 0.0, 4.0, 1.0)
        ret[17,:] = (1., 0.0, 3.0, 2.0)
        ret[18,:] = (1., 0.0, 2.0, 3.0)
        ret[19,:] = (1., 0.0, 1.0, 4.0)
        ret[20,:] = (1., 0.0, 0.0, 5.0)
        return ret
    elif l == 6:
        ret[0 ,:] = (1., 6.0, 0.0, 0.0)
        ret[1 ,:] = (1., 5.0, 1.0, 0.0)
        ret[2 ,:] = (1., 5.0, 0.0, 1.0)
        ret[3 ,:] = (1., 4.0, 2.0, 0.0)
        ret[4 ,:] = (1., 4.0, 1.0, 1.0)
        ret[5 ,:] = (1., 4.0, 0.0, 2.0)
        ret[6 ,:] = (1., 3.0, 3.0, 0.0)
        ret[7 ,:] = (1., 3.0, 2.0, 1.0)
        ret[8 ,:] = (1., 3.0, 1.0, 2.0)
        ret[9 ,:] = (1., 3.0, 0.0, 3.0)
        ret[10,:] = (1., 2.0, 4.0, 0.0)
        ret[11,:] = (1., 2.0, 3.0, 1.0)
        ret[12,:] = (1., 2.0, 2.0, 2.0)
        ret[13,:] = (1., 2.0, 1.0, 3.0)
        ret[14,:] = (1., 2.0, 0.0, 4.0)
        ret[15,:] = (1., 1.0, 5.0, 0.0)
        ret[16,:] = (1., 1.0, 4.0, 1.0)
        ret[17,:] = (1., 1.0, 3.0, 2.0)
        ret[18,:] = (1., 1.0, 2.0, 3.0)
        ret[19,:] = (1., 1.0, 1.0, 4.0)
        ret[20,:] = (1., 1.0, 0.0, 5.0)
        ret[21,:] = (1., 0.0, 6.0, 0.0)
        ret[22,:] = (1., 0.0, 5.0, 1.0)
        ret[23,:] = (1., 0.0, 4.0, 2.0)
        ret[24,:] = (1., 0.0, 3.0, 3.0)
        ret[25,:] = (1., 0.0, 2.0, 4.0)
        ret[26,:] = (1., 0.0, 1.0, 5.0)
        ret[27,:] = (1., 0.0, 0.0, 6.0)
        return ret

def spherical(l, ml):
    if l == 0:
        ncart = 1
        ret = np.zeros((ncart, 4), dtype=np.float64)
        ret[0,:] = (1.0, 0.0, 0.0, 0.0)
        return ret
    elif l == 1:
        if ml == -1:
            ncart = 1
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = ( 1.00000000, 0.0, 1.0, 0.0)
            return ret
        elif ml == 0:
            ncart = 1
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = ( 1.00000000, 0.0, 0.0, 1.0)
            return ret
        elif ml == 1:
            ncart = 1
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = ( 1.00000000, 1.0, 0.0, 0.0)
            return ret
    elif l == 2:
        if ml == -2:
            ncart = 1
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = ( 1.73205081, 1.0, 1.0, 0.0)
            return ret
        elif ml == -1:
            ncart = 1
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = ( 1.73205081, 0.0, 1.0, 1.0)
            return ret
        elif ml == 0:
            ncart = 3
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = (-0.50000000, 2.0, 0.0, 0.0)
            ret[1,:] = (-0.50000000, 0.0, 2.0, 0.0)
            ret[2,:] = ( 1.00000000, 0.0, 0.0, 2.0)
            return ret
        elif ml == 1:
            ncart = 1
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = ( 1.73205081, 1.0, 0.0, 1.0)
            return ret
        elif ml == 2:
            ncart = 2
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = ( 0.86602540, 2.0, 0.0, 0.0)
            ret[1,:] = (-0.86602540, 0.0, 2.0, 0.0)
            return ret
    elif l == 3:
        if ml == -3:
            ncart = 2
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = ( 2.37170825, 2.0, 1.0, 0.0)
            ret[1,:] = (-0.79056942, 0.0, 3.0, 0.0)
            return ret
        elif ml == -2:
            ncart = 1
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = ( 3.87298335, 1.0, 1.0, 1.0)
            return ret
        elif ml == -1:
            ncart = 3
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = (-0.61237244, 2.0, 1.0, 0.0)
            ret[1,:] = (-0.61237244, 0.0, 3.0, 0.0)
            ret[2,:] = ( 2.44948974, 0.0, 1.0, 2.0)
            return ret
        elif ml == 0:
            ncart = 3
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = (-1.50000000, 2.0, 0.0, 1.0)
            ret[1,:] = (-1.50000000, 0.0, 2.0, 1.0)
            ret[2,:] = ( 1.00000000, 0.0, 0.0, 3.0)
            return ret
        elif ml == 1:
            ncart = 3
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = (-0.61237244, 3.0, 0.0, 0.0)
            ret[1,:] = (-0.61237244, 1.0, 2.0, 0.0)
            ret[2,:] = ( 2.44948974, 1.0, 0.0, 2.0)
            return ret
        elif ml == 2:
            ncart = 2
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = ( 1.93649167, 2.0, 0.0, 1.0)
            ret[1,:] = (-1.93649167, 0.0, 2.0, 1.0)
            return ret
        elif ml == 3:
            ncart = 2
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = ( 0.79056942, 3.0, 0.0, 0.0)
            ret[1,:] = (-2.37170825, 1.0, 2.0, 0.0)
            return ret
    elif l == 4:
        if ml == -4:
            ncart = 2
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = ( 2.95803989, 3.0, 1.0, 0.0)
            ret[1,:] = (-2.95803989, 1.0, 3.0, 0.0)
            return ret
        elif ml == -3:
            ncart = 2
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = ( 6.27495020, 2.0, 1.0, 1.0)
            ret[1,:] = (-2.09165007, 0.0, 3.0, 1.0)
            return ret
        elif ml == -2:
            ncart = 3
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = (-1.11803399, 3.0, 1.0, 0.0)
            ret[1,:] = (-1.11803399, 1.0, 3.0, 0.0)
            ret[2,:] = ( 6.70820393, 1.0, 1.0, 2.0)
            return ret
        elif ml == -1:
            ncart = 3
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = (-2.37170825, 2.0, 1.0, 1.0)
            ret[1,:] = (-2.37170825, 0.0, 3.0, 1.0)
            ret[2,:] = ( 3.16227766, 0.0, 1.0, 3.0)
            return ret
        elif ml == 0:
            ncart = 6
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = ( 0.37500000, 4.0, 0.0, 0.0)
            ret[1,:] = ( 0.75000000, 2.0, 2.0, 0.0)
            ret[2,:] = (-3.00000000, 2.0, 0.0, 2.0)
            ret[3,:] = ( 0.37500000, 0.0, 4.0, 0.0)
            ret[4,:] = (-3.00000000, 0.0, 2.0, 2.0)
            ret[5,:] = ( 1.00000000, 0.0, 0.0, 4.0)
            return ret
        elif ml == 1:
            ncart = 3
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = (-2.37170825, 3.0, 0.0, 1.0)
            ret[1,:] = (-2.37170825, 1.0, 2.0, 1.0)
            ret[2,:] = ( 3.16227766, 1.0, 0.0, 3.0)
            return ret
        elif ml == 2:
            ncart = 4
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = (-0.55901699, 4.0, 0.0, 0.0)
            ret[1,:] = ( 3.35410197, 2.0, 0.0, 2.0)
            ret[2,:] = ( 0.55901699, 0.0, 4.0, 0.0)
            ret[3,:] = (-3.35410197, 0.0, 2.0, 2.0)
            return ret
        elif ml == 3:
            ncart = 2
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = ( 2.09165007, 3.0, 0.0, 1.0)
            ret[1,:] = (-6.27495020, 1.0, 2.0, 1.0)
            return ret
        elif ml == 4:
            ncart = 3
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = ( 0.73950997, 4.0, 0.0, 0.0)
            ret[1,:] = (-4.43705984, 2.0, 2.0, 0.0)
            ret[2,:] = ( 0.73950997, 0.0, 4.0, 0.0)
            return ret
    elif l == 5:
        if ml == -5:
            ncart = 3
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = ( 3.50780380, 4.0, 1.0, 0.0)
            ret[1,:] = (-7.01560760, 2.0, 3.0, 0.0)
            ret[2,:] = ( 0.70156076, 0.0, 5.0, 0.0)
            return ret
        elif ml == -4:
            ncart = 2
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = ( 8.87411967, 3.0, 1.0, 1.0)
            ret[1,:] = (-8.87411967, 1.0, 3.0, 1.0)
            return ret
        elif ml == -3:
            ncart = 5
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = (-1.56873755, 4.0, 1.0, 0.0)
            ret[1,:] = (-1.04582503, 2.0, 3.0, 0.0)
            ret[2,:] = ( 12.54990040, 2.0, 1.0, 2.0)
            ret[3,:] = ( 0.52291252, 0.0, 5.0, 0.0)
            ret[4,:] = (-4.18330013, 0.0, 3.0, 2.0)
            return ret
        elif ml == -2:
            ncart = 3
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = (-5.12347538, 3.0, 1.0, 1.0)
            ret[1,:] = (-5.12347538, 1.0, 3.0, 1.0)
            ret[2,:] = ( 10.24695077, 1.0, 1.0, 3.0)
            return ret
        elif ml == -1:
            ncart = 6
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = ( 0.48412292, 4.0, 1.0, 0.0)
            ret[1,:] = ( 0.96824584, 2.0, 3.0, 0.0)
            ret[2,:] = (-5.80947502, 2.0, 1.0, 2.0)
            ret[3,:] = ( 0.48412292, 0.0, 5.0, 0.0)
            ret[4,:] = (-5.80947502, 0.0, 3.0, 2.0)
            ret[5,:] = ( 3.87298335, 0.0, 1.0, 4.0)
            return ret
        elif ml == 0:
            ncart = 6
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = ( 1.87500000, 4.0, 0.0, 1.0)
            ret[1,:] = ( 3.75000000, 2.0, 2.0, 1.0)
            ret[2,:] = (-5.00000000, 2.0, 0.0, 3.0)
            ret[3,:] = ( 1.87500000, 0.0, 4.0, 1.0)
            ret[4,:] = (-5.00000000, 0.0, 2.0, 3.0)
            ret[5,:] = ( 1.00000000, 0.0, 0.0, 5.0)
            return ret
        elif ml == 1:
            ncart = 6
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = ( 0.48412292, 5.0, 0.0, 0.0)
            ret[1,:] = ( 0.96824584, 3.0, 2.0, 0.0)
            ret[2,:] = (-5.80947502, 3.0, 0.0, 2.0)
            ret[3,:] = ( 0.48412292, 1.0, 4.0, 0.0)
            ret[4,:] = (-5.80947502, 1.0, 2.0, 2.0)
            ret[5,:] = ( 3.87298335, 1.0, 0.0, 4.0)
            return ret
        elif ml == 2:
            ncart = 4
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = (-2.56173769, 4.0, 0.0, 1.0)
            ret[1,:] = ( 5.12347538, 2.0, 0.0, 3.0)
            ret[2,:] = ( 2.56173769, 0.0, 4.0, 1.0)
            ret[3,:] = (-5.12347538, 0.0, 2.0, 3.0)
            return ret
        elif ml == 3:
            ncart = 5
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = (-0.52291252, 5.0, 0.0, 0.0)
            ret[1,:] = ( 1.04582503, 3.0, 2.0, 0.0)
            ret[2,:] = ( 4.18330013, 3.0, 0.0, 2.0)
            ret[3,:] = ( 1.56873755, 1.0, 4.0, 0.0)
            ret[4,:] = (-12.54990040, 1.0, 2.0, 2.0)
            return ret
        elif ml == 4:
            ncart = 3
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = ( 2.21852992, 4.0, 0.0, 1.0)
            ret[1,:] = (-13.31117951, 2.0, 2.0, 1.0)
            ret[2,:] = ( 2.21852992, 0.0, 4.0, 1.0)
            return ret
        elif ml == 5:
            ncart = 3
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = ( 0.70156076, 5.0, 0.0, 0.0)
            ret[1,:] = (-7.01560760, 3.0, 2.0, 0.0)
            ret[2,:] = ( 3.50780380, 1.0, 4.0, 0.0)
            return ret
    elif l == 6:
        if ml == -6:
            ncart = 3
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = ( 4.03015974, 5.0, 1.0, 0.0)
            ret[1,:] = (-13.43386579, 3.0, 3.0, 0.0)
            ret[2,:] = ( 4.03015974, 1.0, 5.0, 0.0)
            return ret
        elif ml == -5:
            ncart = 3
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = ( 11.63406904, 4.0, 1.0, 1.0)
            ret[1,:] = (-23.26813809, 2.0, 3.0, 1.0)
            ret[2,:] = ( 2.32681381, 0.0, 5.0, 1.0)
            return ret
        elif ml == -4:
            ncart = 4
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = (-1.98431348, 5.0, 1.0, 0.0)
            ret[1,:] = ( 19.84313483, 3.0, 1.0, 2.0)
            ret[2,:] = ( 1.98431348, 1.0, 5.0, 0.0)
            ret[3,:] = (-19.84313483, 1.0, 3.0, 2.0)
            return ret
        elif ml == -3:
            ncart = 5
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = (-8.15139942, 4.0, 1.0, 1.0)
            ret[1,:] = (-5.43426628, 2.0, 3.0, 1.0)
            ret[2,:] = ( 21.73706512, 2.0, 1.0, 3.0)
            ret[3,:] = ( 2.71713314, 0.0, 5.0, 1.0)
            ret[4,:] = (-7.24568837, 0.0, 3.0, 3.0)
            return ret
        elif ml == -2:
            ncart = 6
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = ( 0.90571105, 5.0, 1.0, 0.0)
            ret[1,:] = ( 1.81142209, 3.0, 3.0, 0.0)
            ret[2,:] = (-14.49137675, 3.0, 1.0, 2.0)
            ret[3,:] = ( 0.90571105, 1.0, 5.0, 0.0)
            ret[4,:] = (-14.49137675, 1.0, 3.0, 2.0)
            ret[5,:] = ( 14.49137675, 1.0, 1.0, 4.0)
            return ret
        elif ml == -1:
            ncart = 6
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = ( 2.86410981, 4.0, 1.0, 1.0)
            ret[1,:] = ( 5.72821962, 2.0, 3.0, 1.0)
            ret[2,:] = (-11.45643924, 2.0, 1.0, 3.0)
            ret[3,:] = ( 2.86410981, 0.0, 5.0, 1.0)
            ret[4,:] = (-11.45643924, 0.0, 3.0, 3.0)
            ret[5,:] = ( 4.58257569, 0.0, 1.0, 5.0)
            return ret
        elif ml == 0:
            ncart = 10
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = (-0.31250000, 6.0, 0.0, 0.0)
            ret[1,:] = (-0.93750000, 4.0, 2.0, 0.0)
            ret[2,:] = ( 5.62500000, 4.0, 0.0, 2.0)
            ret[3,:] = (-0.93750000, 2.0, 4.0, 0.0)
            ret[4,:] = ( 11.25000000, 2.0, 2.0, 2.0)
            ret[5,:] = (-7.50000000, 2.0, 0.0, 4.0)
            ret[6,:] = (-0.31250000, 0.0, 6.0, 0.0)
            ret[7,:] = ( 5.62500000, 0.0, 4.0, 2.0)
            ret[8,:] = (-7.50000000, 0.0, 2.0, 4.0)
            ret[9,:] = ( 1.00000000, 0.0, 0.0, 6.0)
            return ret
        elif ml == 1:
            ncart = 6
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = ( 2.86410981, 5.0, 0.0, 1.0)
            ret[1,:] = ( 5.72821962, 3.0, 2.0, 1.0)
            ret[2,:] = (-11.45643924, 3.0, 0.0, 3.0)
            ret[3,:] = ( 2.86410981, 1.0, 4.0, 1.0)
            ret[4,:] = (-11.45643924, 1.0, 2.0, 3.0)
            ret[5,:] = ( 4.58257569, 1.0, 0.0, 5.0)
            return ret
        elif ml == 2:
            ncart = 8
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = ( 0.45285552, 6.0, 0.0, 0.0)
            ret[1,:] = ( 0.45285552, 4.0, 2.0, 0.0)
            ret[2,:] = (-7.24568837, 4.0, 0.0, 2.0)
            ret[3,:] = (-0.45285552, 2.0, 4.0, 0.0)
            ret[4,:] = ( 7.24568837, 2.0, 0.0, 4.0)
            ret[5,:] = (-0.45285552, 0.0, 6.0, 0.0)
            ret[6,:] = ( 7.24568837, 0.0, 4.0, 2.0)
            ret[7,:] = (-7.24568837, 0.0, 2.0, 4.0)
            return ret
        elif ml == 3:
            ncart = 5
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = (-2.71713314, 5.0, 0.0, 1.0)
            ret[1,:] = ( 5.43426628, 3.0, 2.0, 1.0)
            ret[2,:] = ( 7.24568837, 3.0, 0.0, 3.0)
            ret[3,:] = ( 8.15139942, 1.0, 4.0, 1.0)
            ret[4,:] = (-21.73706512, 1.0, 2.0, 3.0)
            return ret
        elif ml == 4:
            ncart = 7
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = (-0.49607837, 6.0, 0.0, 0.0)
            ret[1,:] = ( 2.48039185, 4.0, 2.0, 0.0)
            ret[2,:] = ( 4.96078371, 4.0, 0.0, 2.0)
            ret[3,:] = ( 2.48039185, 2.0, 4.0, 0.0)
            ret[4,:] = (-29.76470225, 2.0, 2.0, 2.0)
            ret[5,:] = (-0.49607837, 0.0, 6.0, 0.0)
            ret[6,:] = ( 4.96078371, 0.0, 4.0, 2.0)
            return ret
        elif ml == 5:
            ncart = 3
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = ( 2.32681381, 5.0, 0.0, 1.0)
            ret[1,:] = (-23.26813809, 3.0, 2.0, 1.0)
            ret[2,:] = ( 11.63406904, 1.0, 4.0, 1.0)
            return ret
        elif ml == 6:
            ncart = 4
            ret = np.zeros((ncart, 4), dtype=np.float64)
            ret[0,:] = ( 0.67169329, 6.0, 0.0, 0.0)
            ret[1,:] = (-10.07539934, 4.0, 2.0, 0.0)
            ret[2,:] = ( 10.07539934, 2.0, 4.0, 0.0)
            ret[3,:] = (-0.67169329, 0.0, 6.0, 0.0)
            return ret


if config['dynamic']['numba'] == 'true':
    from numba import jit, vectorize
    cartesian = jit(nopython=True, cache=True, nogil=True)(cartesian)
    spherical = jit(nopython=True, cache=True, nogil=True)(spherical)
    num_grid = jit(nopython=True, cache=True, nogil=True)(num_grid)
    _test = jit(nopython=True, cache=True, nogil=True)(_test)
    _evaluate_mos = jit(nopython=True, cache=True, nogil=True)(_evaluate_mos)
    _evaluate_basis = jit(nopython=True, cache=True, nogil=True)(_evaluate_basis)

## -*- coding: utf-8 -*-
#'''
#Evaluate Molecular Orbitals
###############################
#With enough information, molecular orbitals may be
#evaluated on a numerical grid and added to a universe.
#'''
#import numpy as np
#from exatomic._config import config
#from exatomic.algorithms.basis import cart_lml_count, spher_lml_count
#
#
#
#def add_molecular_orbitals(universe, params=None, vectors=None):
#    """
#    Provided momatrix, basis_set and basis_set_order attributes,
#    molecular orbitals of a universe are evaluated on a numerical grid
#    and added to the field attribute. By default, the momatrix data
#    evaluated is in the 'coefficient' column. vectors can be defined
#    as a list of tuples corresponding to ('column', vector_number),
#    eg. vectors=[('coefficient', 39), ('coefficient_beta', 39)] (provided
#    'coefficient' and 'coefficient_beta' exist in the momatrix attribute).
#
#    Args
#        universe (exatomic.container.Universe): an exatomic universe
#        params (iterable(len==3 or len==9)): rmin, rmax, nr or cartesian sets of those
#        vectors (int, list, range): the MO vectors to evaluate
#
#    Warning
#        Removes any fields already attached to the universe
#    """
#    if universe.atom['frame'].cat.as_ordered().max() > 0:
#        raise NotImplementedError('Molecular orbitals only works on single frame universes')
#    if hasattr(universe, '_field'):
#        del universe.__dict__['_field']
#
#    x, y, z = _determine_params(universe, params)
#    vectors = _determine_vectors(universe, vectors)
#
#    atoms = universe.atom[['x', 'y', 'z']].values
#    momatrix = universe.momatrix.square().values.T
#    if universe.basis_set.spherical:
#        centers, lvalues, mlvalues, compnt, nprims, npntrs, expnts, ds = _obtain_arrays(universe)
#        #print(centers, lvalues, mlvalues, compnt, nprims, npntrs, expnts, sep='\n\n')
#        c2s = [spherical(l, ml) for l, ml in zip(lvalues, mlvalues)]
#    else:
#        centers, lvalues, ivalues, jvalues, kvalues, compnt, nprims, npntrs, expnts, ds = _obtain_arrays(universe)
#        #c2s = [cartesian(l) for l in ]
#        #print(centers, lvalues, ivalues, jvalues, kvalues, compnt, nprims, npntrs, expnts, ds, sep='\n\n')
#    nx = len(x)
#    nv = len(vectors)
#    fields = np.empty((nx, nv), dtype=np.float64)
#
#    # It is hard getting the variable sized arrays working in a numbafied function
#    # So just stack them all together with a component and pointer array to keep track
#    c2sptrs = np.empty(len(c2s), dtype=np.int64)
#    c2scmps = np.empty(len(c2s), dtype=np.int64)
#    tot = 0
#    for i, mat in enumerate(c2s):
#        c2sptrs[i] = tot
#        c2scmps[i] = len(mat)
#        tot += len(mat)
#    c2s = np.concatenate(c2s)
#
#    print(expnts)
#    print(expnts[-1])
#
#    tmp = _test(compnt, nprims, npntrs, centers, atoms, x, y, z, expnts, ds, lvalues)
#    print(tmp)
#    tmp = _evaluate_basis(fields, c2s, c2sptrs, c2scmps, centers, compnt,
#                          lvalues, nprims, npntrs, x, y, z, expnts, ds, atoms)
#    print(tmp)
#
#    return c2s, c2sptrs, c2scmps
#    #tmp = _evaluate_mos(fields, x, y, z, momatrix, vectors, atoms, centers,
#    #                    car2sph, car2sphptrs, car2sphcmps, compnt, nprims, npntrs, expnts, ds)
#    #print(tmp)
#
#def _test(compnt, nprims, npntrs, centers, atoms, x, y, z, expnts, ds, lvalues):
#    cnt = 0
#    xa = np.empty(len(x), dtype=np.float64)
#    ya = np.empty(len(x), dtype=np.float64)
#    za = np.empty(len(x), dtype=np.float64)
#    r2 = np.empty(len(x), dtype=np.float64)
#    ex = np.empty(len(x), dtype=np.float64)
#    t1, t2, t3 = 0., 0., 0.
#    a = 0.
#    d = 0.
#    for shl, mldegen in enumerate(compnt):
#        npntr = npntrs[shl]
#        nprim = nprims[shl]
#        for ml in range(mldegen):
#            l = lvalues[cnt]
#            t1 = atoms[centers[cnt],0]
#            t2 = atoms[centers[cnt],1]
#            t3 = atoms[centers[cnt],2]
#            xa = x - t1
#            ya = y - t2
#            za = z - t3
#            r2 = xa * xa + ya * ya + za * za
#            for k in range(nprim):
#                a = expnts[npntr+k]
#                d = ds[l,npntr+k]
#                ex = a * r2
#            cnt += 1
#    return a, d
#
#def _evaluate_basis(fields, c2s, c2sptrs, c2scmps, centers, compnt, lvalues,
#                    nprims, npntrs, x, y, z, expnts, ds, atoms):
#    cnt = 0
#    #d, px, py, pz = 0., 0, 0, 0
#    xa = np.empty(len(x), dtype=np.float64)
#    ya = np.empty(len(x), dtype=np.float64)
#    za = np.empty(len(x), dtype=np.float64)
#    r2 = np.empty(len(x), dtype=np.float64)
#    pe = np.empty(len(x), dtype=np.float64)
#    ex = np.empty(len(x), dtype=np.float64)
#    to = np.empty(len(x), dtype=np.float64)
#    a0, a1, a2, a, d = 0., 0., 0., 0., 0.
#    e = 2.718281828459045
#    for shl, mldegen in enumerate(compnt):
#        npntr = npntrs[shl]
#        nprim = nprims[shl]
#        for ml in range(mldegen):
#            ptr = c2sptrs[cnt]
#            l = lvalues[cnt]
#            a0 = atoms[centers[cnt],0]
#            a1 = atoms[centers[cnt],1]
#            a2 = atoms[centers[cnt],2]
#            xa = x - a0
#            ya = y - a1
#            za = z - a2
#            r2 = xa * xa + ya * ya + za * za
#            for j in range(c2scmps[cnt]):
#                c = c2s[ptr+j,0]
#                px = c2s[ptr+j,1]
#                py = c2s[ptr+j,2]
#                pz = c2s[ptr+j,3]
#                pe = c * xa ** px * ya ** py * za ** pz
#                for k in range(nprim):
#                    a = expnts[npntr+k]
#                    d = ds[l,npntr+k]
#                    ex = d * e ** (-a * r2)
#                #fields[:,cnt] = x
#                #tot2 += c2s[ptr+j,0]
#                #tot2 += c2s[ptr+j,1]
#                #tot2 += c2s[ptr+j,2]
#                #tot2 += c2s[ptr+j,3]
#            cnt += 1
#    #return xa, ya, za
#    return xa, ya, za
##    return fields
#
#def _evaluate_mos(fields, x, y, z, momatrix, vectors, atoms, centers,
#                  car2sph, car2sphptrs, car2sphcmps, compnt, nprims, npntrs, expnts, ds):
#    cnt = 0
#    tot = 0.
#    nbas = momatrix.shape[0]
#    bfns = np.zeros((len(x), nbas), dtype=np.float64)
#    nshell = len(compnt)
#    for shl, mldegen in enumerate(compnt):
#        nprim = nprims[shl]
#        pntr = npntrs[shl]
#        for ml in range(mldegen):
#            carptr = car2sphptrs[cnt]
#            carcmp = car2sphcmps[cnt]
#            xa = x - atoms[centers[shl], 0]
#            ya = y - atoms[centers[shl], 1]
#            za = z - atoms[centers[shl], 2]
#            r2 = xa * xa + ya * ya + za * za
#            for car in range(carcmp):
#                tot += car2sph[carptr+car,0]
#                tot += car2sph[carptr+car,1]
#                tot += car2sph[carptr+car,2]
#                tot += car2sph[carptr+car,3]
#            cnt += 1
#    return tot
#    #  subroutine driver(mos,xlo,xhi,nx,ylo,yhi,ny,zlo,zhi,nz,
#    # &                  nvectors,vectors,nbas,momat,
#    # &                  natoms,atompos,centers,labels,
#    # &                  nshell,ncomp,nprim,nptr,
#    # &                  nexp,ntypes,alphas,coefs)
#
#def _obtain_arrays(universe):
#    shells = universe.basis_set.functions_by_shell()
#    shells = shells.groupby(shells.index.get_level_values(0)).apply(lambda x: x.sum())
#    nshell = universe.atom['set'].map(shells).sum()
#    nexpnt = (universe.basis_set[np.abs(universe.basis_set['d']) > 0].groupby('set').apply(
#                                 lambda x: x.shape[0]) * universe.atom['set'].value_counts()).sum()
#    centers = universe.basis_set_order['center'].values
#    lvalues = np.array(universe.basis_set_order['L'].values)
#    if universe.basis_set.spherical:
#        mlvalues = universe.basis_set_order['ml'].values
#        lml_count = spher_lml_count
#    else:
#        ivalues = universe.basis_set_order['l'].values
#        jvalues = universe.basis_set_order['m'].values
#        kvalues = universe.basis_set_order['n'].values
#        lml_count = cart_lml_count
#    compnt = np.empty(nshell, dtype=np.int64)
#    nprims = np.empty(nshell, dtype=np.int64)
#    npntrs = np.empty(nshell, dtype=np.int64)
#    expnts = np.empty(nexpnt, dtype=np.float64)
#    lmax = universe.basis_set['L'].cat.as_ordered().max()
#    ds = np.empty((lmax + 1, nexpnt), dtype=np.float64)
#    bases = universe.basis_set[np.abs(universe.basis_set['d']) > 0].groupby('set')
#    cnt, ptr, xpc = 0, 1, 0
#    for seht, center in zip(universe.atom['set'], universe.atom.index):
#        b = bases.get_group(seht)
#        for sh, grp in b.groupby('shell'):
#            if len(grp) == 0: continue
#            compnt[cnt] = lml_count[grp['L'].values[0]]
#            nprims[cnt] = grp.shape[0]
#            npntrs[cnt] = ptr
#            ptr += nprims[cnt]
#            cnt += 1
#        for l, d, exp in zip(b['L'], b['d'], b['alpha']):
#            expnts[xpc] = exp
#            for i, ang in enumerate(ds):
#                ds[i][xpc] = d if i == l else 0.
#            xpc += 1
#    if universe.basis_set.spherical:
#        return centers, lvalues, mlvalues, compnt, nprims, npntrs, expnts, ds
#    else:
#        return centers, lvalues, ivalues, jvalues, kvalues, compnt, nprims, npntrs, expnts, ds
#
#def _determine_params(universe, params):
#    if params is None:
#        dr = 51
#        rmin = universe.atom[['x', 'y', 'z']].min().min()
#        rmax = universe.atom[['x', 'y', 'z']].max().max()
#        return num_grid(rmin, rmax, dr)
#    if len(params) % 3 == 0:
#        return num_grid(*params)
#    raise Exception('params must have len divisible by 3')
#
#def _determine_mocoefs(universe, vectors):
#    raise NotImplementedError('I will get to it.')
#
#def _determine_vectors(universe, vectors):
#    if isinstance(vectors, int):
#        return np.array([vectors])
#    if isinstance(vectors, range):
#        return np.array(vectors)
#    if isinstance(vectors, (list, tuple)):
#        try:
#            t = len(vectors[0])
#            cols = [i[0] for i in vectors]
#            if not all(col in universe.momatrix.columns for col in cols):
#                raise Exception('vectors column must be in universe.momatrix')
#            return _determine_mocoefs(universe, vectors)
#        except:
#            raise Exception('vectors must contain (column, vector) pairs')
#    if vectors is None:
#        total = universe.momatrix['orbital'].max()
#        if total < 20:
#            return np.array(range(total))
#        try:
#            try:
#                na = universe.frame['alpha_electrons'].ix[0]
#                nb = universe.frame['beta_electrons'].ix[0]
#                nclosed = (na + nb) // 2
#            except KeyError:
#                nclosed = universe.frame['total_electrons'].ix[0] // 2
#        except KeyError:
#            try:
#                nclosed = universe.atom['Zeff'].sum() // 2
#            except KeyError:
#                nclosed = universe.atom['Z'].sum() // 2
#        if nclosed - 15 < 0:
#            return np.array(range(nclosed + 15))
#        else:
#            return np.array(range(nclosed - 15, nclosed + 5))
#
#
#def num_grid(xmin, xmax, nx, ymin=0., ymax=0., ny=0, zmin=0., zmax=0., nz=0):
#    ny = nx if ny == 0 else ny
#    ymin = xmin if ymin == 0. else ymin
#    ymax = xmax if ymax == 0. else ymax
#    nz = nx if nz == 0 else nz
#    zmin = xmin if zmin == 0. else zmin
#    zmax = xmax if zmax == 0. else zmax
#    dx = (xmax - xmin) / nx
#    dy = (ymax - ymin) / ny
#    dz = (zmax - zmin) / nz
#    xt = np.empty(nx, dtype=np.float64)
#    yt = np.empty(ny, dtype=np.float64)
#    zt = np.empty(nz, dtype=np.float64)
#    for i in range(nx):
#        xt[i] = xmin + i * dx
#    for i in range(ny):
#        yt[i] = ymin + i * dy
#    for i in range(nz):
#        zt[i] = zmin + i * dz
#    tot = nx * ny * nz
#    x = np.empty(tot, dtype=np.float64)
#    y = np.empty(tot, dtype=np.float64)
#    z = np.empty(tot, dtype=np.float64)
#    c = 0
#    for i in range(nx):
#        for j in range(ny):
#            for k in range(nz):
#                x[c] = xt[i]
#                y[c] = yt[j]
#                z[c] = zt[k]
#                c += 1
#    return x, y, z
#
#def cartesian(l, ncart):
#    ret = np.zeros((ncart, 4), dtype=np.float64)
#    if l == 0:
#        ret[0,:] = (1.0, 0.0, 0.0, 0.0)
#        return ret
#    elif l == 1:
#        ret[0 ,:] = (1., 1.0, 0.0, 0.0)
#        ret[1 ,:] = (1., 0.0, 1.0, 0.0)
#        ret[2 ,:] = (1., 0.0, 0.0, 1.0)
#        return ret
#    elif l == 2:
#        ret[0 ,:] = (1., 2.0, 0.0, 0.0)
#        ret[1 ,:] = (1., 1.0, 1.0, 0.0)
#        ret[2 ,:] = (1., 1.0, 0.0, 1.0)
#        ret[3 ,:] = (1., 0.0, 2.0, 0.0)
#        ret[4 ,:] = (1., 0.0, 1.0, 1.0)
#        ret[5 ,:] = (1., 0.0, 0.0, 2.0)
#        return ret
#    elif l == 3:
#        ret[0 ,:] = (1., 3.0, 0.0, 0.0)
#        ret[1 ,:] = (1., 2.0, 1.0, 0.0)
#        ret[2 ,:] = (1., 2.0, 0.0, 1.0)
#        ret[3 ,:] = (1., 1.0, 2.0, 0.0)
#        ret[4 ,:] = (1., 1.0, 1.0, 1.0)
#        ret[5 ,:] = (1., 1.0, 0.0, 2.0)
#        ret[6 ,:] = (1., 0.0, 3.0, 0.0)
#        ret[7 ,:] = (1., 0.0, 2.0, 1.0)
#        ret[8 ,:] = (1., 0.0, 1.0, 2.0)
#        ret[9 ,:] = (1., 0.0, 0.0, 3.0)
#        return ret
#    elif l == 4:
#        ret[0 ,:] = (1., 4.0, 0.0, 0.0)
#        ret[1 ,:] = (1., 3.0, 1.0, 0.0)
#        ret[2 ,:] = (1., 3.0, 0.0, 1.0)
#        ret[3 ,:] = (1., 2.0, 2.0, 0.0)
#        ret[4 ,:] = (1., 2.0, 1.0, 1.0)
#        ret[5 ,:] = (1., 2.0, 0.0, 2.0)
#        ret[6 ,:] = (1., 1.0, 3.0, 0.0)
#        ret[7 ,:] = (1., 1.0, 2.0, 1.0)
#        ret[8 ,:] = (1., 1.0, 1.0, 2.0)
#        ret[9 ,:] = (1., 1.0, 0.0, 3.0)
#        ret[10,:] = (1., 0.0, 4.0, 0.0)
#        ret[11,:] = (1., 0.0, 3.0, 1.0)
#        ret[12,:] = (1., 0.0, 2.0, 2.0)
#        ret[13,:] = (1., 0.0, 1.0, 3.0)
#        ret[14,:] = (1., 0.0, 0.0, 4.0)
#        return ret
#    elif l == 5:
#        ret[0 ,:] = (1., 5.0, 0.0, 0.0)
#        ret[1 ,:] = (1., 4.0, 1.0, 0.0)
#        ret[2 ,:] = (1., 4.0, 0.0, 1.0)
#        ret[3 ,:] = (1., 3.0, 2.0, 0.0)
#        ret[4 ,:] = (1., 3.0, 1.0, 1.0)
#        ret[5 ,:] = (1., 3.0, 0.0, 2.0)
#        ret[6 ,:] = (1., 2.0, 3.0, 0.0)
#        ret[7 ,:] = (1., 2.0, 2.0, 1.0)
#        ret[8 ,:] = (1., 2.0, 1.0, 2.0)
#        ret[9 ,:] = (1., 2.0, 0.0, 3.0)
#        ret[10,:] = (1., 1.0, 4.0, 0.0)
#        ret[11,:] = (1., 1.0, 3.0, 1.0)
#        ret[12,:] = (1., 1.0, 2.0, 2.0)
#        ret[13,:] = (1., 1.0, 1.0, 3.0)
#        ret[14,:] = (1., 1.0, 0.0, 4.0)
#        ret[15,:] = (1., 0.0, 5.0, 0.0)
#        ret[16,:] = (1., 0.0, 4.0, 1.0)
#        ret[17,:] = (1., 0.0, 3.0, 2.0)
#        ret[18,:] = (1., 0.0, 2.0, 3.0)
#        ret[19,:] = (1., 0.0, 1.0, 4.0)
#        ret[20,:] = (1., 0.0, 0.0, 5.0)
#        return ret
#    elif l == 6:
#        ret[0 ,:] = (1., 6.0, 0.0, 0.0)
#        ret[1 ,:] = (1., 5.0, 1.0, 0.0)
#        ret[2 ,:] = (1., 5.0, 0.0, 1.0)
#        ret[3 ,:] = (1., 4.0, 2.0, 0.0)
#        ret[4 ,:] = (1., 4.0, 1.0, 1.0)
#        ret[5 ,:] = (1., 4.0, 0.0, 2.0)
#        ret[6 ,:] = (1., 3.0, 3.0, 0.0)
#        ret[7 ,:] = (1., 3.0, 2.0, 1.0)
#        ret[8 ,:] = (1., 3.0, 1.0, 2.0)
#        ret[9 ,:] = (1., 3.0, 0.0, 3.0)
#        ret[10,:] = (1., 2.0, 4.0, 0.0)
#        ret[11,:] = (1., 2.0, 3.0, 1.0)
#        ret[12,:] = (1., 2.0, 2.0, 2.0)
#        ret[13,:] = (1., 2.0, 1.0, 3.0)
#        ret[14,:] = (1., 2.0, 0.0, 4.0)
#        ret[15,:] = (1., 1.0, 5.0, 0.0)
#        ret[16,:] = (1., 1.0, 4.0, 1.0)
#        ret[17,:] = (1., 1.0, 3.0, 2.0)
#        ret[18,:] = (1., 1.0, 2.0, 3.0)
#        ret[19,:] = (1., 1.0, 1.0, 4.0)
#        ret[20,:] = (1., 1.0, 0.0, 5.0)
#        ret[21,:] = (1., 0.0, 6.0, 0.0)
#        ret[22,:] = (1., 0.0, 5.0, 1.0)
#        ret[23,:] = (1., 0.0, 4.0, 2.0)
#        ret[24,:] = (1., 0.0, 3.0, 3.0)
#        ret[25,:] = (1., 0.0, 2.0, 4.0)
#        ret[26,:] = (1., 0.0, 1.0, 5.0)
#        ret[27,:] = (1., 0.0, 0.0, 6.0)
#        return ret
#
#def spherical(l, ml):
#    if l == 0:
#        ncart = 1
#        ret = np.zeros((ncart, 4), dtype=np.float64)
#        ret[0,:] = (1.0, 0.0, 0.0, 0.0)
#        return ret
#    elif l == 1:
#        if ml == -1:
#            ncart = 1
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = ( 1.00000000, 0.0, 1.0, 0.0)
#            return ret
#        elif ml == 0:
#            ncart = 1
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = ( 1.00000000, 0.0, 0.0, 1.0)
#            return ret
#        elif ml == 1:
#            ncart = 1
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = ( 1.00000000, 1.0, 0.0, 0.0)
#            return ret
#    elif l == 2:
#        if ml == -2:
#            ncart = 1
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = ( 1.73205081, 1.0, 1.0, 0.0)
#            return ret
#        elif ml == -1:
#            ncart = 1
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = ( 1.73205081, 0.0, 1.0, 1.0)
#            return ret
#        elif ml == 0:
#            ncart = 3
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = (-0.50000000, 2.0, 0.0, 0.0)
#            ret[1,:] = (-0.50000000, 0.0, 2.0, 0.0)
#            ret[2,:] = ( 1.00000000, 0.0, 0.0, 2.0)
#            return ret
#        elif ml == 1:
#            ncart = 1
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = ( 1.73205081, 1.0, 0.0, 1.0)
#            return ret
#        elif ml == 2:
#            ncart = 2
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = ( 0.86602540, 2.0, 0.0, 0.0)
#            ret[1,:] = (-0.86602540, 0.0, 2.0, 0.0)
#            return ret
#    elif l == 3:
#        if ml == -3:
#            ncart = 2
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = ( 2.37170825, 2.0, 1.0, 0.0)
#            ret[1,:] = (-0.79056942, 0.0, 3.0, 0.0)
#            return ret
#        elif ml == -2:
#            ncart = 1
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = ( 3.87298335, 1.0, 1.0, 1.0)
#            return ret
#        elif ml == -1:
#            ncart = 3
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = (-0.61237244, 2.0, 1.0, 0.0)
#            ret[1,:] = (-0.61237244, 0.0, 3.0, 0.0)
#            ret[2,:] = ( 2.44948974, 0.0, 1.0, 2.0)
#            return ret
#        elif ml == 0:
#            ncart = 3
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = (-1.50000000, 2.0, 0.0, 1.0)
#            ret[1,:] = (-1.50000000, 0.0, 2.0, 1.0)
#            ret[2,:] = ( 1.00000000, 0.0, 0.0, 3.0)
#            return ret
#        elif ml == 1:
#            ncart = 3
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = (-0.61237244, 3.0, 0.0, 0.0)
#            ret[1,:] = (-0.61237244, 1.0, 2.0, 0.0)
#            ret[2,:] = ( 2.44948974, 1.0, 0.0, 2.0)
#            return ret
#        elif ml == 2:
#            ncart = 2
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = ( 1.93649167, 2.0, 0.0, 1.0)
#            ret[1,:] = (-1.93649167, 0.0, 2.0, 1.0)
#            return ret
#        elif ml == 3:
#            ncart = 2
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = ( 0.79056942, 3.0, 0.0, 0.0)
#            ret[1,:] = (-2.37170825, 1.0, 2.0, 0.0)
#            return ret
#    elif l == 4:
#        if ml == -4:
#            ncart = 2
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = ( 2.95803989, 3.0, 1.0, 0.0)
#            ret[1,:] = (-2.95803989, 1.0, 3.0, 0.0)
#            return ret
#        elif ml == -3:
#            ncart = 2
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = ( 6.27495020, 2.0, 1.0, 1.0)
#            ret[1,:] = (-2.09165007, 0.0, 3.0, 1.0)
#            return ret
#        elif ml == -2:
#            ncart = 3
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = (-1.11803399, 3.0, 1.0, 0.0)
#            ret[1,:] = (-1.11803399, 1.0, 3.0, 0.0)
#            ret[2,:] = ( 6.70820393, 1.0, 1.0, 2.0)
#            return ret
#        elif ml == -1:
#            ncart = 3
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = (-2.37170825, 2.0, 1.0, 1.0)
#            ret[1,:] = (-2.37170825, 0.0, 3.0, 1.0)
#            ret[2,:] = ( 3.16227766, 0.0, 1.0, 3.0)
#            return ret
#        elif ml == 0:
#            ncart = 6
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = ( 0.37500000, 4.0, 0.0, 0.0)
#            ret[1,:] = ( 0.75000000, 2.0, 2.0, 0.0)
#            ret[2,:] = (-3.00000000, 2.0, 0.0, 2.0)
#            ret[3,:] = ( 0.37500000, 0.0, 4.0, 0.0)
#            ret[4,:] = (-3.00000000, 0.0, 2.0, 2.0)
#            ret[5,:] = ( 1.00000000, 0.0, 0.0, 4.0)
#            return ret
#        elif ml == 1:
#            ncart = 3
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = (-2.37170825, 3.0, 0.0, 1.0)
#            ret[1,:] = (-2.37170825, 1.0, 2.0, 1.0)
#            ret[2,:] = ( 3.16227766, 1.0, 0.0, 3.0)
#            return ret
#        elif ml == 2:
#            ncart = 4
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = (-0.55901699, 4.0, 0.0, 0.0)
#            ret[1,:] = ( 3.35410197, 2.0, 0.0, 2.0)
#            ret[2,:] = ( 0.55901699, 0.0, 4.0, 0.0)
#            ret[3,:] = (-3.35410197, 0.0, 2.0, 2.0)
#            return ret
#        elif ml == 3:
#            ncart = 2
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = ( 2.09165007, 3.0, 0.0, 1.0)
#            ret[1,:] = (-6.27495020, 1.0, 2.0, 1.0)
#            return ret
#        elif ml == 4:
#            ncart = 3
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = ( 0.73950997, 4.0, 0.0, 0.0)
#            ret[1,:] = (-4.43705984, 2.0, 2.0, 0.0)
#            ret[2,:] = ( 0.73950997, 0.0, 4.0, 0.0)
#            return ret
#    elif l == 5:
#        if ml == -5:
#            ncart = 3
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = ( 3.50780380, 4.0, 1.0, 0.0)
#            ret[1,:] = (-7.01560760, 2.0, 3.0, 0.0)
#            ret[2,:] = ( 0.70156076, 0.0, 5.0, 0.0)
#            return ret
#        elif ml == -4:
#            ncart = 2
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = ( 8.87411967, 3.0, 1.0, 1.0)
#            ret[1,:] = (-8.87411967, 1.0, 3.0, 1.0)
#            return ret
#        elif ml == -3:
#            ncart = 5
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = (-1.56873755, 4.0, 1.0, 0.0)
#            ret[1,:] = (-1.04582503, 2.0, 3.0, 0.0)
#            ret[2,:] = ( 12.54990040, 2.0, 1.0, 2.0)
#            ret[3,:] = ( 0.52291252, 0.0, 5.0, 0.0)
#            ret[4,:] = (-4.18330013, 0.0, 3.0, 2.0)
#            return ret
#        elif ml == -2:
#            ncart = 3
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = (-5.12347538, 3.0, 1.0, 1.0)
#            ret[1,:] = (-5.12347538, 1.0, 3.0, 1.0)
#            ret[2,:] = ( 10.24695077, 1.0, 1.0, 3.0)
#            return ret
#        elif ml == -1:
#            ncart = 6
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = ( 0.48412292, 4.0, 1.0, 0.0)
#            ret[1,:] = ( 0.96824584, 2.0, 3.0, 0.0)
#            ret[2,:] = (-5.80947502, 2.0, 1.0, 2.0)
#            ret[3,:] = ( 0.48412292, 0.0, 5.0, 0.0)
#            ret[4,:] = (-5.80947502, 0.0, 3.0, 2.0)
#            ret[5,:] = ( 3.87298335, 0.0, 1.0, 4.0)
#            return ret
#        elif ml == 0:
#            ncart = 6
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = ( 1.87500000, 4.0, 0.0, 1.0)
#            ret[1,:] = ( 3.75000000, 2.0, 2.0, 1.0)
#            ret[2,:] = (-5.00000000, 2.0, 0.0, 3.0)
#            ret[3,:] = ( 1.87500000, 0.0, 4.0, 1.0)
#            ret[4,:] = (-5.00000000, 0.0, 2.0, 3.0)
#            ret[5,:] = ( 1.00000000, 0.0, 0.0, 5.0)
#            return ret
#        elif ml == 1:
#            ncart = 6
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = ( 0.48412292, 5.0, 0.0, 0.0)
#            ret[1,:] = ( 0.96824584, 3.0, 2.0, 0.0)
#            ret[2,:] = (-5.80947502, 3.0, 0.0, 2.0)
#            ret[3,:] = ( 0.48412292, 1.0, 4.0, 0.0)
#            ret[4,:] = (-5.80947502, 1.0, 2.0, 2.0)
#            ret[5,:] = ( 3.87298335, 1.0, 0.0, 4.0)
#            return ret
#        elif ml == 2:
#            ncart = 4
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = (-2.56173769, 4.0, 0.0, 1.0)
#            ret[1,:] = ( 5.12347538, 2.0, 0.0, 3.0)
#            ret[2,:] = ( 2.56173769, 0.0, 4.0, 1.0)
#            ret[3,:] = (-5.12347538, 0.0, 2.0, 3.0)
#            return ret
#        elif ml == 3:
#            ncart = 5
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = (-0.52291252, 5.0, 0.0, 0.0)
#            ret[1,:] = ( 1.04582503, 3.0, 2.0, 0.0)
#            ret[2,:] = ( 4.18330013, 3.0, 0.0, 2.0)
#            ret[3,:] = ( 1.56873755, 1.0, 4.0, 0.0)
#            ret[4,:] = (-12.54990040, 1.0, 2.0, 2.0)
#            return ret
#        elif ml == 4:
#            ncart = 3
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = ( 2.21852992, 4.0, 0.0, 1.0)
#            ret[1,:] = (-13.31117951, 2.0, 2.0, 1.0)
#            ret[2,:] = ( 2.21852992, 0.0, 4.0, 1.0)
#            return ret
#        elif ml == 5:
#            ncart = 3
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = ( 0.70156076, 5.0, 0.0, 0.0)
#            ret[1,:] = (-7.01560760, 3.0, 2.0, 0.0)
#            ret[2,:] = ( 3.50780380, 1.0, 4.0, 0.0)
#            return ret
#    elif l == 6:
#        if ml == -6:
#            ncart = 3
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = ( 4.03015974, 5.0, 1.0, 0.0)
#            ret[1,:] = (-13.43386579, 3.0, 3.0, 0.0)
#            ret[2,:] = ( 4.03015974, 1.0, 5.0, 0.0)
#            return ret
#        elif ml == -5:
#            ncart = 3
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = ( 11.63406904, 4.0, 1.0, 1.0)
#            ret[1,:] = (-23.26813809, 2.0, 3.0, 1.0)
#            ret[2,:] = ( 2.32681381, 0.0, 5.0, 1.0)
#            return ret
#        elif ml == -4:
#            ncart = 4
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = (-1.98431348, 5.0, 1.0, 0.0)
#            ret[1,:] = ( 19.84313483, 3.0, 1.0, 2.0)
#            ret[2,:] = ( 1.98431348, 1.0, 5.0, 0.0)
#            ret[3,:] = (-19.84313483, 1.0, 3.0, 2.0)
#            return ret
#        elif ml == -3:
#            ncart = 5
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = (-8.15139942, 4.0, 1.0, 1.0)
#            ret[1,:] = (-5.43426628, 2.0, 3.0, 1.0)
#            ret[2,:] = ( 21.73706512, 2.0, 1.0, 3.0)
#            ret[3,:] = ( 2.71713314, 0.0, 5.0, 1.0)
#            ret[4,:] = (-7.24568837, 0.0, 3.0, 3.0)
#            return ret
#        elif ml == -2:
#            ncart = 6
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = ( 0.90571105, 5.0, 1.0, 0.0)
#            ret[1,:] = ( 1.81142209, 3.0, 3.0, 0.0)
#            ret[2,:] = (-14.49137675, 3.0, 1.0, 2.0)
#            ret[3,:] = ( 0.90571105, 1.0, 5.0, 0.0)
#            ret[4,:] = (-14.49137675, 1.0, 3.0, 2.0)
#            ret[5,:] = ( 14.49137675, 1.0, 1.0, 4.0)
#            return ret
#        elif ml == -1:
#            ncart = 6
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = ( 2.86410981, 4.0, 1.0, 1.0)
#            ret[1,:] = ( 5.72821962, 2.0, 3.0, 1.0)
#            ret[2,:] = (-11.45643924, 2.0, 1.0, 3.0)
#            ret[3,:] = ( 2.86410981, 0.0, 5.0, 1.0)
#            ret[4,:] = (-11.45643924, 0.0, 3.0, 3.0)
#            ret[5,:] = ( 4.58257569, 0.0, 1.0, 5.0)
#            return ret
#        elif ml == 0:
#            ncart = 10
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = (-0.31250000, 6.0, 0.0, 0.0)
#            ret[1,:] = (-0.93750000, 4.0, 2.0, 0.0)
#            ret[2,:] = ( 5.62500000, 4.0, 0.0, 2.0)
#            ret[3,:] = (-0.93750000, 2.0, 4.0, 0.0)
#            ret[4,:] = ( 11.25000000, 2.0, 2.0, 2.0)
#            ret[5,:] = (-7.50000000, 2.0, 0.0, 4.0)
#            ret[6,:] = (-0.31250000, 0.0, 6.0, 0.0)
#            ret[7,:] = ( 5.62500000, 0.0, 4.0, 2.0)
#            ret[8,:] = (-7.50000000, 0.0, 2.0, 4.0)
#            ret[9,:] = ( 1.00000000, 0.0, 0.0, 6.0)
#            return ret
#        elif ml == 1:
#            ncart = 6
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = ( 2.86410981, 5.0, 0.0, 1.0)
#            ret[1,:] = ( 5.72821962, 3.0, 2.0, 1.0)
#            ret[2,:] = (-11.45643924, 3.0, 0.0, 3.0)
#            ret[3,:] = ( 2.86410981, 1.0, 4.0, 1.0)
#            ret[4,:] = (-11.45643924, 1.0, 2.0, 3.0)
#            ret[5,:] = ( 4.58257569, 1.0, 0.0, 5.0)
#            return ret
#        elif ml == 2:
#            ncart = 8
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = ( 0.45285552, 6.0, 0.0, 0.0)
#            ret[1,:] = ( 0.45285552, 4.0, 2.0, 0.0)
#            ret[2,:] = (-7.24568837, 4.0, 0.0, 2.0)
#            ret[3,:] = (-0.45285552, 2.0, 4.0, 0.0)
#            ret[4,:] = ( 7.24568837, 2.0, 0.0, 4.0)
#            ret[5,:] = (-0.45285552, 0.0, 6.0, 0.0)
#            ret[6,:] = ( 7.24568837, 0.0, 4.0, 2.0)
#            ret[7,:] = (-7.24568837, 0.0, 2.0, 4.0)
#            return ret
#        elif ml == 3:
#            ncart = 5
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = (-2.71713314, 5.0, 0.0, 1.0)
#            ret[1,:] = ( 5.43426628, 3.0, 2.0, 1.0)
#            ret[2,:] = ( 7.24568837, 3.0, 0.0, 3.0)
#            ret[3,:] = ( 8.15139942, 1.0, 4.0, 1.0)
#            ret[4,:] = (-21.73706512, 1.0, 2.0, 3.0)
#            return ret
#        elif ml == 4:
#            ncart = 7
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = (-0.49607837, 6.0, 0.0, 0.0)
#            ret[1,:] = ( 2.48039185, 4.0, 2.0, 0.0)
#            ret[2,:] = ( 4.96078371, 4.0, 0.0, 2.0)
#            ret[3,:] = ( 2.48039185, 2.0, 4.0, 0.0)
#            ret[4,:] = (-29.76470225, 2.0, 2.0, 2.0)
#            ret[5,:] = (-0.49607837, 0.0, 6.0, 0.0)
#            ret[6,:] = ( 4.96078371, 0.0, 4.0, 2.0)
#            return ret
#        elif ml == 5:
#            ncart = 3
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = ( 2.32681381, 5.0, 0.0, 1.0)
#            ret[1,:] = (-23.26813809, 3.0, 2.0, 1.0)
#            ret[2,:] = ( 11.63406904, 1.0, 4.0, 1.0)
#            return ret
#        elif ml == 6:
#            ncart = 4
#            ret = np.zeros((ncart, 4), dtype=np.float64)
#            ret[0,:] = ( 0.67169329, 6.0, 0.0, 0.0)
#            ret[1,:] = (-10.07539934, 4.0, 2.0, 0.0)
#            ret[2,:] = ( 10.07539934, 2.0, 4.0, 0.0)
#            ret[3,:] = (-0.67169329, 0.0, 6.0, 0.0)
#            return ret
#
#
#if config['dynamic']['numba'] == 'true':
#    from numba import jit, vectorize
#    cartesian = jit(nopython=True, cache=True, nogil=True)(cartesian)
#    spherical = jit(nopython=True, cache=True, nogil=True)(spherical)
#    num_grid = jit(nopython=True, cache=True, nogil=True)(num_grid)
#    _test = jit(nopython=True, cache=True, nogil=True)(_test)
#    _evaluate_mos = jit(nopython=True, cache=True, nogil=True)(_evaluate_mos)
#    _evaluate_basis = jit(nopython=True, cache=True, nogil=True)(_evaluate_basis)
