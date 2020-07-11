# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Atomic Position Data
############################
This module provides a collection of dataframes supporting nuclear positions,
forces, velocities, symbols, etc. (all data associated with atoms as points).
"""
from numbers import Integral
import numpy as np
import pandas as pd
from exa import DataFrame, Series
from exa.util.units import Length
from exatomic.base import sym2z, sym2mass
from exatomic.algorithms.distance import modv
from exatomic.core.error import PeriodicUniverseError
from exatomic.algorithms.geometry import make_small_molecule
from exatomic import plotter


class Atom(DataFrame):
    """
    The atom dataframe.

    +-------------------+----------+-------------------------------------------+
    | Column            | Type     | Description                               |
    +===================+==========+===========================================+
    | x                 | float    | position in x (req.)                      |
    +-------------------+----------+-------------------------------------------+
    | y                 | float    | position in y (req.)                      |
    +-------------------+----------+-------------------------------------------+
    | z                 | float    | position in z (req.)                      |
    +-------------------+----------+-------------------------------------------+
    | frame             | category | non-unique integer (req.)                 |
    +-------------------+----------+-------------------------------------------+
    | symbol            | category | element symbol (req.)                     |
    +-------------------+----------+-------------------------------------------+
    | fx                | float    | force in x                                |
    +-------------------+----------+-------------------------------------------+
    | fy                | float    | force in y                                |
    +-------------------+----------+-------------------------------------------+
    | fz                | float    | force in z                                |
    +-------------------+----------+-------------------------------------------+
    | vx                | float    | velocity in x                             |
    +-------------------+----------+-------------------------------------------+
    | vy                | float    | velocity in y                             |
    +-------------------+----------+-------------------------------------------+
    | vz                | float    | velocity in z                             |
    +-------------------+----------+-------------------------------------------+
    """
    _index = 'atom'
    _cardinal = ('frame', np.int64)
    _categories = {'symbol': str, 'set': np.int64, 'molecule': np.int64,
                   'label': np.int64}
    _columns = ['x', 'y', 'z', 'symbol']

    #@property
    #def _constructor(self):
    #    return Atom

    @property
    def nframes(self):
        """Return the total number of frames in the atom table."""
        return np.int64(self.frame.cat.as_ordered().max() + 1)

    @property
    def last_frame(self):
        """Return the last frame of the atom table."""
        return self[self.frame == self.nframes - 1]

    @property
    def unique_atoms(self):
        """Return unique atom symbols of the last frame."""
        return self.last_frame.symbol.unique()

    @staticmethod
    def _determine_center(attr, coords):
        """Determine the center of the molecule with respect to
        the given attribute data. Used for the center of nuclear
        charge and center of mass."""
        center = 1/np.sum(attr)*np.sum(np.multiply(np.transpose(coords), attr), axis=1)
        center = pd.Series(center, index=['x', 'y', 'z'])
        return center

    def center(self, idx=None, frame=None, to=None):
        """
        Return a copy of a single frame of the atom table
        centered around a specific atom index. There is also
        the ability to center the molecule to the center of
        nuclear charge (NuclChrg) or center of mass (Mass).

        Args:
            idx (int): Atom index in the atom table
            frame (int): Frame to perform the operation on
            to (str): Tells the program which centering algorithm to use

        Returs:
            frame (:class:`exatomic.Universe.atom`): Atom frame
        """
        if frame is None: frame = self.last_frame.copy()
        else: frame = self[self.frame == frame].copy()
        if to is None:
            if idx is None: raise TypeError("Must provide an atom to center to")
            center = frame.iloc[idx]
        elif to == 'NuclChrg':
            try:
                Z = frame['Z'].astype(int).values
            except KeyError:
                Z = frame['symbol'].map(sym2z).astype(int).values
            center = self._determine_center(attr=Z, coords=frame[['x', 'y', 'z']].values)
        elif to == 'Mass':
            mass = frame['symbol'].map(sym2mass).astype(int).values
            center = self._determine_center(attr=mass, coords=frame[['x', 'y', 'z']].values)
        else:
            raise NotImplementedError("Sorry the centering option {} is not available".format(to))
        for r in ['x', 'y', 'z']:
            if center[r] > 0: frame[r] = frame[r] - center[r]
            else: frame[r] = frame[r] + np.abs(center[r])
        return Atom(frame)

    def rotate(self, theta, axis=None, frame=None, degrees=True):
        """
        Return a copy of a single frame of the atom table rotated
        around the specified rotation axis by the specified angle.
        As we have the rotation axis and the rotation angle we are
        able to use the Rodrigues' formula to get the rotated
        vectors.

        Args:
            theta (float): The angle that you wish to rotate by
            axis (list): The axis of rotation
            frame (int): The frame that you wish to rotate
            degrees (bool): If true convert from degrees to radians

        Returns:
            frame (:class:`exatomic.Universe.atom`): Atom frame
        """
        if axis is None: axis = [0, 0, 1]
        if frame is None: frame = self.last_frame.copy()
        else: frame = self[self.frame == frame].copy()

        if all(map(lambda x: x == 0., axis)) or theta == 0.: return frame
        # as we have the rotation axis and the angle we will rotate over
        # we implement the Rodrigues' rotation formula
        # v_rot = v*np.cos(theta) + (np.cross(k,v))*np.sin(theta) + k*(np.dot(k,v))*(1-np.cos(theta))

        # convert units if not degrees
        if degrees: theta = theta*np.pi/180.

        # normalize rotation axis vector
        norm = np.linalg.norm(axis)
        try:
            axis /= norm
        except ZeroDivisionError:
            raise ZeroDivisionError("Trying to normalize axis {} by a 0 value".format(axis))
        # get the coordinates
        coords = frame[['x', 'y', 'z']].values
        # generate the first term in rodrigues formula
        a = coords * np.cos(theta)
        # generate second term in rodrigures formula
        # this creates a matrix of size coords.shape[0]
        b = np.cross(axis, coords) * np.sin(theta)
        # generate the last term in rodrigues formula
        # we use np.outer to make a dyadic productof the result from the dot product vector
        # and the axis vector
        c = np.outer(np.dot(coords, axis), axis) * (1-np.cos(theta))
        rotated = a + b + c
        frame[['x', 'y', 'z']] = rotated
        return Atom(frame)

    def translate(self, dx=0, dy=0, dz=0, vector=None, frame=None, units='au'):
        """
        Return a copy of a single frame of the atom table translated by
        some specified distance.

        Note:
            Vector can be used instead of dx, dy, dz as it will be decomposed
            into those components. If vector and any of the others are
            specified the values in vector will be used.

        Args:
            dx (float): Displacement distance in x
            dy (float): Displacement distance in y
            dz (float): Displacement distance in z
            vector (list): Displacement vector
            units (str): Units that are used for the displacement

        Returns:
            frame (:class:`exatomic.Universe.atom`): Atom frame
        """
        if frame is None: frame = self.last_frame.copy()
        else: frame = self[self.frame == frame].copy()
        # check if vector is specified
        if vector is not None:
            # convert vector units to au
            vector = [i * Length[units, 'au'] for i in vector]
            dx = vector[0]
            dy = vector[1]
            dz = vector[2]
        # add the values to each respective coordinate
        frame['x'] += dx
        frame['y'] += dy
        frame['z'] += dz
        return Atom(frame)

    def align_to_axis(self, adx0, adx1, axis=None, frame=None, center_to=None):
        '''
        This a short method to center and align the molecule along some defined axis.

        Args:
            adx0 (int): Atom to place at the origin
            adx1 (int): Atom to align along the axis
            axis (list): Axis that the vector adx0-adx1 will align to
            frame (int): Frame to align

        Returns:
            aligned (:class:`exatomic.Universe.atom`): Aligned atom frame
        '''
        if frame is None: atom = self.last_frame.copy()
        else: atom = self[self.frame == frame].copy()
        cols = ['x', 'y', 'z']
        # define the original vector
        v0 = atom.iloc[adx1][cols].values.astype(np.float64) - atom.iloc[adx0][cols].values.astype(np.float64)
        # get the vector to align with and normalize
        v1 = axis/np.linalg.norm(axis)
        # find the normal vector to rotate around
        n = np.cross(v0, v1)
        # find the angle to rotate the vector
        theta = np.arccos(np.dot(v0, v1) / (np.linalg.norm(v0)*np.linalg.norm(v1)))
        # use the center method to center the molecule
        centered = Atom(atom).center(adx0, frame=frame, to=center_to)
        # rotate the molecule around the normal vector
        aligned = centered.rotate(theta=theta, axis=n, degrees=False)
        return Atom(aligned)

    def to_xyz(self, tag='symbol', header=False, comments='', columns=None,
               frame=None, units='Angstrom'):
        """
        Return atomic data in XYZ format, by default without the first 2 lines.
        If multiple frames are specified, return an XYZ trajectory format. If
        frame is not specified, by default returns the last frame in the table.

        Args:
            tag (str): column name to use in place of 'symbol'
            header (bool): if True, return the first 2 lines of XYZ format
            comment (str, list): comment(s) to put in the comment line
            frame (int, iter): frame or frames to return
            units (str): units (default angstroms)

        Returns:
            ret (str): XYZ formatted atomic data
        """
        # TODO :: this is conceptually a duplicate of XYZ.from_universe
        columns = (tag, 'x', 'y', 'z') if columns is None else columns
        frame = self.nframes - 1 if frame is None else frame
        if isinstance(frame, Integral): frame = [frame]
        if not isinstance(comments, list): comments = [comments]
        if len(comments) == 1: comments = comments * len(frame)
        df = self[self['frame'].isin(frame)].copy()
        if tag not in df.columns:
            if tag == 'Z':
                stoz = sym2z()
                df[tag] = df['symbol'].map(stoz)
        df['x'] *= Length['au', units]
        df['y'] *= Length['au', units]
        df['z'] *= Length['au', units]
        grps = df.groupby('frame')
        ret = ''
        formatter = {tag: '{:<5}'.format}
        stargs = {'columns': columns, 'header': False,
                  'index': False, 'formatters': formatter}
        t = 0
        for _, grp in grps:
            if not len(grp): continue
            tru = (header or comments[t] or len(frame) > 1)
            hdr = '\n'.join([str(len(grp)), comments[t], '']) if tru else ''
            ret = ''.join([ret, hdr, grp.to_string(**stargs), '\n'])
            t += 1
        return ret

    def get_element_masses(self):
        """Compute and return element masses from symbols."""
        return self['symbol'].astype('O').map(sym2mass)

    def get_atom_labels(self):
        """
        Compute and return enumerated atoms.

        Returns:
            labels (:class:`~exa.core.numerical.Series`): Enumerated atom labels (of type int)
        """
        nats = self.cardinal_groupby().size().values
        labels = Series([i for nat in nats for i in range(nat)], dtype='category')
        labels.index = self.index
        return labels

    @classmethod
    def from_small_molecule_data(cls, center=None, ligand=None, distance=None, geometry=None,
                                 offset=None, plane=None, axis=None, domains=None, unit='Angstrom'):
        '''
        A minimal molecule builder for simple one-center, homogeneous ligand
        molecules of various general chemistry molecular geometries. If domains
        is not specified and geometry is ambiguous (like 'bent'),
        it just guesses the simplest geometry (smallest number of domains).

        Args
            center (str): atomic symbol of central atom
            ligand (str): atomic symbol of ligand atoms
            distance (float): distance between central atom and any ligand
            geometry (str): molecular geometry
            domains (int): number of electronic domains
            offset (np.array): 3-array of position of central atom
            plane (str): cartesian plane of molecule (eg. for 'square_planar')
            axis (str): cartesian axis of molecule (eg. for 'linear')

        Returns
            exatomic.atom.Atom: Atom table of small molecule
        '''
        return cls(make_small_molecule(center=center, ligand=ligand, distance=distance,
                                       geometry=geometry, offset=offset, plane=plane,
                                       axis=axis, domains=domains, unit=unit))


class UnitAtom(DataFrame):
    """
    In unit cell coordinates (sparse) for periodic systems. These coordinates
    are used to update the corresponding :class:`~exatomic.atom.Atom` object
    """
    _index = 'atom'
    _columns = ['x', 'y', 'z']

    #@property
    #def _constructor(self):
    #    return UnitAtom

    @classmethod
    def from_universe(cls, universe):
        if universe.periodic:
            if "rx" not in universe.frame.columns:
                universe.frame.compute_cell_magnitudes()
            a, b, c = universe.frame[["rx", "ry", "rz"]].max().values
            x = modv(universe.atom['x'].values, a)
            y = modv(universe.atom['y'].values, b)
            z = modv(universe.atom['z'].values, c)
            df = pd.DataFrame.from_dict({'x': x, 'y': y, 'z': z})
            df.index = universe.atom.index
            return cls(df[universe.atom[['x', 'y', 'z']] != df])
        raise PeriodicUniverseError()


class ProjectedAtom(DataFrame):
    """
    Projected atom coordinates (e.g. on 3x3x3 supercell). These coordinates are
    typically associated with their corresponding indices in another dataframe.

    Note:
        This table is computed when periodic two body properties are computed;
        it doesn't have meaning outside of that context.

    See Also:
        :func:`~exatomic.two.compute_periodic_two`.
    """
    _index = 'two'
    _columns = ['x', 'y', 'z']

    #@property
    #def _constructor(self):
    #    return ProjectedAtom


class VisualAtom(DataFrame):
    """
    """
    _index = 'atom'
    _columns = ['x', 'y', 'z']

    @classmethod
    def from_universe(cls, universe):
        """
        """
        if universe.frame.is_periodic():
            atom = universe.atom[['x', 'y', 'z']].copy()
            atom.update(universe.unit_atom)
            bonded = universe.atom_two.loc[universe.atom_two['bond'] == True, 'atom1'].astype(np.int64)
            prjd = universe.projected_atom.loc[bonded.index].to_dense()
            prjd['atom'] = bonded
            prjd.drop_duplicates('atom', inplace=True)
            prjd.set_index('atom', inplace=True)
            atom.update(prjd)
            return cls(atom[atom != universe.atom[['x', 'y', 'z']]])
        raise PeriodicUniverseError()

    #@property
    #def _constructor(self):
    #    return VisualAtom


class Frequency(DataFrame):
    """
    The Frequency dataframe.

    +-------------------+----------+-------------------------------------------+
    | Column            | Type     | Description                               |
    +===================+==========+===========================================+
    | frame             | category | non-unique integer (req.)                 |
    +-------------------+----------+-------------------------------------------+
    | frequency         | float    | frequency of oscillation (cm-1) (req.)    |
    +-------------------+----------+-------------------------------------------+
    | freqdx            | int      | index of frequency of oscillation (req.)  |
    +-------------------+----------+-------------------------------------------+
    | dx                | float    | atomic displacement in x direction (req.) |
    +-------------------+----------+-------------------------------------------+
    | dy                | float    | atomic displacement in y direction (req.) |
    +-------------------+----------+-------------------------------------------+
    | dz                | float    | atomic displacement in z direction (req.) |
    +-------------------+----------+-------------------------------------------+
    | ir_int            | float    | ir intensity of the vibrational mode      |
    +-------------------+----------+-------------------------------------------+
    | symbol            | str      | atomic symbol (req.)                      |
    +-------------------+----------+-------------------------------------------+
    | label             | int      | atomic identifier                         |
    +-------------------+----------+-------------------------------------------+
    """
    _index = 'frequency'
    _cardinal = ('frame', np.int64)
    _categories = {'symbol': str, 'label': np.int64}
    _columns = ['dx', 'dy', 'dz', 'symbol', 'frequency', 'freqdx', 'ir_int']
    #@property
    #def _constructor(self):
    #    return Frequency

    def displacement(self, freqdx):
        return self[self['freqdx'] == freqdx][['dx', 'dy', 'dz', 'symbol']]

    def ir_spectra(self, fwhm=15, lineshape='gaussian', xrange=None, res=None, invert_x=False, **kwargs):
        '''
        Generate an IR spectra with the plotter classes. We can define a gaussian or lorentzian
        lineshape functions. For the most part we pass all of the kwargs directly into the
        plotter.Plot class.

        Args:
            fwhm (float): Full-width at half-maximum
            lineshape (str): Switch between the different lineshape functions available
            xrange (list): X-bounds for the plot
            res (float): Resolution for the plot line
            invert_x (bool): Invert x-axis
        '''
        # define the lineshape and store the function call in the line variable
        try:
            line = getattr(plotter, lineshape)
        except AttributeError:
            raise NotImplementedError("Sorry we have not yet implemented the lineshape {}.".format(lineshape))
        # define a default parameter for the plot width
        # we did this for a full-screen jupyter notebook on a 1920x1080 monitor
        if not "plot_width" in kwargs:
            kwargs.update(plot_width=900)
        # define xbounds
        xrange = [0, 4000] if xrange is None else xrange
        # deal with inverted bounds
        if xrange[0] > xrange[1]:
            xrange = sorted(xrange)
            invert_x = True
        # define the resolution
        res = fwhm/50 if res is None else res
        # define the class
        plot = plotter.Plot(**kwargs)
        # this is designed for a single frame
        if self['frame'].unique().shape[0] != 1:
            raise NotImplementedError("We have not yet expanded to include multiple frames")
        # grab the locations of the peaks between the bounds
        freqdx = self['freqdx'].drop_duplicates().index
        freq = self.loc[freqdx, 'frequency']
        freq = freq[freq.between(*xrange)]
        # grab the ir intensity data
        # we use the frequency indexes instead of drop duplicates as we may have similar intensities
        inten = self.loc[freq.index, 'ir_int'].astype(np.float64).values
        # change to using the values instead as we no longer need the index data
        # we could also use jit for the lineshape functions as we only deal with numpy arrays
        freq = freq.values
        x_data = np.arange(*xrange, res)
        # get the y data by calling the lineshape function generator
        y_data = line(freq=freq, x=x_data, fwhm=fwhm, inten=inten)
        # plot the lineshape data
        plot.fig.line(x_data, y_data)
        # plot the points on the plot to show were the frequency values are
        # more useful when we have nearly degenerate vibrations
        plot.fig.scatter(freq, line(freq=freq, x=freq, fwhm=fwhm, inten=inten))
        if invert_x:
            plot.set_xrange(xmin=xrange[1], xmax=xrange[0])
        else:
            plot.set_xrange(xmin=xrange[0], xmax=xrange[1])
        # display the figure with our generated method
        plot.show()

def add_vibrational_mode(uni, freqdx):
    displacements = uni.frequency.displacements(freqdx)
    if not all(displacements['symbol'] == uni.atom['symbol']):
        print('Mismatch in ordering of atoms and frequencies.')
        return
    displaced = []
    frames = []
    # Should these only be absolute values?
    factor = np.abs(np.sin(np.linspace(-4*np.pi, 4*np.pi, 200)))
    for fac in factor:
        moved = uni.atom.copy()
        moved['x'] += displacements['dx'].values * fac
        moved['y'] += displacements['dy'].values * fac
        moved['z'] += displacements['dz'].values * fac
        displaced.append(moved)
        frames.append(uni.frame)
    movie = pd.concat(displaced).reset_index()
    movie['frame'] = np.repeat(range(len(factor)), len(uni.atom))
    uni.frame = pd.concat(frames).reset_index()
    uni.atom = movie

