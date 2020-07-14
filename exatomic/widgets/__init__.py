# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0

from .widget import TensorContainer, DemoContainer, DemoUniverse, UniverseWidget

def exhibition_widget():
    """An exhibition widget from static resources
    that contains enough data to demonstrate numerous pieces
    of functionality in the UniverseWidget.

    Control each scene individually by selecting active
    scenes based on the index of their layout reading
    left to right, top to bottom. The Camera tab allows
    for linking cameras across scenes. The Fill tab
    controls the atomic display model and Axis will display
    a unit vector (defaults to the origin).

    Returns:
        UniverseWidget featuring the application.
            scene 0: trajectory animation (Animate tab)
            scene 1: orbital isosurfaces (Fields, Contours tabs)
            scene 2: NMR shielding tensor (Tensor tab)

    Note:
        All scenes are active by default and not all required
        data are exposed in each scene. Therefore, "unselect"
        active scenes in the Active Scenes tab (at least until
        there is better exception handling on the JS side).

    Note:
        This widget provides test cases for marching cubes
        (and marching squares) over 3D (2D) scalar fields, animated
        trajectories, and plotting parametric surfaces, but
        not all at the same time. The aim is to provide a
        complex enough test case that covers a good portion of
        the JS code so that updates can be checked (albeit in
        a time-consuming manner). It may also serve to uncover
        python-related bugs related to a UniverseWidget housing
        multiple independent UniverseScenes.

    Note:
        The use of an entire tensor table is not well supported
        by the application yet. The aim is to improve functionality
        to be similar to the functionality for isosurfaces.

    """
    import exatomic
    from exatomic import gaussian

    trj_file = 'H2O.traj.xyz'
    orb_file = 'g09-ch3nh2-631g.out'
    nmr_file = 'g16-nitromalonamide-6-31++g-nmr.out'

    trj = exatomic.XYZ(
        exatomic.base.resource(trj_file)).to_universe()

    orb = gaussian.Output(
        exatomic.base.resource(orb_file)).to_universe()
    orb.add_molecular_orbitals()

    nmr = gaussian.Output(
        exatomic.base.resource(nmr_file)).to_universe()
    nmr.tensor = nmr.nmr_shielding

    return exatomic.UniverseWidget(trj, orb, nmr)
