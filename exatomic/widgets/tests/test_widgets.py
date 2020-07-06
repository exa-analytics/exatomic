# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0

from unittest import TestCase

from exatomic import XYZ

from ..widget import DemoContainer, DemoUniverse, UniverseWidget

# TODO : need good test universe with fields

h2 = '''2

H 0.0 0.0 -0.35
H 0.0 0.0  0.35
'''

class TestDemoContainer(TestCase):
    def setUp(self):
        self.box = DemoContainer()

    def test_field_folder(self):
        fol = self.box._field_folder()
        self.assertListEqual(list(fol._controls.keys()),
                             ['main', 'options', 'alpha', 'iso', 'nx', 'ny', 'nz'])

    def test_init_gui(self):
        main = self.box._init_gui()
        self.assertListEqual(list(main.keys()),
                             ['close', 'clear', 'active', 'saves', 'camera', 'geom', 'field'])


class TestDemoUniverse(TestCase):
    def setUp(self):
        self.box = DemoUniverse()

    def test_field_folder(self):
        fol = self.box._field_folder()
        self.assertListEqual(fol._get(keys=True),
                             ['main', 'fopts', 'Hydrogenic'])
                          #'alpha', 'iso', 'nx', 'ny', 'nz'])
                          # these aren't active if folder isn't open

    def test_init_gui(self):
        main = self.box._init_gui()
        self.assertListEqual(list(main.keys()),
                             ['close', 'clear', 'active', 'saves', 'camera', 'field'])


class TestUniverseWidget(TestCase):
    def setUp(self):
        xyz = XYZ(h2)
        xyz.parse_atom()
        self.uni = xyz.to_universe()
        self.box = UniverseWidget(self.uni)

    def test_frame_folder(self):
        fol = self.box._frame_folder(self.uni.atom.nframes)
        self.assertListEqual(list(fol._controls.keys()),
                             ['main', 'playing', 'scn_frame'])

    def test_init_gui(self):
        main = self.box._init_gui()
        self.assertListEqual(list(main.keys()),
                             ['close', 'clear', 'active', 'saves',
                              'camera', 'atom_3d', 'axis', 'frame',
                              'distance'])
