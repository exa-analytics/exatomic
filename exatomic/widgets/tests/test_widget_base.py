# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
from os.path import isfile, join, abspath
import exatomic
from unittest import TestCase
from ..widget_base import ExatomicScene, UniverseScene, ExatomicBox, _scene_grid


class TestExatomicScene(TestCase):
    def setUp(self):
        self.scn = ExatomicScene()
        self.scn.savedir = abspath(join(abspath(exatomic.__file__),
                                        '../static/'))

    def test_save_camera(self):
        self.scn._save_camera({'key': 'value'})
        self.assertDictEqual(self.scn.cameras[0], {'key': 'value'})

    def test_save_image(self):
        # A simple tiny red dot in base64
        cont = ("data:image/png;base64, iVBORw0KGgoAAAANSUhEUgA"
                "AAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GI"
                "AXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg==")
        self.scn._save_image(cont)
        self.scn.imgname = 'myimg.png'
        self.scn._save_image(cont)
        self.assertTrue(isfile(join(self.scn.savedir, '000000.png')))
        self.assertTrue(isfile(join(self.scn.savedir, 'myimg.png')))
        os.remove(abspath(join(self.scn.savedir, '000000.png')))
        os.remove(abspath(join(self.scn.savedir, 'myimg.png')))

    def test_handle_custom_msg(self):
        # How to test receiving msg from JS?
        pass

    def test_set_camera(self):
        # How to test sending msg to JS?
        pass

    def test_close(self):
        # How to test sending msg to JS?
        pass

    def test_init(self):
        self.assertEqual(self.scn.layout.flex, '1 1 auto')
        self.assertEqual(self.scn.layout.height, 'auto')
        self.assertEqual(self.scn.layout.min_height, '400px')
        self.assertEqual(self.scn.layout.min_width, '300px')
        scn = ExatomicScene(min_height='500px', min_width='400px')
        self.assertEqual(scn.layout.min_height, '500px')
        self.assertEqual(scn.layout.min_width, '400px')


class TestUniverseScene(TestCase):
    def test_traits(self):
        scn = UniverseScene(atom_x='[[0.0,0.0]]')
        self.assertEqual(scn.atom_x, '[[0.0,0.0]]')


class TestExatomicBox(TestCase):
    def setUp(self):
        self.box = ExatomicBox(2)

    def test_update_active(self):
        self.assertListEqual(self.box.active_scene_indices, [0, 1])
        self.box._controls['active']._controls['0'].value = False
        self.box._update_active(None)
        self.assertListEqual(self.box.active_scene_indices, [1])
        self.box._controls['active']._controls['0'].value = True
        self.box._update_active(None)
        self.assertListEqual(self.box.active_scene_indices, [0, 1])

    def test_active_folder(self):
        fol = self.box._active_folder()
        self.assertListEqual(list(fol._controls.keys()), ['main', '0', '1'])

    def test_save_folder(self):
        fol = self.box._save_folder()
        self.assertListEqual(list(fol._controls.keys()),
                             ['main', 'dir', 'name', 'save'])

    def test_camera_folder(self):
        fol = self.box._camera_folder()
        self.assertListEqual(list(fol._controls.keys()),
                             ['main', 'link', 'get', 'set'])

    def test_field_folder(self):
        fol = self.box._field_folder()
        self.assertListEqual(list(fol._controls.keys()),
                             ['main', 'alpha', 'iso', 'nx', 'ny', 'nz'])

    def test_init_gui(self):
        main = self.box._init_gui()
        self.assertListEqual(list(main.keys()),
                             ['close', 'clear', 'active', 'saves', 'camera'])

    def test_active(self):
        self.assertEqual(self.box.scenes, self.box.active())

    def test_init(self):
        # 3 cases here as in test_scene_grid but focus on
        # proper passing of kwargs to scenes // whatnot
        # TODO : more variations
        ExatomicBox()


class TestSceneGrid(TestCase):
    def test_scene_grid(self):
        # 3 cases -- from least specific to most specific
        # 1. a number of scenes
        # 2. iterable of scenes
        # 2. iterable of kwargs
        #flat, widg = _scene_grid([1], None, None, True, False,
        flat, widg = _scene_grid([1], None, None, False,
                                 ExatomicScene, {})
        self.assertEqual(len(flat), 1)
        self.assertEqual(len(widg.children), 1)
        flat, widg = _scene_grid([{'geom': False}, {'geom': True}],
                                 None, None, False,
                                 ExatomicScene, {})
        self.assertEqual(len(flat), 2)
        self.assertEqual(len(widg.children), 1)
        self.assertEqual(flat[0].geom, False)
        self.assertEqual(flat[1].geom, True)
        flat, widg = _scene_grid([UniverseScene(),
                                  UniverseScene(),
                                  ExatomicScene()],
                                 None, None, False,
                                 ExatomicScene, {})
        self.assertEqual(len(flat), 3)
        self.assertEqual(len(widg.children), 2)

