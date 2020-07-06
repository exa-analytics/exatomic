# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0

from unittest import TestCase

from ipywidgets import Button

from ..widget_utils import _ListDict, Folder, GUIBox, gui_field_widgets


class TestFolder(TestCase):

    def test_init(self):
        fol = Folder(Button(), _ListDict([('a', Button()), ('b', Button())]))
        self.assertEqual(fol['main'].active, True)
        self.assertEqual(fol['main'].layout.width, '98%')
        self.assertEqual(fol['a'].active, False)
        self.assertEqual(fol['a'].layout.width, '93%')

        fol = Folder(Button(), _ListDict([('a', Button()), ('b', Button())]),
                     show=True, level=1)
        self.assertEqual(fol['main'].active, True)
        self.assertEqual(fol['main'].layout.width, '93%')
        self.assertEqual(fol['a'].active, True)
        self.assertEqual(fol['a'].layout.width, '88%')

    def test_activate(self):
        fol = Folder(Button(), _ListDict([('a', Button()), ('b', Button())]))
        fol.activate('a')
        self.assertTrue(fol['a'].active)
        self.assertTrue(not fol['b'].active)
        fol.activate()
        self.assertTrue(fol['b'].active)

    def test_deactivate(self):
        fol = Folder(Button(), _ListDict([('a', Button()), ('b', Button())]))
        fol.activate()
        fol.deactivate('a')
        self.assertTrue(fol['a'].disabled)
        self.assertTrue(not fol['b'].disabled)
        fol.deactivate()
        self.assertTrue(fol['b'].disabled)

    def test_insert(self):
        fol = Folder(Button(), _ListDict([('a', Button()), ('b', Button())]))
        fol.insert(1, 'c', Button(), active=True)
        self.assertListEqual(list(fol._controls.keys()),
                             ['main', 'c', 'a', 'b'])

    def test_update(self):
        fol = Folder(Button(), _ListDict([('a', Button()), ('b', Button())]))
        fol.update([('c', Button())])
        self.assertEqual(type(fol['c']), Button)

    def move_to_end(self):
        fol = Folder(Button(), _ListDict([('a', Button()), ('b', Button())]))
        fol.move_to_end('a')
        self.assertEqual(list(fol._controls.keys())[-1], 'a')

    def test_pop(self):
        fol = Folder(Button(), _ListDict([('a', Button()), ('b', Button())]))
        fol.pop('a')
        with self.assertRaises(KeyError):
            _ = fol['a']

    def test_get(self):
        fol = Folder(Button(), _ListDict([('a', Button()), ('b', Button())]))
        self.assertListEqual(fol._get(keys=True), ['main'])
        fol.activate()
        self.assertListEqual(fol._get(keys=True), ['main', 'a', 'b'])



class TestListDict(TestCase):

    def test_init(self):
        with self.assertRaises(TypeError):
            _ = _ListDict([('a', 1), (2, [1, 2])])

    def pop_as_dict(self):
        ld = _ListDict([('a', 1), ('b', [1, 2])])
        self.assertEqual(ld.pop('a'), 1)
        self.assertEqual(len(ld), 1)
        self.assertEqual(ld.pop('b'), [1, 2])
        self.assertEqual(len(ld), 0)

    def test_pop_as_list(self):
        ld = _ListDict([('a', 1), ('b', [1, 2])])
        self.assertEqual(ld.pop(1), [1, 2])
        self.assertEqual(len(ld), 1)
        self.assertEqual(ld.pop(0), 1)
        self.assertEqual(len(ld), 0)

    def test_insert(self):
        ld = _ListDict([('a', 1), ('b', [1, 2])])
        ld.insert(0, 'c', 3)
        self.assertListEqual(list(ld.keys()), ['c', 'a', 'b'])
        ld.insert(2, 'd', 5)
        self.assertListEqual(list(ld.keys()), ['c', 'a', 'd', 'b'])

    def test_set(self):
        ld = _ListDict([('a', 1), ('b', [1, 2])])
        ld['c'] = 2
        self.assertEqual(ld['c'], 2)
        with self.assertRaises(TypeError):
            ld[2] = 2

    def test_get(self):
        ld = _ListDict([('a', 1), ('b', [1, 2])])
        self.assertEqual(ld['a'], 1)
        self.assertEqual(ld['b'], [1, 2])
        self.assertEqual(ld[0], 1)
        self.assertEqual(ld[1], [1, 2])
        self.assertEqual(ld[:1], [1])
        self.assertEqual(ld[:2], [1, [1, 2]])
        with self.assertRaises(TypeError):
            _ = ld['a', 'b']


class TestGUIBox(TestCase):
    def test_init(self):
        gui = GUIBox()
        self.assertEqual(gui.layout.flex, '0 0 240px')


class TestGuiFieldWidgets(TestCase):

    def test_gui_field_widgets(self):
        a = gui_field_widgets(False, False)
        self.assertEqual(a['iso'].value, 3.0)
        a = gui_field_widgets(True, True)
        self.assertEqual(a['iso'].value, 0.0005)
        a = gui_field_widgets(True, False)
        self.assertEqual(a['iso'].value, 0.03)
