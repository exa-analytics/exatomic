# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Tests for :mod:`~exa.core.editor`
##################################
"""
import io
import os
from tempfile import mkdtemp
from unittest import TestCase
from contextlib import redirect_stdout
import pandas as pd
from exa import Editor



class TestEditor(TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Generate the file path to the exa.core.editor module (which will be used as
        the test for the :class:`~exa.core.editor.Editor` class that it provides).
        """
        cls.path = os.path.abspath(os.path.join(os.path.abspath(__file__),
                                                "..", "..", "editor.py"))
        with open(cls.path) as f:
            cls.lines = f.readlines()
        cls.fl = Editor.from_file(cls.path)

    def test_loaders(self):
        """
        The editor provides there different methods for instantiation; from a
        file on disk, from a file stream, or from a string.
        """
        fl = Editor.from_file(self.path)
        with open(self.path) as f:
            tm = Editor.from_stream(f)
            f.seek(0)
            tr = Editor.from_string(f.read())
        self.assertTrue(len(self.lines) == len(fl) == len(tm) == len(tr))
        self.assertTrue(all(fl[i] == tm[i] == tr[i] for i in range(len(self.lines))))

    def test_find_regex(self):
        od = self.fl.find('Args:')
        self.assertIsInstance(od, list)
        self.assertIsInstance(od[0], tuple)
        self.assertTrue(len(od) == 16)
        self.assertTrue(self.fl.cursor == 0)
        n0, line0 = self.fl.find_next('Args:')
        self.assertIsInstance(n0, int)
        self.assertIsInstance(line0, str)
        self.assertIn(n0, od[0])
        self.assertTrue(self.fl.cursor > 0)
        n1, line1 = self.fl.find_next('Args:')
        self.assertTrue(n1 > n0)
        self.assertIsInstance(line1, str)
        self.assertIn(n1, od[1])
        od1 = self.fl.regex('Args:')
        self.assertTrue(od == od1)    # True in this case; depends on regex used

    def test_insert_format_plus(self):
        n = len(self.fl)
        lines = {0: '{INSERTED}'}
        self.fl.insert(lines)
        self.assertTrue(len(self.fl) == n + 1)
        self.assertTrue(self.fl[0] == lines[0])
        self.fl.delete_lines(range(5, len(self.fl)))
        self.assertTrue(len(self.fl) == 5)
        self.assertTrue(len(self.fl.variables) == 1)
        formatted = self.fl.format(INSERTED='replaced').splitlines()
        self.assertTrue(formatted[0] == 'replaced')
        del self.fl[0]
        self.assertTrue(len(self.fl) == 4)

    def test_write(self):
        f = io.StringIO()
        with redirect_stdout(f):
            self.fl.write()
        flstr = str(self.fl)
        self.assertEqual(flstr, f.getvalue()[:-1])    # account for newline=""
        dir_ = mkdtemp()
        path = os.path.join(dir_, "editor.py")
        self.fl.write(path=path)
        with open(path) as f:
            text = f.read()
        self.assertEqual(flstr, text)
        os.remove(path)
        os.rmdir(dir_)

    def test_format_inplace(self):
        ed = Editor("hello {name}", ignore=True)
        self.assertEqual(str(ed), "hello {name}")
        self.assertEqual(ed.variables, ['{name}'])
        ed.format(inplace=True, name="world")
        self.assertEqual(str(ed), "hello world")

    def test_head(self):
        ed = Editor("hello\nworld")
        f = io.StringIO()
        with redirect_stdout(f):
            ed.head(1)
        self.assertEqual("hello", f.getvalue())

    def test_tail(self):
        ed = Editor("hello\nworld")
        f = io.StringIO()
        with redirect_stdout(f):
            ed.tail(1)
        self.assertEqual("world", f.getvalue())

    def test_append(self):
        ed = Editor("hello", ignore=True)
        ed.append("world")
        self.assertEqual("hello\nworld", str(ed))

    def test_preappend(self):
        ed = Editor("world", ignore=True)
        ed.prepend("hello")
        self.assertEqual("hello\nworld", str(ed))

    def test_remove_blank_lines(self):
        ed = Editor("hello\n\nworld")
        self.assertEqual("hello\n\nworld", str(ed))
        ed.remove_blank_lines()
        self.assertEqual("hello\nworld", str(ed))

    def test_find_keys(self):
        keys = self.fl.find('Args:', keys_only=True)
        self.assertIsInstance(keys[0], int)
        self.assertIsInstance(keys, list)
        self.assertTrue(len(keys) > 1)

    def test_replace(self):
        ed = Editor("hello world", ignore=True)
        ed.replace("world", "universe")
        self.assertEqual(str(ed), "hello universe")

    def test_pandas_dataframe(self):
        ed = Editor("hello\nworld")
        df = ed.pandas_dataframe(0, len(ed), 1)
        self.assertTrue(df.equals(pd.DataFrame([["hello"], ["world"]])))
        df = ed.pandas_dataframe(0, len(ed), ["text"])
        self.assertTrue(df.equals(pd.DataFrame([["hello"], ["world"]], columns=["text"])))

    def test_dunder(self):
        ed = Editor("hello world", ignore=True)
        self.assertEqual(str(ed), "hello world")
        self.assertEqual(len(ed), 1)
        self.assertTrue("hello" in ed)
        self.assertTrue(callable(ed["find"]))
        ed[0] = "hi"
        self.assertEqual(str(ed), "hi")
        for line in ed:
            pass
        self.assertEqual(line, str(ed))
        del ed[0]
        self.assertEqual(str(ed), "")

    def test_insert(self):
        ed = Editor("world", ignore=True)
        ed.insert({0: "hello"})
        self.assertEqual(str(ed), "hello\nworld")

    def test_delete_lines(self):
        ed = Editor("hello\nworld")
        ed.delete_lines([0])
        self.assertEqual(str(ed), "world")
