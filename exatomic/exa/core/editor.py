# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Editor
####################################
Text-editor-like functionality for programatically manipulating raw text input
and output data and converting this data into container objects. This class
does not behave like a fully fledged text editor but does have some basic find,
replace, insert, etc. functionality.
"""
from __future__ import print_function
import logging
import io, os, re, sys
import pandas as pd
import numpy as np
import warnings


class Editor(object):
    """
    An editor is a representation of a text file on disk that can be
    programmatically manipulated.

    Text lines are stored in memory; no files remain open. This class does not
    strive to be a fully fledged text editor rather a base class for converting
    input and output data from text on disk to some type of (exa framework)
    container object (and vice versa).

    >>> template = "Hello World!\\nHello {user}"
    >>> editor = Editor(template)
    >>> editor[0]
    'Hello World!'
    >>> len(editor)
    2
    >>> del editor[0]
    >>> len(editor)
    1
    >>> editor.write(fullpath=None, user='Alice')
    Hello Alice

    Tip:
        Editor line numbers use a 0 base index. To increase the number of lines
        displayed by the repr, increase the value of the **nprint** attribute.

    Warning:
        For large text with repeating strings be sure to use the **as_interned**
        argument.

    Attributes:
        name (str): Data/file/misc name
        description (str): Data/file/misc description
        meta (dict): Additional metadata as key, value pairs
        nrpint (int): Number of lines to display when printing
        cursor (int): Line number position of the cusor (see :func:`~exa.core.editor.Editor.find_next`)
    """
    _getter_prefix = 'parse'
    _fmt = '{0}: {1}\n'.format   # Format for printing lines (see __repr__)

    @property
    def log(self):
        name = '.'.join([self.__module__,
                         self.__class__.__name__])
        return logging.getLogger(name)

    def write(self, path=None, *args, **kwargs):
        """
        Perform formatting and write the formatted string to a file or stdout.

        Optional arguments can be used to format the editor's contents. If no
        file path is given, prints to standard output.

        Args:
            path (str): Full file path (default None, prints to stdout)
            *args: Positional arguments to format the editor with
            **kwargs: Keyword arguments to format the editor with
        """
        if path is None:
            print(self.format(*args, **kwargs))
        else:
            with io.open(path, 'w', newline="") as f:
                f.write(self.format(*args, **kwargs))

    def format(self, *args, **kwargs):
        """
        Format the string representation of the editor.

        Args:
            inplace (bool): If True, overwrite editor's contents with formatted contents
        """
        inplace = kwargs.pop("inplace", False)
        if not inplace:
            return str(self).format(*args, **kwargs)
        self._lines = str(self).format(*args, **kwargs).splitlines()

    def head(self, n=10):
        """
        Display the top of the file.

        Args:
            n (int): Number of lines to display
        """
        print("".join(self._lines[:n]), end="")

    def tail(self, n=10):
        """
        Display the bottom of the file.

        Args:
            n (int): Number of lines to display
        """
        print("".join(self._lines[-n:]), end="")

    def append(self, lines):
        """
        Args:
            lines (list): List of line strings to append to the end of the editor
        """
        if isinstance(lines, list):
            self._lines = self._lines + lines
        elif isinstance(lines, str):
            lines = lines.split('\n')
            self._lines = self._lines + lines
        else:
            raise TypeError(f"Unsupported type '{type(lines)}' for lines")

    def prepend(self, lines):
        """
        Args:
            lines (list): List of line strings to insert at the beginning of the editor
        """
        if isinstance(lines, list):
            self._lines = lines + self._lines
        elif isinstance(lines, str):
            lines = lines.split('\n')
            self._lines = lines + self._lines
        else:
            raise TypeError(f"Unsupported type '{type(lines)}' for lines")

    def insert(self, lines=None):
        """
        Insert lines into the editor.

        Note:
            To insert before the first line, use :func:`~exa.core.editor.Editor.preappend`
            (or key 0); to insert after the last line use :func:`~exa.core.editor.Editor.append`.

        Args:
            lines (dict): Dictionary of lines of form (lineno, string) pairs
        """
        for i, (key, line) in enumerate(lines.items()):
            n = key + i
            first_half = self._lines[:n]
            last_half = self._lines[n:]
            self._lines = first_half + [line] + last_half

    def remove_blank_lines(self):
        """Remove all blank lines (blank lines are those with zero characters)."""
        to_remove = []
        for i, line in enumerate(self):
            ln = line.strip()
            if ln == '':
                to_remove.append(i)
        self.delete_lines(to_remove)

    def delete_lines(self, lines):
        """
        Delete all lines with given line numbers.

        Args:
            lines (list): List of integers corresponding to line numbers to delete
        """
        for k, i in enumerate(lines):
            del self[i-k]    # Accounts for the fact that len(self) decrease upon deletion

    def find(self, *strings, **kwargs):
        """
        Search the entire editor for lines that match the string.

        .. code-block:: Python

            string = '''word one
            word two
            three'''
            ed = Editor(string)
            ed.find('word')          # [(0, "word one"), (1, "word two")]
            ed.find('word', 'three') # {'word': [...], 'three': [(2, "three")]}

        Args:
            strings (str): Any number of strings to search for
            keys_only (bool): Only return keys
            start (int): Optional line to start searching on
            stop (int): Optional line to stop searching on

        Returns:
            results: If multiple strings searched a dictionary of string key, (line number, line) values (else just values)
        """
        start = kwargs.pop("start", 0)
        stop = kwargs.pop("stop", None)
        keys_only = kwargs.pop("keys_only", False)
        results = {string: [] for string in strings}
        stop = len(self) if stop is None else stop
        for i, line in enumerate(self[start:stop]):
            for string in strings:
                if string in line:
                    if keys_only:
                        results[string].append(i)
                    else:
                        results[string].append((i, line))
        if len(strings) == 1:
            return results[strings[0]]
        return results

    def find_next(self, *strings, **kwargs):
        """
        From the editor's current cursor position find the next instance of the
        given string.

        Args:
            strings (iterable): String or strings to search for

        Returns:
            tup (tuple): Tuple of cursor position and line or None if not found

        Note:
            This function cycles the entire editor (i.e. cursor to length of
            editor to zero and back to cursor position).
        """
        start = kwargs.pop("start", None)
        keys_only = kwargs.pop("keys_only", False)
        staht = start if start is not None else self.cursor
        for start, stop in [(staht, len(self)), (0, staht)]:
            for i in range(start, stop):
                for string in strings:
                    if string in self[i]:
                        tup = (i, self[i])
                        self.cursor = i + 1
                        if keys_only: return i
                        return tup

    def regex(self, *patterns, **kwargs):
        """
        Search the editor for lines matching the regular expression.
        re.MULTILINE is not currently supported.

        Args:
            patterns: Regular expressions to search each line for
            keys_only (bool): Only return keys
            flags (re.FLAG): flags passed to re.search

        Returns:
            results (dict): Dictionary of pattern keys, line values (or groups - default)
        """
        start = kwargs.pop("start", 0)
        stop = kwargs.pop("stop", None)
        keys_only = kwargs.pop("keys_only", False)
        flags = kwargs.pop("flags", 0)
        results = {pattern: [] for pattern in patterns}
        stop = stop if stop is not None else -1
        for i, line in enumerate(self[start:stop]):
            for pattern in patterns:
                grps = re.search(pattern, line, flags=flags)
                if grps and keys_only:
                    results[pattern].append(i)
                elif grps and grps.groups():
                    for group in grps.groups():
                        results[pattern].append((i, group))
                elif grps:
                    results[pattern].append((i, line))
        if len(patterns) == 1:
            return results[patterns[0]]
        return results

    def replace(self, pattern, replacement):
        """
        Replace all instances of a pattern with a replacement.

        Args:
            pattern (str): Pattern to replace
            replacement (str): Text to insert
        """
        for i, line in enumerate(self):
            if pattern in line:
                self[i] = line.replace(pattern, replacement)

    def pandas_dataframe(self, start, stop, ncol, **kwargs):
        """
        Returns the result of tab-separated pandas.read_csv on
        a subset of the file.

        Args:
            start (int): line number where structured data starts
            stop (int): line number where structured data stops
            ncol (int or list): the number of columns in the structured
                data or a list of that length with column names

        Returns:
            pd.DataFrame: structured data
        """
        if isinstance(ncol, (int, np.int, np.int64, np.int32)):
            return pd.read_csv(io.StringIO('\n'.join(self[start:stop])), delim_whitespace=True, names=range(ncol), **kwargs)
        else:
            return pd.read_csv(io.StringIO('\n'.join(self[start:stop])), delim_whitespace=True, names=ncol, **kwargs)

    def to_stream(self):
        """Create an StringIO object from the current editor text."""
        return io.StringIO(str(self))

    @property
    def variables(self):
        """
        Display a list of templatable variables present in the file.

        Templating is accomplished by creating a bracketed object in the same
        way that Python performs `string formatting`_. The editor is able to
        replace the placeholder value of the template. Integer templates are
        positional arguments.

        .. _string formatting: https://docs.python.org/3.6/library/string.html
        """
        string = str(self)
        constants = [match[1:-1] for match in re.findall('{{[A-z0-9]}}', string)]
        variables = re.findall('{[A-z0-9]*}', string)
        return sorted(set(variables).difference(constants))

    @classmethod
    def from_file(cls, path, **kwargs):
        """Create an editor instance from a file on disk."""
        lines = lines_from_file(path)
        if 'meta' not in kwargs:
            kwargs['meta'] = {'from': 'file'}
        kwargs['meta']['filepath'] = path
        return cls(lines, **kwargs)

    @classmethod
    def from_stream(cls, f, **kwargs):
        """Create an editor instance from a file stream."""
        lines = lines_from_stream(f)
        if 'meta' not in kwargs:
            kwargs['meta'] = {'from': 'stream'}
        kwargs['meta']['filepath'] = f.name if hasattr(f, 'name') else None
        return cls(lines, **kwargs)

    @classmethod
    def from_string(cls, string, **kwargs):
        """Create an editor instance from a string template."""
        return cls(lines_from_string(string), **kwargs)

    def __init__(self, path_stream_or_string, as_interned=False, nprint=30,
                 name=None, description=None, meta=None, encoding=None, ignore=False):
        # Backporting file check
        textobj = path_stream_or_string
        if (isinstance(textobj, str) and len(textobj.split("\n")) == 1
                and ignore == False and not os.path.exists(textobj)):
            warnings.warn("Possibly incorrect file path! {}".format(textobj))
        #if len(path_stream_or_string) < 256 and os.path.exists(path_stream_or_string):
        if (isinstance(path_stream_or_string, str) and
                len(path_stream_or_string) < 32760 and
                os.path.exists(path_stream_or_string)):
            self._lines = lines_from_file(path_stream_or_string, as_interned, encoding)
        elif isinstance(path_stream_or_string, (list, tuple)):
            self._lines = path_stream_or_string
        elif isinstance(path_stream_or_string, (io.TextIOWrapper, io.StringIO)):
            self._lines = lines_from_stream(path_stream_or_string, as_interned)
        elif isinstance(path_stream_or_string, str):
            self._lines = lines_from_string(path_stream_or_string, as_interned)
        else:
            raise TypeError('Unknown type for arg data: {}'.format(type(path_stream_or_string)))
        self.name = name
        self.description = description
        self.meta = meta
        self.nprint = 30
        self.cursor = 0
        self.log.debug('contains {} lines'.format(len(self._lines)))

    def __delitem__(self, line):
        del self._lines[line]     # "line" is the line number minus one

    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key)
        return self._lines[key]

    def __setitem__(self, line, value):
        self._lines[line] = value

    def __iter__(self):
        for line in self._lines:
            yield line

    def __len__(self):
        return len(self._lines)

    def __str__(self):
        return '\n'.join(self._lines)

    def __contains__(self, item):
        for obj in self:
            if item in obj:
                return True

    def __repr__(self):
        r = ''
        nn = len(self)
        n = len(str(nn))
        if nn > self.nprint * 2:
            for i in range(self.nprint):
                ln = str(i).rjust(n, ' ')
                r += self._fmt(ln, self._lines[i])
            r += '...\n'.rjust(n, ' ')
            for i in range(nn - self.nprint, nn):
                ln = str(i).rjust(n, ' ')
                r += self._fmt(ln, self._lines[i])
        else:
            for i, line in enumerate(self):
                ln = str(i).rjust(n, ' ')
                r += self._fmt(ln, line)
        return r


def lines_from_file(path, as_interned=False, encoding=None):
    """
    Create a list of file lines from a given filepath.

    Args:
        path (str): File path
        as_interned (bool): List of "interned" strings (default False)

    Returns:
        strings (list): File line list
    """
    lines = None
    with io.open(path, encoding=encoding) as f:
        if as_interned:
            lines = [sys.intern(line) for line in f.read().splitlines()]
        else:
            lines = f.read().splitlines()
    return lines


def lines_from_stream(f, as_interned=False):
    """
    Create a list of file lines from a given file stream.

    Args:
        f (io.TextIOWrapper): File stream
        as_interned (bool): List of "interned" strings (default False)

    Returns:
        strings (list): File line list
    """
    if as_interned:
        return [sys.intern(line) for line in f.read().splitlines()]
    return f.read().splitlines()


def lines_from_string(string, as_interned=False):
    """
    Create a list of file lines from a given string.

    Args:
        string (str): File string
        as_interned (bool): List of "interned" strings (default False)

    Returns:
        strings (list): File line list
    """
    if as_interned:
        return [sys.intern(line) for line in string.splitlines()]
    return string.splitlines()
