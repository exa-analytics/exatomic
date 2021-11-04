# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Disk I/O Utilities
#######################################
"""
import os
import tarfile
from exa import Editor


def read_tarball(path, shortkey=False, classes=Editor):
    """
    Read a (possibly compressed) tarball archive and return a dictionary of
    editors.

    .. code-block:: python

        eds = read_tarball(path, classes=MyEditor)    # All files read as type MyEditor
        # Only read the special file as type MyEditor (default to Editor)
        eds = read_tarball(path.bz2, classes={'specialfile': MyEditor})
        eds = read_tarball(path, classes=myfunc)      # Complex function that returns classes

    Args:
        path (str): Path to tarball archive
        keypath (bool): Full member path as key (true, default); file name as key (false)
        classes: Class, dictionary of classes, or callable to return class
    """
    editors = {}
    with tarfile.open(path) as tar:
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if f is None:
                continue
            if shortkey:
                name = os.path.basename(member.name)
            else:
                name = member.name
            if isinstance(classes, type):
                cls = classes
            elif isinstance(classes, dict):
                cls = classes.get(name, Editor)
            elif callable(classes):
                cls = classes(name)
            else:
                raise TypeError("Wrong type for classes argument (with type {})".format(type(classes)))
            editors[name] = cls(f.read().decode(), name=name)
    return editors

