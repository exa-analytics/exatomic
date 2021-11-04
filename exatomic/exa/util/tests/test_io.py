# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
import tarfile
from os import remove, rmdir, path, makedirs
from tempfile import mkdtemp
from exa.util.io import read_tarball


def test_tarball():
    tmpdir = mkdtemp()
    archive_dir = path.join(tmpdir, "tmp")
    makedirs(archive_dir)
    with open(path.join(archive_dir, "file.txt"), "w") as f:
        f.write("hello {value}")
    archive = path.join(tmpdir, "tmp.tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(archive_dir)
    eds = read_tarball(archive, shortkey=True)
    assert len(eds) == 1
    ed = eds["file.txt"]
    assert ed.variables == ["{value}"]
    assert str(ed) == "hello {value}"
    remove(path.join(archive_dir, "file.txt"))
    remove(archive)
    rmdir(path.join(tmpdir, "tmp"))
    rmdir(tmpdir)
