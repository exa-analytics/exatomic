# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
from os import path
from tempfile import mkdtemp
from exa.util.utility import datetime_header, mkp, convert_bytes, get_internal_modules


def test_get_internal_modules():
    lst = get_internal_modules()
    assert len(lst) > 0
    assert lst[0].__name__.startswith("exa")


def test_convert_bytes():
    a, b = convert_bytes(2049)
    assert a >= 2.0
    assert b == "KiB"
    a, b = convert_bytes(10000000)
    assert a >= 9.5367
    assert b == "MiB"
    a, b = convert_bytes(10000000000)
    assert a >= 9.3132
    assert b == "GiB"
    a, b = convert_bytes(10000000000000)
    assert a >= 9.0949
    assert b == "TiB"


def test_mkp():
    dir_ = mkdtemp()
    pth = path.join(dir_, "tmp")
    mkp(pth)
    assert path.exists(pth)


def test_datetime_header():
    assert isinstance(datetime_header(), str)
