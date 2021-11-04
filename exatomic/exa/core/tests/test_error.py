# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
from exa.core.error import ExaException, RequiredColumnError


def test_exceptions():
    e = ExaException("test")
    assert "test" in str(e)
    e = RequiredColumnError("col", "DataFrame")
    assert RequiredColumnError._msg.format("col", "DataFrame") == str(e)
