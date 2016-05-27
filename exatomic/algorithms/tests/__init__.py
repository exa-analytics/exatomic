# -*- coding: utf-8 -*-
from atomic.algorithms.tests.test_nonjitted import TestNonjitted
try:
    from atomic.algorithms.tests.test_jitted import TestJitted
except:
    pass
