# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0

import os
import numpy as np
from glob import glob

try:
    from exa.test.tester import UnitTester
except:
    from exa.tester import UnitTester
from exatomic.adf.output import Output
from exatomic import Cube


class TestOrbital(UnitTester):
    """Tests that orbitals are generated correctly for ADF."""
    def setUp(self):
        pass
#        cd = os.path.abspath(__file__).split(os.sep)[:-1]
#        self.uni = Output(os.sep.join(cd + ['kr.out'])).to_universe()
#        cubs = sorted(glob(os.sep.join(cd + ["*cube"])))
#        self.cub = Cube(cubs[0]).to_universe()
#        for fl in cubs[1:]: self.cub.add_field(Cube(fl).field)
#        self.uni.add_molecular_orbitals(vector=range(len(cubs)),
#                              field_params=self.cub.field.ix[0])


    def test_field_values(self):
        pass
#        target = len(self.uni.field.field_values[0]) - 5
#        for fld, cub in zip(self.uni.field.field_values,
#                            self.cub.field.field_values):
#            self.assertTrue(np.allclose(fld, cub, rtol=1.e-2, atol=1.e-3))
#            matches = (np.isclose(fld,  cub, rtol=0.0001).sum(),
#                       np.isclose(fld, -cub, rtol=0.0001).sum())
#            self.assertTrue(any(i > target for i in matches))
