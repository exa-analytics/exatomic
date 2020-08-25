# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
from unittest import TestCase
from exatomic.algorithms.geometry import make_small_molecule


class TestMakeSmallMolecule(TestCase):

    def test_2_domain(self):
        cen = ['C', 'O']
        lig = ['S', 'H']
        dis = [1.5, 1.1]
        geo = ['linear', 'bent']
        for i, (c, l, d, g) in enumerate(zip(cen, lig, dis, geo)):
            if i:
                with self.assertRaises(NotImplementedError):
                    df = make_small_molecule(c, l, d, g)
            else:
                df = make_small_molecule(c, l, d, g)
                self.assertTrue(df.shape[0], 3)

    def test_3_domain(self):
        for geo in ['trigonal_planar', 'trigonal_pyramidal', 't_shaped']:
            pass
        pass

    def test_4_domain(self):
        for geo in ['tetrahedral', 'square_planar', 'seesaw']:
            pass
        pass

    def test_5_domain(self):
        for geo in ['trigonal_bipyramidal', 'square_pyramidal']:
            pass
        pass

    def test_6_domain(self):
        for geo in ['octahedral']:
            pass
        pass

    def test_offset(self):
        pass

    def test_plane(self):
        pass

    def test_axis(self):
        pass
