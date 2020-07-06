# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0

# from ..numerical import fac, fac2, dfac21, _CFunction, _SFunction
#
#
# class TestNumerical(TestCase):
#     def setUp(self):
#         self.ns = range(-2, 10)
#
#     def test_fac(self):
#         chk = np.array([0, 0, 1, 1, 2, 6, 24, 120,
#                         720, 5040, 40320, 362880])
#         for c, n in zip(chk, self.ns):
#             self.assertEqual(fac(n), c)
#
#     def test_fac2(self):
#         chk = np.array([0, 1, 1, 1, 2, 3, 8,
#                         15, 48, 105, 384, 945])
#         for c, n in zip(chk, self.ns):
#             self.assertEqual(fac2(n), c)
#
#     def test_dfac21(self):
#         for n in self.ns:
#             self.assertEqual(fac2(2*n-1), dfac21(n))
#
#
# class TestNormalization(TestCase):
#     def setUp(self):
#         self.alphas = []
#
#     def test_normalize(self):
#         pass
#
#     def test_prim_normalize(self):
#         pass
#
#     def test_sto_normalize(self):
#         pass
#
# class TestBasisFunctions(TestCase):
#
#     def setUp(self):
#         # self.bargs = (0, 0, 0, 0, [1.], [1.])
#         self.cargs = (0, 0, 0, 0, [1.], [1.], 0, 0, 0)
#         self.sargs = (0, 0, 0, 0, [1.], [1.], 0, 0)
#         # Add gaussian{True/False}
#
#     # def test_abstract(self):
#     #     with self.assertRaises(TypeError) as ctx:
#     #         BasisFunction(*self.bargs)
#
#     def test_cartesian(self):
#         f = _CFunction(*self.cargs)
#         N = _prim_cart_norm(f.alphas, f.l, f.m, f.n)
#         self.assertTrue(np.allclose(f.Ns, N))
#
#     # def test_sto(self):
#     #     f = CartesianBasisFunction(*self.cargs, gaussian=False)
#     #     N = _prim_sto_norm(f.alphas, f.rpow)
#     #     self.assertTrue(np.allclose(f.Ns, N))
#
#     def test_spherical(self):
#         f = _SFunction(*self.sargs)
#         N = _prim_sphr_norm(f.alphas, f.L)
#         self.assertTrue(np.allclose(f.Ns, N))
