import numpy as np
import pandas as pd
from unittest import TestCase
from os import sep

#from exatomic import va
from exatomic.base import resource
from exatomic.va import VA, get_data, gen_delta
from exatomic.gaussian import Fchk
from exatomic.nwchem import Output

class TestGetData(TestCase):
    def setUp(self):
        pass

    def test_getter_small(self):
        path = sep.join(resource('va-roa-h2o2-def2tzvp-514.5-00.out').split(sep)[:-1])+sep+'*'
        df = get_data(path=path, attr='roa', soft=Output, f_start='va-roa-h2o2-def2tzvp-514.5-',
                      f_end='.out')
        self.assertEqual(df.shape[0], 130)
        df = get_data(path=path, attr='gradient', soft=Output, f_start='va-roa-h2o2-def2tzvp-514.5-',
                      f_end='.out')
        self.assertEqual(df.shape[0], 52)

    def test_getter_large(self):
        path = sep.join(resource('va-roa-methyloxirane-def2tzvp-488.9-00.out').split(sep)[:-1])+sep+'*'
        df = get_data(path=path, attr='roa', soft=Output, f_start='va-roa-methyloxirane-def2tzvp-488.9-',
                      f_end='.out')
        self.assertEqual(df.shape[0], 160)
        df = get_data(path=path, attr='gradient', soft=Output, f_start='va-roa-methyloxirane-def2tzvp-488.9-',
                      f_end='.out')
        self.assertEqual(df.shape[0], 160)

class TestVROA(TestCase):
    def setUp(self):
        self.h2o2_freq = Fchk(resource('g16-h2o2-def2tzvp-freq.fchk'))
        self.methyloxirane_freq = Fchk(resource('g16-methyloxirane-def2tzvp-freq.fchk'))

    def test_vroa(self):
        self.h2o2_freq.parse_frequency()
        self.h2o2_freq.parse_frequency_ext()
        delta = gen_delta(delta_type=2, freq=self.h2o2_freq.frequency.copy())
        va_corr = VA()
        path = sep.join(resource('va-roa-h2o2-def2tzvp-514.5-00.out').split(sep)[:-1])+sep+'*'
        va_corr.roa = get_data(path=path, attr='roa', soft=Output, f_start='va-roa-h2o2-def2tzvp-514.5-',
                               f_end='.out')
        va_corr.roa['exc_freq'] = np.tile(514.5, len(va_corr.roa))
        va_corr.gradient = get_data(path=path, attr='gradient', soft=Output, 
                                    f_start='va-roa-h2o2-def2tzvp-514.5-', f_end='.out')
        va_corr.gradient['exc_freq'] = np.tile(514.5, len(va_corr.gradient))
        va_corr.vroa(uni=self.h2o2_freq, delta=delta['delta'].values)
        scatter_data = np.array([[ 3.47311779e+02,  0.00000000e+00, -3.27390198e+02,
                                  -8.44921542e+01, -4.22102267e-02, -3.41332079e-02,
                                  -3.91676006e-03,  5.14500000e+02],
                                 [ 8.60534577e+02,  1.00000000e+00,  1.75268228e+02,
                                  -8.09603043e+00, -8.26286589e+00,  1.65666769e-02,
                                  -3.01543530e-03,  5.14500000e+02],
                                 [ 1.24319010e+03,  2.00000000e+00, -3.35605422e+02,
                                  -7.26516978e+01,  1.06370293e-06, -3.45429748e-02,
                                  -4.20725882e-03,  5.14500000e+02],
                                 [ 1.37182002e+03,  3.00000000e+00,  2.20210484e+02,
                                   5.78999940e+01, -4.46257676e+00,  2.29930063e-02,
                                  -6.16087427e-04,  5.14500000e+02],
                                 [ 3.59750268e+03,  4.00000000e+00, -3.50819253e+03,
                                  -3.90826518e+02,  5.90325028e-02, -3.49292932e-01,
                                  -4.98353528e-02,  5.14500000e+02],
                                 [ 3.59821746e+03,  5.00000000e+00,  5.05236006e+03,
                                   4.00023286e+02,  6.60389814e+00,  4.97827311e-01,
                                   7.91921951e-02,  5.14500000e+02]])
        raman_data = np.array([[3.47311779e+02, 0.00000000e+00, 3.42307581e-05, 6.38955258e-01,
                                2.04527298e+01, 5.14500000e+02],
                               [8.60534577e+02, 1.00000000e+00, 1.90190339e-01, 1.07090867e+00,
                                6.85033386e+01, 5.14500000e+02],
                               [1.24319010e+03, 2.00000000e+00, 1.27606294e-08, 3.46909710e-01,
                                1.11011130e+01, 5.14500000e+02],
                               [1.37182002e+03, 3.00000000e+00, 3.94139588e-02, 1.19286864e+00,
                                4.52663091e+01, 5.14500000e+02],
                               [3.59750268e+03, 4.00000000e+00, 1.52822910e-02, 6.21166773e+00,
                                2.01524180e+02, 5.14500000e+02],
                               [3.59821746e+03, 5.00000000e+00, 1.60412161e+00, 5.19841596e+00,
                                4.55091201e+02, 5.14500000e+02]])
        self.assertTrue(np.allclose(va_corr.scatter.values, scatter_data))
        scatter_data = scatter_data.T
        raman_data = raman_data.T
        print(va_corr.scatter.to_string())
        # test all columns of the respective dataframe to get a better sense of what is broken
        self.assertTrue(np.allclose(va_corr.scatter['freq'].values,           scatter_data[0]))
        self.assertTrue(np.allclose(va_corr.scatter['freqdx'].values,         scatter_data[1]))
        self.assertTrue(np.allclose(va_corr.scatter['beta_g*1e6'].values,     scatter_data[2]))
        self.assertTrue(np.allclose(va_corr.scatter['beta_A*1e6'].values,     scatter_data[3]))
        self.assertTrue(np.allclose(va_corr.scatter['alpha_g*1e6'].values,    scatter_data[4]))
        self.assertTrue(np.allclose(va_corr.scatter['backscatter'].values,    scatter_data[5]))
        self.assertTrue(np.allclose(va_corr.scatter['forwardscatter'].values, scatter_data[6]))
        self.assertTrue(np.allclose(va_corr.scatter['exc_freq'].values,       scatter_data[7]))

        self.assertTrue(np.allclose(va_corr.raman['freq'].values,          raman_data[0]))
        self.assertTrue(np.allclose(va_corr.raman['freqdx'].values,        raman_data[1]))
        self.assertTrue(np.allclose(va_corr.raman['alpha_squared'].values, raman_data[2]))
        self.assertTrue(np.allclose(va_corr.raman['beta_alpha'].values,    raman_data[3]))
        self.assertTrue(np.allclose(va_corr.raman['raman_int'].values,     raman_data[4]))
        self.assertTrue(np.allclose(va_corr.raman['exc_freq'].values,      raman_data[5]))

    def test_select_freq(self):
        self.methyloxirane_freq.parse_frequency()
        self.methyloxirane_freq.parse_frequency_ext()
        delta = gen_delta(delta_type=2, freq=self.methyloxirane_freq.frequency.copy())
        va_corr = VA()
        path = sep.join(resource('va-roa-methyloxirane-def2tzvp-488.9-00.out').split(sep)[:-1])+sep+'*'
        va_corr.roa = get_data(path=path, attr='roa', soft=Output, 
                               f_start='va-roa-methyloxirane-def2tzvp-488.9-', f_end='.out')
        va_corr.roa['exc_freq'] = np.tile(488.9, len(va_corr.roa))
        va_corr.gradient = get_data(path=path, attr='gradient', soft=Output, 
                                    f_start='va-roa-methyloxirane-def2tzvp-488.9-', f_end='.out')
        va_corr.gradient['exc_freq'] = np.tile(488.9, len(va_corr.gradient))
        va_corr.vroa(uni=self.methyloxirane_freq, delta=delta['delta'].values)
        scatter_data = np.array([[ 1.12639199e+03,  1.00000000e+01, -6.15736884e+01,
                                  -1.53103521e+01, -1.68892383e-01, -6.40100535e-03,
                                  -8.61815897e-04,  4.88900000e+02],
                                 [ 1.15100631e+03,  1.10000000e+01, -1.06898371e+02,
                                  -2.76794343e+01,  3.53857297e+00, -1.11479855e-02,
                                   1.28026956e-03,  4.88900000e+02],
                                 [ 1.24937064e+03,  1.20000000e+01,  5.23431984e+01,
                                   5.17874012e+00, -8.24615516e+00,  5.19066673e-03,
                                  -5.18260038e-03,  4.88900000e+02],
                                 [ 1.37094149e+03,  1.30000000e+01,  3.49746537e+01,
                                  -9.43653998e+00, -8.72689935e-02,  3.05559747e-03,
                                   6.47745423e-04,  4.88900000e+02],
                                 [ 1.39064221e+03,  1.40000000e+01, -5.31532967e+01,
                                  -5.65151249e+00, -6.11336634e+00, -5.28356488e-03,
                                  -5.16165231e-03,  4.88900000e+02],
                                 [ 1.44754882e+03,  1.50000000e+01, -1.25064010e+02,
                                  -2.49033712e+01, -8.12931706e-02, -1.28030529e-02,
                                  -1.66110131e-03,  4.88900000e+02]])
        raman_data = np.array([[1.12639199e+03, 1.00000000e+01, 3.61528920e-04, 5.94192417e-01,
                                1.90792325e+01, 4.88900000e+02],
                               [1.15100631e+03, 1.10000000e+01, 3.36862711e-02, 1.62723253e-01,
                                1.12706729e+01, 4.88900000e+02],
                               [1.24937064e+03, 1.20000000e+01, 3.10356963e-01, 1.61670230e+00,
                                1.07598727e+02, 4.88900000e+02],
                               [1.37094149e+03, 1.30000000e+01, 6.63217766e-03, 2.37302109e-01,
                                8.78745947e+00, 4.88900000e+02],
                               [1.39064221e+03, 1.40000000e+01, 6.30361373e-02, 8.04145907e-01,
                                3.70791737e+01, 4.88900000e+02],
                               [1.44754882e+03, 1.50000000e+01, 5.36564516e-05, 1.07944901e+00,
                                3.45520265e+01, 4.88900000e+02]])
        scatter_data = scatter_data.T
        raman_data = raman_data.T

        # test all columns of the respective dataframe to get a better sense of what is broken
        self.assertTrue(np.allclose(va_corr.scatter['freq'].values,           scatter_data[0]))
        self.assertTrue(np.allclose(va_corr.scatter['freqdx'].values,         scatter_data[1]))
        self.assertTrue(np.allclose(va_corr.scatter['beta_g*1e6'].values,     scatter_data[2]))
        self.assertTrue(np.allclose(va_corr.scatter['beta_A*1e6'].values,     scatter_data[3]))
        self.assertTrue(np.allclose(va_corr.scatter['alpha_g*1e6'].values,    scatter_data[4]))
        self.assertTrue(np.allclose(va_corr.scatter['backscatter'].values,    scatter_data[5]))
        self.assertTrue(np.allclose(va_corr.scatter['forwardscatter'].values, scatter_data[6]))
        self.assertTrue(np.allclose(va_corr.scatter['exc_freq'].values,       scatter_data[7]))

        self.assertTrue(np.allclose(va_corr.raman['freq'].values,          raman_data[0]))
        self.assertTrue(np.allclose(va_corr.raman['freqdx'].values,        raman_data[1]))
        self.assertTrue(np.allclose(va_corr.raman['alpha_squared'].values, raman_data[2]))
        self.assertTrue(np.allclose(va_corr.raman['beta_alpha'].values,    raman_data[3]))
        self.assertTrue(np.allclose(va_corr.raman['raman_int'].values,     raman_data[4]))
        self.assertTrue(np.allclose(va_corr.raman['exc_freq'].values,      raman_data[5]))

