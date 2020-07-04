import numpy as np
from unittest import TestCase
from exatomic.va import GenInput as gi, gen_delta
from exatomic.gaussian import Fchk
from exatomic.adf import Output
from exatomic.base import resource

class TestGenInput(TestCase):
    def setUp(self):
        self.h2o2 = Fchk(resource('g16-h2o2-def2tzvp-freq.fchk'))
        self.ch4 = Output(resource('adf-ch4-opt-freq.out'))

    def test_delta_small(self):
        delta_0 = np.array([[0.10591816, 0.],
                            [0.08949634, 1.],
                            [0.10522441, 2.],
                            [0.10582653, 3.],
                            [0.10712271, 4.],
                            [0.10699982, 5.]])
        delta_1 = np.array([[0.05947087, 0.],
                            [0.05947087, 1.],
                            [0.05947087, 2.],
                            [0.05947087, 3.],
                            [0.05947087, 4.],
                            [0.05947087, 5.]])
        delta_2 = np.array([[0.05671023, 0.],
                            [0.05960412, 1.],
                            [0.05672138, 2.],
                            [0.05669342, 3.],
                            [0.05189421, 4.],
                            [0.05188801, 5.]])
        delta_3 = np.array([[0.01000000, 0.],
                            [0.01000000, 1.],
                            [0.01000000, 2.],
                            [0.01000000, 3.],
                            [0.01000000, 4.],
                            [0.01000000, 5.]])
        self.h2o2.parse_frequency()
        test_0 = gen_delta(delta_type=0, freq=self.h2o2.frequency.copy())
        test_1 = gen_delta(delta_type=1, freq=self.h2o2.frequency.copy())
        test_2 = gen_delta(delta_type=2, freq=self.h2o2.frequency.copy())
        test_3 = gen_delta(delta_type=3, disp=0.01, freq=self.h2o2.frequency.copy())
        self.assertTrue(np.allclose(test_0.values, delta_0))
        self.assertTrue(np.allclose(test_1.values, delta_1))
        self.assertTrue(np.allclose(test_2.values, delta_2))
        self.assertTrue(np.allclose(test_3.values, delta_3))
        self.assertRaises(ValueError, gen_delta, delta_type=3, freq=self.h2o2.frequency.copy())

    def test_all_small(self):
        self.h2o2.parse_atom()
        self.h2o2.parse_frequency()
        self.h2o2.parse_frequency_ext()
        all_freq = gi(uni=self.h2o2, delta_type=2)
        self.assertEqual(all_freq.disp.shape[0], 52)
        self.assertTrue(np.allclose(np.concatenate([[0.], self.h2o2.frequency_ext['freq'].values]),
                                                all_freq.disp['modes'].drop_duplicates().values))

        self.ch4.parse_atom()
        self.ch4.parse_frequency()
        inputs = gi(uni=self.ch4, delta_type=2)
        self.assertEqual(inputs.disp.shape[0], 95)

    def test_select_freq(self):
        self.h2o2.parse_atom()
        self.h2o2.parse_frequency()
        self.h2o2.parse_frequency_ext()
        freq_1_2_3 = gi(uni=self.h2o2, delta_type=2, fdx=[1,2,3])
        self.assertEqual(freq_1_2_3.disp.shape[0], 28)
        self.assertTrue(np.allclose(np.concatenate([[0.], self.h2o2.frequency_ext.loc[[1,2,3],'freq'].values]),
                                    freq_1_2_3.disp['modes'].drop_duplicates().values))
        freq_4 = gi(uni=self.h2o2, delta_type=2, fdx=[4])
        self.assertEqual(freq_4.disp.shape[0], 12)
        self.assertTrue(np.allclose(np.concatenate([[0.], [self.h2o2.frequency_ext.loc[4,'freq']]]),
                                    freq_4.disp['modes'].drop_duplicates().values))

