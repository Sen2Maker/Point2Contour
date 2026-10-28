import unittest
import numpy as np
from postprocessing.reconstruct import validate


def empty_evidence():
    arrays = dict(pc=np.array([[0., 0, 0], [1, 0, 0]]), ep=np.empty((0, 3)), raw=np.empty((0, 2, 3)))
    rounds = {}
    for k in range(3):
        rounds.update({f'raw_{k}': np.empty((0, 2, 3)), f'mode_{k}': np.empty((0, 2, 3)),
                       f'candidate_i_{k}': np.empty(0, dtype=int), f'bin_{k}': np.empty(0, dtype=int)})
    return arrays, rounds


class EvidenceSchemaTests(unittest.TestCase):
    def test_empty_predictions_accepted(self):
        validate(*empty_evidence())

    def test_empty_point_cloud_refused(self):
        a, d = empty_evidence(); a['pc'] = np.empty((0, 3))
        with self.assertRaises(ValueError): validate(a, d)

    def test_nonfinite_input_refused(self):
        a, d = empty_evidence(); a['pc'][0, 0] = np.nan
        with self.assertRaises(ValueError): validate(a, d)

    def test_unaligned_round_refused(self):
        a, d = empty_evidence(); d['mode_0'] = np.ones((1, 2, 3))
        with self.assertRaises(ValueError): validate(a, d)

    def test_mixed_final_input_refused(self):
        a, d = empty_evidence(); a['raw'] = np.ones((1, 2, 3))
        with self.assertRaises(ValueError): validate(a, d)

    def test_noninteger_identity_refused(self):
        a, d = empty_evidence(); d['bin_0'] = np.empty(0, dtype=float)
        with self.assertRaises(ValueError): validate(a, d)


if __name__ == '__main__':
    unittest.main()
