import unittest
import numpy as np
from postprocessing.evidence import select_latest_responses, support_fraction


class EvidenceTests(unittest.TestCase):
    def item(self, ids, value):
        n = len(ids)
        return dict(query_ids=np.array(ids, dtype=int), direction_ids=np.zeros(n, dtype=int),
                    segments=np.full((n, 2, 3), value), mode_segments=np.full((n, 2, 3), value + 10))

    def test_latest_and_alignment(self):
        ordinary, modal, source = select_latest_responses([self.item([1, 2], 1), self.item([1], 2)])
        np.testing.assert_array_equal(ordinary[:, 0, 0], [2, 1])
        np.testing.assert_array_equal(modal[:, 0, 0], [12, 11])
        self.assertEqual(source, [(1, 0), (0, 1)])

    def test_empty(self):
        self.assertEqual(select_latest_responses([])[0].shape, (0, 2, 3))
        self.assertEqual(support_fraction([[0, 0, 0], [1, 0, 0]], np.empty((0, 2, 3))), 0)

    def test_duplicates_refused(self):
        with self.assertRaises(ValueError):
            select_latest_responses([self.item([1, 1], 0)])

    def test_support_and_reversal(self):
        line = np.array([[0., 0, 0], [1, 0, 0]])
        self.assertEqual(support_fraction(line, line[None, ::-1]), 1)
        self.assertEqual(support_fraction(line, (line + [0, 1, 0])[None]), 0)

    def test_finite_not_infinite_lines(self):
        self.assertLess(support_fraction([[0, 0, 0], [10, 0, 0]], np.array([[[0, 0, 0], [1, 0, 0]]])), .65)

    def test_nonfinite_refused(self):
        item = self.item([1], 0.)
        item['segments'][0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            select_latest_responses([item])


if __name__ == '__main__':
    unittest.main()
