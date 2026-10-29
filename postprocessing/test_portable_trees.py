import unittest
import numpy as np
from ._portable_trees import FrozenTrees


class TreeTests(unittest.TestCase):
    def model(self):
        dtype=[('value','f8'),('feature_idx','u4'),('threshold','f8'),
               ('missing_go_to_left','u1'),('left','u4'),('right','u4'),('is_leaf','u1')]
        nodes=np.zeros(3,dtype=dtype)
        nodes[0]=(0,0,0.5,1,1,2,0)
        nodes[1]=(-2,0,0,0,0,0,1)
        nodes[2]=(3,0,0,0,0,0,1)
        return FrozenTrees({'a_baseline':1.,'a_features':1,'a_classes':np.array([0,1]),
                            'a_count':1,'a_tree_0':nodes},'a')

    def test_boundary_and_missing(self):
        actual=self.model().decision_function([[.5],[np.nextafter(.5,np.inf)],[np.nan]])
        np.testing.assert_array_equal(actual,[-1,4,-1])

    def test_probabilities(self):
        p=self.model().predict_proba([[0],[1]])
        np.testing.assert_array_equal(p.sum(1),[1,1])
        self.assertTrue(p[0,1]<.5<p[1,1])

    def test_empty(self):
        self.assertEqual(self.model().predict_proba(np.empty((0,1))).shape,(0,2))

    def test_shape(self):
        with self.assertRaises(ValueError): self.model().predict([[1,2]])

    def test_classifier_labels(self):
        np.testing.assert_array_equal(self.model().predict([[0],[1],[np.nan]]),[0,1,0])

    def test_tied_probability_uses_first_class(self):
        model=self.model(); model.baseline=2.
        np.testing.assert_array_equal(model.predict([[0]]),[0])

    def test_regression_values_unchanged(self):
        model=self.model(); model.classes_=np.array([])
        np.testing.assert_array_equal(model.predict([[0],[1],[np.nan]]),[-1,4,-1])
        with self.assertRaises(ValueError): model.predict_proba([[0]])

    def test_empty_labels(self):
        self.assertEqual(self.model().predict(np.empty((0,1))).shape,(0,))


if __name__=='__main__': unittest.main()
