"""CPU-only boundary tests; no model or training data is loaded."""
import unittest
from checkpoint_state import next_training_epoch, completed_epoch_fields


class CheckpointBoundaryTests(unittest.TestCase):
    def test_legacy_checkpoint_resumes_unstarted_epoch(self):
        for epoch in (0, 10, 20, 80):
            self.assertEqual(next_training_epoch(dict(epoch=epoch)), epoch)

    def test_new_checkpoint_resumes_next_epoch(self):
        for epoch in range(90):
            self.assertEqual(next_training_epoch(completed_epoch_fields(epoch)), epoch + 1)

    def test_ninety_epochs_without_skipping(self):
        for epoch in range(90):
            completed = list(range(epoch + 1))
            remaining = list(range(next_training_epoch(completed_epoch_fields(epoch)), 90))
            self.assertEqual(completed + remaining, list(range(90)))

    def test_inconsistent_metadata_refused(self):
        checkpoint = completed_epoch_fields(10)
        checkpoint['next_epoch'] = 12
        with self.assertRaises(ValueError):
            next_training_epoch(checkpoint)

    def test_unknown_phase_refused(self):
        with self.assertRaises(ValueError):
            next_training_epoch(dict(epoch=10, checkpoint_phase='unknown'))


if __name__ == '__main__':
    unittest.main()
