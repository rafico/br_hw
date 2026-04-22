import os
import random
import unittest

import numpy as np

from tests.optional_deps import require_modules

require_modules("torch")

import torch

from utils_determinism import hash_config, seed_everything


class DeterminismTests(unittest.TestCase):
    def test_seed_everything_repeats_python_numpy_torch_sequences(self):
        seed_everything(51)
        seq_a = (
            random.random(),
            np.random.rand(4).tolist(),
            torch.rand(4).tolist(),
        )

        seed_everything(51)
        seq_b = (
            random.random(),
            np.random.rand(4).tolist(),
            torch.rand(4).tolist(),
        )

        self.assertEqual(seq_a[0], seq_b[0])
        self.assertEqual(seq_a[1], seq_b[1])
        self.assertEqual(seq_a[2], seq_b[2])
        self.assertEqual(os.environ["PYTHONHASHSEED"], "51")
        self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)

    def test_hash_config_is_stable_for_key_order_and_stringifies_values(self):
        cfg_a = {"b": 2, "a": np.int32(3)}
        cfg_b = {"a": np.int32(3), "b": 2}

        digest_a = hash_config(cfg_a)
        digest_b = hash_config(cfg_b)

        self.assertEqual(digest_a, digest_b)
        self.assertEqual(len(digest_a), 10)


if __name__ == "__main__":
    unittest.main()
