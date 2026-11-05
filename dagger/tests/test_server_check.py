import contextlib
import io
import json
import os
import unittest
from unittest.mock import patch

import torch

from dagger.server_check import main


class ServerCheckTests(unittest.TestCase):
    def test_unselected_gpus_are_never_allocated(self):
        output = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), patch("sys.argv", ["server_check", "--cuda"]), \
             patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "device_count", return_value=8), \
             patch.object(torch, "ones", side_effect=AssertionError("unallocated GPU touched")) as allocate, \
             contextlib.redirect_stdout(output):
            self.assertEqual(main(), 1)
        allocate.assert_not_called()
        result = json.loads(output.getvalue().splitlines()[-1])
        self.assertIn("gpu-selection", result["failed"])
        self.assertEqual(result["isaac_simulator_smoke"], "not_run")

    def test_missing_data_reports_failure(self):
        output = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), patch("sys.argv", ["server_check", "--data"]), \
             contextlib.redirect_stdout(output):
            self.assertEqual(main(), 1)
        result = json.loads(output.getvalue().splitlines()[-1])
        self.assertFalse(result["preflight_pass"])
        self.assertIn("dataset-runtime", result["failed"])


if __name__ == "__main__":
    unittest.main()
