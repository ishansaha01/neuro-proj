import logging
import subprocess
import sys

import numpy as np

from neurosim.fus.backends import BackendConfig, compute_bank
from neurosim.fus.backends.phantoms import homogeneous_medium


def test_importing_the_package_does_not_import_the_solver():
    code = "import sys, neurosim.fus.backends; print('kwave' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


def test_compute_bank_logs_the_cost_estimate_first(grid, soft, transducer, roi, caplog):
    medium = homogeneous_medium(grid, soft, alpha_power=1.5)
    with caplog.at_level(logging.INFO, logger="neurosim.fus.backends"):
        compute_bank(medium, transducer, roi, np.array([0.5e6]), BackendConfig(backend="ray"))
    first = caplog.records[0].getMessage()
    assert "backend 'ray'" in first and "estimated cost" in first and "run_time_s" in first
