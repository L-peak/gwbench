"""Test-session setup.

Importing torch and MKL-backed numpy in the same process can load two OpenMP
runtimes on Windows.  Set the standard workaround before pytest collects test
modules that import torch.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
