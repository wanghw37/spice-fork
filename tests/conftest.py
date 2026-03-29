#!/usr/bin/env python
"""Pytest configuration and shared fixtures for SPICE tests."""

import os
import sys
import types
import pytest
from unittest.mock import MagicMock

# Add the project root to the Python path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)


def _setup_mocks():
    """Set up mocks for fstlib and medicc modules."""

    def mock_create_symbol_table(max_cn=8, separator="X"):
        symbols = []
        idx = 0
        for cn1 in range(max_cn + 1):
            for cn2 in range(max_cn + 1):
                symbols.append((idx, f"{cn1}{separator}{cn2}"))
                idx += 1
        return symbols

    class MockFst:
        def __init__(self):
            self._symbols = None

        def set_input_symbols(self, s):
            self._symbols = s

        def input_symbols(self):
            return self._symbols

        def output_symbols(self):
            return self._symbols

        def set_output_symbols(self, s):
            pass

        def add_state(self):
            return 0

        def add_states(self, n):
            pass

        def set_start(self, s):
            pass

        def start(self):
            return 0

        def set_final(self, s, w=0):
            pass

        def final(self, state):
            return 0

        def add_arc(self, state, arc):
            pass

        def add_arcs(self, state, arcs):
            pass

        def arcs(self, state):
            return iter([])

        def arcsort(self, label):
            return self

        def __invert__(self):
            return self

        def __mul__(self, other):
            return self

    mock_fstlib = MagicMock()
    mock_fstlib.Fst = MockFst
    mock_fstlib.intersect = lambda x, **kwargs: x
    mock_fstlib.disambiguate = lambda x: x
    mock_fstlib.prune = lambda x, weight=0: x
    mock_fstlib.shortestpath = lambda x, nshortest=1: x
    mock_fstlib.encode_determinize_minimize = lambda x: x
    mock_fstlib.factory = MagicMock()
    mock_fstlib.factory.from_string = lambda *args, **kwargs: MockFst()
    mock_fstlib.Semiring = MagicMock()
    mock_fstlib.Semiring.TROPICAL = "tropical"

    mock_medicc = MagicMock()
    mock_medicc.create_symbol_table = mock_create_symbol_table
    mock_medicc.create_nstep_fst = lambda n, fst: MockFst()
    mock_medicc.create_1step_del_fst = lambda *args, **kwargs: MockFst()
    mock_medicc.factory = MagicMock()
    mock_medicc.factory._get_int_cns_from_symbol_table = lambda st, sep: {}
    mock_medicc.factory.create_1step_del_fst = lambda *args, **kwargs: MockFst()
    mock_medicc.factory.create_loh_fst = lambda *args, **kwargs: MockFst()
    mock_medicc.factory.create_1step_WGD_fst = lambda *args, **kwargs: MockFst()
    mock_medicc.io = MagicMock()
    mock_medicc.io.read_fst = lambda no_wgd=False: MockFst()
    mock_medicc.tools = MagicMock()

    sys.modules["fstlib"] = mock_fstlib
    sys.modules["medicc"] = mock_medicc


def pytest_configure(config):
    """Configure pytest with custom markers."""
    _setup_mocks()
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


@pytest.fixture(scope="session")
def repo_root_dir():
    """Return the repository root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def example_data_exists(repo_root_dir):
    """Check if example data files exist."""
    example_data = os.path.join(repo_root_dir, "data", "example_data.tsv")
    return os.path.exists(example_data)


@pytest.fixture(scope="session")
def knn_train_data_exists(repo_root_dir):
    """Check if KNN training data exists."""
    knn_data = os.path.join(
        repo_root_dir, "spice", "objects", "train_events_sv_and_unamb.pickle"
    )
    return os.path.exists(knn_data)
