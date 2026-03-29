"""L1 Validation: FST Event Counting

Tests verify that FST-based event counting correctly computes
the minimum number of events from copy-number profiles.
"""

import pytest
import numpy as np

pytest.importorskip("fstlib")


class TestFSTEventCounting:
    """Test FST-based event count calculation."""

    @pytest.fixture
    def fst_assets(self):
        """Load FST assets for testing."""
        from spice.event_inference.fst_assets import (
            SYMBOL_TABLE,
            SYMBOL_DICT,
            fsa_from_string,
            T_forced_WGD,
            T_noWGD,
            T,
            get_diploid_fsa,
        )

        return {
            "symbol_table": SYMBOL_TABLE,
            "symbol_dict": SYMBOL_DICT,
            "T_forced_WGD": T_forced_WGD,
            "T_noWGD": T_noWGD,
            "T": T,
            "diploid_fsa_nowgd": get_diploid_fsa(total_copy_numbers=False),
            "diploid_fsa_wgd": get_diploid_fsa(total_copy_numbers=False),
            "fsa_from_string": fsa_from_string,
        }

    def test_single_gain_event(self, fst_assets):
        """Test L1.T1: Single gain produces 1 event."""
        import fstlib

        profile_str = "X1X2X2X1X"
        fsa = fst_assets["fsa_from_string"](profile_str)

        score = fstlib.score(
            fst_assets["T_noWGD"], fst_assets["diploid_fsa_nowgd"], fsa
        )

        n_events = score - 1
        assert n_events == 1, f"Expected 1 event, got {n_events}"

    def test_single_loss_event(self, fst_assets):
        """Test L1.T2: Single loss produces 1 event."""
        import fstlib

        profile_str = "X1X0X0X1X"
        fsa = fst_assets["fsa_from_string"](profile_str)

        score = fstlib.score(
            fst_assets["T_noWGD"], fst_assets["diploid_fsa_nowgd"], fsa
        )

        n_events = score - 1
        assert n_events == 1, f"Expected 1 event, got {n_events}"

    def test_multiple_events(self, fst_assets):
        """Test L1.T3: Multiple events counted correctly."""
        import fstlib

        profile_str = "X1X2X2X1X0X0X1X"
        fsa = fst_assets["fsa_from_string"](profile_str)

        score = fstlib.score(
            fst_assets["T_noWGD"], fst_assets["diploid_fsa_nowgd"], fsa
        )

        n_events = score - 1
        assert n_events == 2, f"Expected 2 events, got {n_events}"

    def test_wgd_gain_event(self, fst_assets):
        """Test L1.T4: WGD profile with gain event."""
        import fstlib
        from spice.event_inference.fst_assets import get_diploid_fsa

        profile_str = "X2X4X4X2X"
        fsa = fst_assets["fsa_from_string"](profile_str)

        diploid_wgd = get_diploid_fsa(total_copy_numbers=False)

        score = fstlib.score(fst_assets["T_forced_WGD"], diploid_wgd, fsa)

        n_events = score - 1
        assert n_events >= 1, f"Expected at least 1 event, got {n_events}"

    def test_neutral_profile(self, fst_assets):
        """Test L1.T5: Neutral profile has 0 events."""
        import fstlib

        profile_str = "X1X1X1X1X"
        fsa = fst_assets["fsa_from_string"](profile_str)

        score = fstlib.score(
            fst_assets["T_noWGD"], fst_assets["diploid_fsa_nowgd"], fsa
        )

        n_events = score - 1
        assert n_events == 0, f"Expected 0 events for neutral profile, got {n_events}"

    def test_symbol_table_consistency(self, fst_assets):
        """Test L1.8: Symbol table max_cn=8, separator='X'."""
        from spice.event_inference.fst_assets import SYMBOL_TABLE, separator, max_cn

        assert separator == "X", f"Expected separator 'X', got {separator}"
        assert max_cn == 8, f"Expected max_cn=8, got {max_cn}"

        symbols = [
            fst_assets["symbol_dict"][i] for i in range(len(fst_assets["symbol_dict"]))
        ]
        assert "X" in symbols, "Separator 'X' not in symbol table"
        assert "0" in symbols, "'0' not in symbol table"
        assert "1" in symbols, "'1' not in symbol table"
