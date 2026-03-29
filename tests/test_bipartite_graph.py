"""L2 Validation: Bipartite Graph Construction

Tests verify that copy-number profiles are correctly encoded
as bipartite graphs with proper start/end breakpoint identification.
"""

import pytest
import numpy as np

from spice.event_inference.events_from_graph import get_starts_and_ends


class TestBipartiteGraph:
    """Test bipartite graph construction from CN profiles."""

    def test_start_breakpoint_single_gain(self):
        """Test L2.1: CN increase produces start node."""
        profile = np.array([1, 2, 2, 1])

        starts, ends = get_starts_and_ends(profile, loh_adjust=False)

        assert len(starts) == 1, f"Expected 1 start total, got {len(starts)}"
        assert starts[0] == 1, f"Expected start at breakpoint 1, got {starts}"

    def test_end_breakpoint_single_gain(self):
        """Test L2.2: CN decrease produces end node."""
        profile = np.array([1, 2, 2, 1])

        starts, ends = get_starts_and_ends(profile, loh_adjust=False)

        assert len(ends) == 1, f"Expected 1 end total, got {len(ends)}"
        assert ends[0] == 3, f"Expected end at breakpoint 3, got {ends}"

    def test_magnitude_encoding(self):
        """Test L2.3: Change magnitude k produces k nodes."""
        profile = np.array([1, 3, 3, 1])

        starts, ends = get_starts_and_ends(profile, loh_adjust=False)

        assert len(starts) == 2, f"Expected 2 starts for magnitude 2, got {len(starts)}"
        assert np.all(starts == 1), f"Expected all starts at breakpoint 1, got {starts}"
        assert len(ends) == 2, f"Expected 2 ends for magnitude 2, got {len(ends)}"
        assert np.all(ends == 3), f"Expected all ends at breakpoint 3, got {ends}"

    def test_gain_and_loss_combination(self):
        """Test L2.T3: Gain + loss correctly identified."""
        profile = np.array([1, 2, 0, 1])

        starts, ends = get_starts_and_ends(profile, loh_adjust=False)

        assert 1 in starts, f"Expected gain start at breakpoint 1, got {starts}"
        loss_ends = np.sum(ends == 2)
        assert loss_ends == 2, f"Expected 2 loss ends at breakpoint 2, got {loss_ends}"

    def test_wgd_mode_prior_profile(self):
        """Test L2.6: WGD mode uses prior_profile=2."""
        profile = np.array([2, 4, 4, 2])

        starts, ends = get_starts_and_ends(
            profile, wgd=True, total_cn=True, loh_adjust=False
        )

        assert len(starts) == 2, f"Expected 2 starts in WGD mode, got {len(starts)}"
        assert len(ends) == 2, f"Expected 2 ends in WGD mode, got {len(ends)}"

    def test_non_wgd_mode_prior_profile(self):
        """Test L2.7: Non-WGD mode uses prior_profile=1."""
        profile = np.array([1, 2, 2, 1])

        starts, ends = get_starts_and_ends(profile, loh_adjust=False)

        assert len(starts) == 1, f"Expected 1 start in non-WGD mode, got {len(starts)}"
        assert len(ends) == 1, f"Expected 1 end in non-WGD mode, got {len(ends)}"
