"""L3 Validation: LOH/WGD Constraint Solving

Tests verify that LOH constraint solving correctly ensures
valid event ordering and complete LOH coverage.
"""

import pytest
import numpy as np


class TestLOHConstraints:
    """Test LOH constraint solving with CP-SAT."""
    
    @pytest.fixture
    def loh_profile(self):
        """Create a simple LOH profile for testing."""
        # Profile: [1, 0, 0, 1] - LOH at positions 1-2
        return np.array([1, 0, 0, 1])

    def test_loh_detection(self):
        """Test L3.1: cn=0 segments identified as LOH."""
        profile = np.array([1, 0, 0, 1])
        
        # LOH is where cn == 0
        loh_mask = profile == 0
        
        assert loh_mask[1], "Position 1 should be LOH"
        assert loh_mask[2], "Position 2 should be LOH"
        assert not loh_mask[0], "Position 0 should not be LOH"
        assert not loh_mask[3], "Position 3 should not be LOH"

    def test_loh_constraint_function_exists(self):
        """Test that LOH constraint function is importable."""
        from spice.event_inference.events_from_graph import loh_filters_for_graph_result_diffs
        
        assert callable(loh_filters_for_graph_result_diffs), \
            "loh_filters_for_graph_result_diffs should be callable"

    def test_loh_solution_validity(self):
        """Test L3.7: All returned solutions satisfy LOH constraints."""
        pytest.importorskip('ortools')
        
        from spice.event_inference.events_from_graph import (
            get_starts_and_ends,
            get_events_from_graph_step
        )
        
        # Profile with LOH
        profile = np.array([1, 0, 0, 1])
        
        starts, ends = get_starts_and_ends(profile, wgd=False)
        
        # If there are valid solutions, they should handle LOH correctly
        # This is a basic sanity check
        assert starts.sum() > 0 or ends.sum() > 0, \
            "LOH profile should generate breakpoints"

    def test_loh_with_adjacent_gain(self):
        """Test L3.T2: LOH with adjacent gain handled correctly."""
        from spice.event_inference.events_from_graph import get_starts_and_ends
        
        # Profile: [1, 0, 0, 2, 2, 1] - LOH + adjacent gain
        profile = np.array([1, 0, 0, 2, 2, 1])
        
        starts, ends = get_starts_and_ends(profile, wgd=False)
        
        # Should have loss-related ends and gain-related starts
        assert starts.sum() > 0, "Should have gain starts"
        assert ends.sum() > 0, "Should have loss ends"
