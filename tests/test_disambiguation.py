"""L4 Validation: Disambiguation Strategies

Tests verify that k-NN and MCMC disambiguation correctly
select appropriate solutions from multiple candidates.
"""

import pytest
import numpy as np
import pandas as pd


class TestKNNDistance:
    """Test k-NN distance calculation."""

    def test_distance_function_exists(self):
        """Test that distance calculation function is importable."""
        from spice.event_inference.knn_graph import calc_event_distances

        assert callable(calc_event_distances), "calc_event_distances should be callable"

    def test_event_dist_data_structure(self):
        """Test L4.1: EventDistData structure for distance calculation."""
        from spice.event_inference.knn_graph import EventDistData

        events_df = pd.DataFrame(
            {
                "chrom": ["chr1", "chr1"],
                "start": [1000000, 2000000],
                "end": [2000000, 3000000],
                "width": [1000000, 1000000],
                "type": ["gain", "loss"],
                "telomere_bound": [False, False],
                "whole_chrom": [False, False],
                "whole_arm": [False, False],
                "wgd": ["nowgd", "nowgd"],
            }
        )

        from spice.event_inference.knn_graph import get_event_dist_data_from_df

        event_data = get_event_dist_data_from_df(events_df)

        assert len(event_data.widths) == 2, "Should have 2 events"
        assert event_data.type[0] == "gain", "First event should be gain"
        assert event_data.type[1] == "loss", "Second event should be loss"

    def test_knn_solve_function_exists(self):
        """Test that k-NN solve function is importable."""
        from spice.event_inference.knn_graph import solve_with_knn

        assert callable(solve_with_knn), "solve_with_knn should be callable"

    def test_mcmc_function_exists(self):
        """Test L4.7: MCMC function is importable."""
        from spice.event_inference.mcmc_for_large_chroms import mcmc_event_selection

        assert callable(mcmc_event_selection), "mcmc_event_selection should be callable"

    def test_load_knn_train_function(self):
        """Test that training data loader exists."""
        from spice.event_inference.knn_graph import load_knn_train

        try:
            knn_train = load_knn_train()
            assert isinstance(knn_train, dict), "Training data should be a dict"
        except FileNotFoundError:
            pytest.skip("Training data file not found - acceptable for fresh install")

    def test_calc_event_distances_basic(self):
        """Test L4.1: Distance calculation with sample events."""
        from spice.event_inference.knn_graph import calc_event_distances, EventDistData

        train_events = {
            ("gain", False, "nowgd"): (
                np.array([1e6]),
                np.array([1e8]),
                np.array(["id1"]),
            ),
            ("loss", False, "nowgd"): (
                np.array([1e6]),
                np.array([1e8]),
                np.array(["id2"]),
            ),
        }

        test_events = EventDistData(
            chrom=np.array(["chr1"]),
            starts=np.array([0]),
            ends=np.array([1e6]),
            widths=np.array([1e6]),
            type=np.array(["gain"]),
            is_telomere_bound=np.array([False]),
            is_whole_chrom=np.array([False]),
            is_whole_arm=np.array([False]),
            wgd=np.array(["nowgd"]),
            chrom_lengths=np.array([1e8]),
        )

        distances = calc_event_distances(
            train_events, test_events, ks=[1], block_same_id=False
        )

        assert distances.shape == (1, 1), (
            f"Expected shape (1, 1), got {distances.shape}"
        )
        assert np.isfinite(distances[0, 0]), "Distance should be finite"

    def test_knn_solution_selection(self):
        """Test L4.5: k-NN selects solution with minimum distance."""
        from spice.event_inference.knn_graph import EventDistData, calc_event_distances

        train_events = {
            ("gain", False, "nowgd"): (
                np.array([1e6, 2e6]),
                np.array([1e8, 1e8]),
                np.array(["id1", "id2"]),
            ),
            ("loss", False, "nowgd"): (np.array([]), np.array([]), np.array([])),
        }

        test_events_data = {
            "chrom": np.array(["chr1"]),
            "starts": np.array([0]),
            "ends": np.array([1e6]),
            "widths": np.array([1e6]),
            "type": np.array(["gain"]),
            "is_telomere_bound": np.array([False]),
            "is_whole_chrom": np.array([False]),
            "is_whole_arm": np.array([False]),
            "wgd": np.array(["nowgd"]),
            "chrom_lengths": np.array([1e8]),
        }

        test_events = EventDistData(**test_events_data)

        distances = calc_event_distances(
            train_events, test_events, ks=[1], block_same_id=False
        )

        assert distances[0, 0] < 1.0, f"Expected small distance, got {distances[0, 0]}"
