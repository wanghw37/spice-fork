#!/usr/bin/env python
"""Regression tests for sparse loci length-scale/type buckets."""

import numpy as np
import pandas as pd
import pytest

from spice import main_loci_functions
from spice.length_scales import DEFAULT_SEGMENT_SIZE_DICT
from spice.segmentation import create_events_in_segmentation
from spice.tsg_og import detection
from spice.tsg_og.detection import _optimize_selection_points, collect_data_per_length_scale
from spice.tsg_og.simulation import SelectionPoints, resimulate_events


def _make_sparse_chr22_events():
    rows = [
        ("gain", 20_000_000, 20_500_000),
        ("loss", 21_000_000, 21_600_000),
        ("gain", 22_000_000, 23_500_000),
        ("loss", 24_000_000, 25_600_000),
        ("gain", 26_000_000, 31_000_000),
        ("loss", 31_500_000, 37_500_000),
        ("gain", 20_000_000, 33_000_000),
    ]
    events = pd.DataFrame(rows, columns=["type", "start", "end"])
    events["chrom"] = "chr22"
    events["pos"] = "internal"
    events["plateau"] = "neither_left_nor_right"
    events["width"] = events["end"] - events["start"]
    events["sample"] = "sample_1"
    events["id"] = [f"id_{i}" for i in range(len(events))]
    return events[["sample", "id", "chrom", "type", "start", "end", "width", "pos", "plateau"]]


def _zero_signal_bootstrap_bounds(cur_chrom):
    empty_events = pd.DataFrame(columns=["chrom", "start", "end"])
    bounds = []
    for cur_length_scale in ["small", "mid1", "mid2", "large"]:
        zero_signal = (
            create_events_in_segmentation(
                empty_events,
                bin_df=DEFAULT_SEGMENT_SIZE_DICT[cur_length_scale],
                skip_tqdm=True,
            )
            .loc[cur_chrom]
            .sum(axis=1)
            .values
        )
        bounds.extend([(zero_signal.copy(), zero_signal.copy()) for _ in range(2)])
    return bounds


def _patch_lightweight_detection_helpers(monkeypatch):
    def _kernel(cur_widths, segment_size, n_kernel, which="locus"):
        kernel_len = int(np.ceil(int(2 * np.max(cur_widths)) / segment_size))
        return np.ones(max(1, kernel_len), dtype=float)

    def _height_multiplier(cur_widths, cur_chrom, cur_length_scale, cur_type, loci_width, **kwargs):
        empty_events = pd.DataFrame(columns=["chrom", "start", "end"])
        return (
            create_events_in_segmentation(
                empty_events,
                bin_df=DEFAULT_SEGMENT_SIZE_DICT[cur_length_scale],
                skip_tqdm=True,
            )
            .loc[cur_chrom]
            .sum(axis=1)
            .values
            + 1.0
        )

    monkeypatch.setattr(detection, "create_convolution_kernel", _kernel)
    monkeypatch.setattr(detection, "create_height_multiplier", _height_multiplier)
    monkeypatch.setattr(
        detection,
        "create_centromere_values",
        lambda *args, **kwargs: {},
    )


def test_collect_data_per_length_scale_returns_zero_signal_placeholders(monkeypatch):
    sparse_events = _make_sparse_chr22_events()
    _patch_lightweight_detection_helpers(monkeypatch)
    monkeypatch.setattr(
        detection,
        "get_signal_bootstrap_bounds",
        lambda *args, **kwargs: _zero_signal_bootstrap_bounds("chr22"),
    )

    data_per_length_scale = collect_data_per_length_scale(
        sparse_events,
        "chr22",
        assert_non_empty=False,
        loci_results_dir="/tmp",
        N_bootstrap=4,
        N_kernel=32,
    )

    empty_bucket = data_per_length_scale[("large", "loss")]
    assert empty_bucket["has_events"] is False
    assert empty_bucket["cur_widths"].size == 0
    assert np.all(empty_bucket["signals"] == 0)
    assert np.all(empty_bucket["signal_bounds"][0] == 0)
    assert np.all(empty_bucket["signal_bounds"][1] == 0)
    assert empty_bucket["kernel"] is None
    assert empty_bucket["height_multiplier"] is None
    assert empty_bucket["cur_loss_norm"] == 1.0


def test_collect_data_per_length_scale_still_raises_by_default(monkeypatch):
    sparse_events = _make_sparse_chr22_events()
    _patch_lightweight_detection_helpers(monkeypatch)
    monkeypatch.setattr(
        detection,
        "get_signal_bootstrap_bounds",
        lambda *args, **kwargs: _zero_signal_bootstrap_bounds("chr22"),
    )

    with pytest.raises(ValueError):
        collect_data_per_length_scale(
            sparse_events,
            "chr22",
            loci_results_dir="/tmp",
            N_bootstrap=4,
            N_kernel=32,
        )


def test_optimize_selection_points_does_not_modify_empty_bins(monkeypatch):
    sparse_events = _make_sparse_chr22_events()
    _patch_lightweight_detection_helpers(monkeypatch)
    monkeypatch.setattr(
        detection,
        "get_signal_bootstrap_bounds",
        lambda *args, **kwargs: _zero_signal_bootstrap_bounds("chr22"),
    )
    data_per_length_scale = collect_data_per_length_scale(
        sparse_events,
        "chr22",
        assert_non_empty=False,
        loci_results_dir="/tmp",
        N_bootstrap=4,
        N_kernel=32,
    )

    np.random.seed(0)
    base_selection_points_per_cluster = [
        [SelectionPoints(loci=[(30_000_000, 0)]) for _ in range(8)]
    ]
    allowed_fitness_change = np.zeros((8, 1), dtype=bool)
    allowed_fitness_change[7, 0] = True
    optimized_selection_points, _, _ = _optimize_selection_points(
        25,
        base_selection_points_per_cluster,
        data_per_length_scale,
        "chr22",
        best_loss=np.inf,
        allow_pos_change=False,
        allowed_fitness_change=allowed_fitness_change,
        up_down_order=[False],
        N_iterations_base=0,
    )

    assert optimized_selection_points[0][7][0].fitness == 0


def test_run_loci_assignment_skips_preprocessing_without_reference_loci(monkeypatch, tmp_path):
    sparse_events = _make_sparse_chr22_events()
    reference_loci_df = pd.DataFrame(
        [{"chrom": "chr1", "pos": 10_000_000, "type": "OG"}]
    )

    def _unexpected_call(*args, **kwargs):
        raise AssertionError("preprocessing should be skipped when no loci are defined")

    monkeypatch.setattr(
        main_loci_functions, "bootstrap_sampling_of_signal", _unexpected_call
    )
    monkeypatch.setattr(
        main_loci_functions, "collect_data_per_length_scale", _unexpected_call
    )

    selection_points, loci_widths = main_loci_functions.run_loci_assignment_per_chrom(
        reference_loci_df=reference_loci_df,
        cur_chrom="chr22",
        final_events_df=sparse_events,
        loci_results_dir=str(tmp_path),
    )

    assert selection_points is None
    assert loci_widths is None


def test_resimulate_events_returns_zero_signal_for_empty_widths():
    overlap_bins, signal = resimulate_events(
        np.array([], dtype=float),
        chrom_size=51_304_566,
        baseline_fitness=1,
        segment_size=200_000,
        remove_centromere_bound=False,
    )

    assert len(overlap_bins) == len(signal)
    assert np.all(signal == 0)
