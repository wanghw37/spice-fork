from functools import reduce
import itertools

import numpy as np

from spice import data_loaders
from spice.length_scales import LS_I_DICT, DEFAULT_SEGMENT_SIZE_DICT, DEFAULT_LENGTH_SCALE_BOUNDARIES
from spice.utils import get_logger

logger = get_logger(__name__)
CHROM_LENS = data_loaders.load_chrom_lengths()
CENTROMERES_OBSERVED = data_loaders.load_centromeres(observed=True, extended=False)


def calc_event_rate_per_loci(
    final_events_df,
    cur_selection_points,
    cur_chrom,
    cur_length_scale,
    cur_type,
    length_scale_boundaries=DEFAULT_LENGTH_SCALE_BOUNDARIES,
):

    length_scale_i = LS_I_DICT[(cur_length_scale, cur_type)]
    
    cur_length_scale_border = length_scale_boundaries[cur_length_scale]
    cur_events = final_events_df.query('pos == "internal" and type == @cur_type and chrom == @cur_chrom and width > @cur_length_scale_border[0] and width <= @cur_length_scale_border[1]').copy()
    if len(cur_events) == 0:
        return np.array([]), np.array([])

    loci_pos = np.array([x[0].pos for x in cur_selection_points[0]])
    loci_fitness = np.maximum(0, np.array([x[0].fitness for x in cur_selection_points[length_scale_i]]))

    event_locus_overlaps = np.logical_and(
        cur_events['start'].values[:, None] < loci_pos,
        cur_events['end'].values[:, None] > loci_pos)
    event_locus_overlaps_norm = (
        (event_locus_overlaps.astype(int)) *
        loci_fitness[None, :] /
        (((event_locus_overlaps.astype(int)) * loci_fitness[None, :]).sum(axis=1) + 1)[:, None])
    assert np.min(event_locus_overlaps_norm) >= 0

    # add baseline rates
    event_locus_overlaps_norm = np.concatenate([event_locus_overlaps_norm, (1 - event_locus_overlaps_norm.sum(axis=1))[:, None]], axis=1)
    assert np.max(np.abs(event_locus_overlaps_norm.sum(axis=1) - 1)) < 1e-5, 'Normalization error'

    return event_locus_overlaps_norm.T

    
def calc_event_rate_per_loci_all_ls(
    final_events_df,
    cur_selection_points,
    cur_chrom,
    mode='mix',
):
    rates_for_all_ls = dict()
    for (cur_length_scale, cur_type) in itertools.product(['small', 'mid1', 'mid2', 'large'], ['gain', 'loss']):
        events_per_loci = calc_event_rate_per_loci(
            final_events_df,
            cur_selection_points,
            cur_chrom,
            cur_length_scale,
            cur_type
        )
        if len(events_per_loci) == 0:
            rates_for_all_ls[cur_length_scale, cur_type] = (0, np.zeros(len(cur_selection_points[0]) + 1))
        else:
            if len(events_per_loci[0]) == 0:
                rates_for_all_ls[cur_length_scale, cur_type] = 0
            else:   
                rates_for_all_ls[cur_length_scale, cur_type] = (events_per_loci.shape[1], events_per_loci.mean(axis=1))

    return rates_for_all_ls


def calc_total_events_per_loci(
    cur_chrom,
    rates_for_all_ls=None,
    final_events_df=None,
    cur_selection_points=None,
):
    if rates_for_all_ls is None:
        rates_for_all_ls = calc_event_rate_per_loci_all_ls(
            final_events_df,
            cur_selection_points,
            cur_chrom)
    total_events_per_loci = dict()
    for key in rates_for_all_ls.keys():
        if rates_for_all_ls[key] == 0:
            total_events_per_loci[key] = 0
        else:
            total_events_per_loci[key] = rates_for_all_ls[key][0] * rates_for_all_ls[key][1]

    return total_events_per_loci