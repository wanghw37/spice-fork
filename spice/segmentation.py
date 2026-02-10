import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from cn_signatures import data_loaders
from cn_signatures.utils import CALC_NEW, create_chrom_type_pos_indices
from cn_signatures.length_scales import DEFAULT_SEGMENT_SIZE_DICT
from cn_signatures.data_loaders import format_chromosomes, load_segmentation
from spice.event_analysis.final_events import classify_event_position

CHROMS = ['chr' + str(x) for x in range(1, 23)] + ['chrX', 'chrY']
CHROM_LENS = data_loaders.load_chrom_lengths()

@CALC_NEW()
def calc_dat_segmentation(dat, breakpoint_dict, show_progress=False, cn_columns=['cn_a', 'cn_b']):
    # import here because the Cython code isn't always compiled
    from cn_signatures.cn_signatures_c.consistent_segmentation_cython import consistent_segmentation

    dat_segmentation = []
    for chrom in tqdm(CHROMS, disable=not show_progress):

        cur_dat = dat.query('chrom == @chrom', engine='python')
        if len(cur_dat) == 0:
            continue
        breakpoints = np.sort(np.unique(np.concatenate([cur_dat.eval('start').values,
                                                        cur_dat.eval('end + 1').values])))
        cur_breakpoints = {chrom: np.sort(np.unique(np.append(breakpoints, breakpoint_dict[chrom])))}

        chrom_dat_consistent = []
        for allele in cn_columns:
            cur_dat_consistent = consistent_segmentation(
                cur_dat[[allele]], column=allele, chrom_col='chrom', breakpoint_dict=cur_breakpoints,
                cython=True, skip_assertions=True, postprocessing=False, show_progress=False)

            cur_dat_consistent = cur_dat_consistent.loc[cur_dat_consistent.eval('end>start')]
            cur_dat_consistent = cur_dat_consistent.astype(float) * cur_dat_consistent.eval('end - start').values[:, None]
            cur_dat_consistent['bin'] = pd.cut(cur_dat_consistent.reset_index()['start']+1, bins=breakpoint_dict[chrom]).values
            cur_dat_consistent = cur_dat_consistent.reset_index().set_index(['chrom', 'bin']).drop(['start', 'end'], axis=1, level=0).groupby(['chrom', 'bin'], observed=False).sum().reset_index()
            cur_dat_consistent['start'] = cur_dat_consistent['bin'].apply(lambda x: x.left).astype(int)
            cur_dat_consistent['end'] = cur_dat_consistent['bin'].apply(lambda x: x.right).astype(int)
            cur_dat_consistent = cur_dat_consistent.set_index(['chrom', 'start', 'end']).drop('bin', axis=1, level=0)
            cur_dat_consistent = cur_dat_consistent.astype(float) / cur_dat_consistent.eval('end-1 - start').values[:, None]

            chrom_dat_consistent.append(cur_dat_consistent)
        chrom_dat_consistent = pd.concat(chrom_dat_consistent, axis=1)
        dat_segmentation.append(chrom_dat_consistent)
    dat_segmentation = pd.concat(dat_segmentation, axis=0)

    dat_segmentation = dat_segmentation.stack('sample_id', future_stack=True).reset_index()
    dat_segmentation['chrom'] = format_chromosomes(dat_segmentation['chrom'])
    dat_segmentation = dat_segmentation.set_index(['sample_id', 'chrom', 'start', 'end']).sort_index()

    return dat_segmentation


def create_segmentation(size):

    cur_breakpoint_dict = {
        chrom: np.append((np.arange(0, CHROM_LENS[chrom], size))[:-1], CHROM_LENS[chrom])
        for chrom in CHROMS}

    cur_segmentation = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(
            [(chrom, cur_breakpoint_dict[chrom][i], cur_breakpoint_dict[chrom][i+1]-1) 
             for chrom in CHROMS for i in range(len(cur_breakpoint_dict[chrom])-1)],
             names=['chrom', 'start', 'end']))
    
    return cur_segmentation


def create_events_in_segmentation(final_events_df, bin_df=100e3, skip_tqdm=False):
    """
    Map events to segmentation bins.

    Returns a DataFrame indexed like `bin_df.index` with a single column
    `'events'` containing lists of event indices from `final_events_df` that
    overlap each segmentation bin.
    """
    if hasattr(bin_df, '__int__'):
        bin_df = load_segmentation(bin_df)
    else:
        assert isinstance(bin_df, pd.DataFrame)
    events_in_segmentation = pd.DataFrame(
        0,
        index=bin_df.index,
        columns=['events']).sort_index()

    for cur_chrom in tqdm(final_events_df['chrom'].unique(), disable=skip_tqdm):
        cur_events = final_events_df.query('chrom == @cur_chrom').copy()
        if len(cur_events) == 0:
            continue
        events_starts = cur_events['start'].values
        events_ends = cur_events['end'].values
        cur_seg = bin_df.loc[[cur_chrom]].reset_index().copy()
        cur_seg['end'] += 1

        cur_breakpoints = np.unique(cur_seg[['start', 'end']].values.flat)
        events_starts_smaller = events_starts < cur_breakpoints[1:, None]
        events_ends_larger = events_ends > cur_breakpoints[:-1, None]

        output = np.matmul(np.logical_and(events_starts_smaller, events_ends_larger), np.ones(len(cur_events), dtype=int))
        events_in_segmentation.loc[cur_chrom, 'events'] = output

    return events_in_segmentation


def create_events_in_segmentation_full(final_events_df, segmentation=100e3, show_tqdm=False):
    if hasattr(segmentation, '__int__'):
        segmentation = create_segmentation(segmentation)
    else:
        assert isinstance(segmentation, pd.DataFrame)
    if 'pos' not in final_events_df.columns:
        final_events_df['pos'] = classify_event_position(final_events_df)

    all_segmented_events = []
    for cur_type in ['gain', 'loss']:
        for cur_pos in ['whole_chrom', 'whole_arm', 'telomere_bound', 'internal']:
            cur_events = final_events_df.loc[(final_events_df['type'] == cur_type) & (final_events_df['pos'] == cur_pos)]
            cur_segmented_events = create_events_in_segmentation(cur_events, segmentation)
            all_segmented_events.append(cur_segmented_events)
    combined_segmented_events = pd.concat(all_segmented_events, axis=1)
    combined_segmented_events.columns = pd.MultiIndex.from_product(
        [['gain', 'loss'], ['whole_chrom', 'whole_arm', 'telomere_bound', 'internal']],
        names=['type', 'pos'])
    return combined_segmented_events


def get_events_at_position(signal, bin_df, cur_chrom, position):
    """
    Get the signal value at a specific genomic position.
    
    Args:
        signal: Array of signal values (typically from events_in_segmentation.loc[cur_chrom].sum(axis=1).values)
        bin_df: DataFrame defining the segmentation bins (with MultiIndex: chrom, start, end)
        chrom: Chromosome name (e.g., 'chr1')
        position: Genomic position (integer)
    
    Returns:
        float: Signal value at the given position, or 0 if position not found
    """
    if hasattr(bin_df, '__int__'):
        bin_df = load_segmentation(bin_df)
    else:
        assert isinstance(bin_df, pd.DataFrame)
    try:
        # Get all bins for the chromosome
        chrom_bins = bin_df.loc[[cur_chrom]].reset_index()
        
        # Find the bin containing this position
        mask = (chrom_bins['start'] <= position) & (chrom_bins['end'] >= position)
        bin_indices = np.where(mask)[0]
        
        if len(bin_indices) == 0:
            return 0.0
        elif len(bin_indices) == 1:
            return float(signal[bin_indices[0]])
        else:
            # If multiple bins match (shouldn't happen with non-overlapping bins),
            # return the sum
            return float(signal[bin_indices].sum())
    except KeyError:
        # Chromosome not in the dataframe
        return 0.0


def get_events_at_position_all_ls(data_per_length_scale, cur_chrom, position):
    events_at_position = []
    for length_scale, bin_df in DEFAULT_SEGMENT_SIZE_DICT.items():
        for cur_type in ['gain', 'loss']:
            signal = data_per_length_scale[(length_scale, cur_type)]['signals']
            value = get_events_at_position(signal, bin_df, cur_chrom, position)
            events_at_position.append(value)
    return events_at_position
