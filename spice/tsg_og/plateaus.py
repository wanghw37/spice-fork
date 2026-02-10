import numpy as np
import pandas as pd

from cn_signatures.utils import get_logger


logger = get_logger(__name__)


def categorize_events_by_plateau_overlap(plateaus_df, cur_events, plateau_border_width=1e6):
    """
    Categorize events based on their overlap with plateaus.
    
    Parameters:
    -----------
    plateaus_df : pandas.DataFrame
        DataFrame containing plateau information with columns 'chrom', 'start', and 'end'
    final_events_df : pandas.DataFrame
        DataFrame containing events information with columns 'chrom', 'start', and 'end'
    plateau_border_width : float, optional
        Width of the border around plateaus to consider for overlaps (default: 1e6)
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'strict_overlap': Events that strictly overlap with plateaus
        - 'left_or_right_overlap': Events that overlap with plateaus on left or right side
        - 'no_strict_overlap': Events that don't strictly overlap with plateaus
        - 'no_overlap': Events that don't overlap with plateaus at all
    """
    
    all_plateau_events_strict = []
    all_plateau_events_left_or_right = []
    
    for _, cur_plateau in plateaus_df.iterrows():
        cur_chrom = cur_plateau['chrom']
        cur_start = cur_plateau['start']
        cur_end = cur_plateau['end']
        
        # Events that strictly overlap with the plateau (both start and end within bounds)
        cur_events_strict = cur_events.query(
            'chrom == @cur_chrom and '
            '(start < @cur_start+@plateau_border_width/2) and '
            '(start > @cur_start-@plateau_border_width/2) and '
            '(end > @cur_end-@plateau_border_width/2) and '
            '(end < @cur_end+@plateau_border_width/2)'
        )
        
        # Events that partially overlap with the plateau (either start OR end within bounds)
        cur_events_left_or_right = cur_events.query(
            'chrom == @cur_chrom and ('
            '(start < @cur_start+@plateau_border_width/2) and '
            '(start > @cur_start-@plateau_border_width/2) or '
            '(end > @cur_end-@plateau_border_width/2) and '
            '(end < @cur_end+@plateau_border_width/2)'
            ')'
        )

        all_plateau_events_strict.append(cur_events_strict)
        all_plateau_events_left_or_right.append(cur_events_left_or_right)
    
    # Combine all events
    all_plateau_events_strict = pd.concat(all_plateau_events_strict, axis=0)
    all_plateau_events_left_or_right = pd.concat(all_plateau_events_left_or_right, axis=0)

    # Remove duplicates
    all_plateau_events_strict = all_plateau_events_strict.loc[
        ~all_plateau_events_strict.index.duplicated(keep='first')]
    all_plateau_events_left_or_right = all_plateau_events_left_or_right.loc[
        ~all_plateau_events_left_or_right.index.duplicated(keep='first')]
    assert not all_plateau_events_strict.index.duplicated().any()
    assert not all_plateau_events_left_or_right.index.duplicated().any()

    # Find events that don't overlap with plateaus
    non_plateau_events = cur_events.loc[np.setdiff1d(
        np.arange(len(cur_events)), all_plateau_events_strict.index)]
    neither_left_nor_right_events = cur_events.loc[np.setdiff1d(
        np.arange(len(cur_events)), all_plateau_events_left_or_right.index)]
    
    cur_events['plateau'] = 'no_overlap'
    cur_events.loc[all_plateau_events_left_or_right.index, 'plateau'] = 'left_or_right_overlap'
    cur_events.loc[all_plateau_events_strict.index, 'plateau'] = 'strict_overlap'
    # cur_events.loc[non_plateau_events.index, 'plateau'] = 'no_strict_overlap'
    cur_events.loc[neither_left_nor_right_events.index, 'plateau'] = 'neither_left_nor_right'
    return cur_events
