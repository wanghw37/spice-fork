import os

import pandas as pd
from joblib import Parallel, delayed

from spice.utils import log_debug


def save_fail_reports(failed_reports, results_dir=None, logger=None):
    if results_dir is None:
        from spice import directories, config
        results_dir = directories['results_dir']
    if len(failed_reports) == 0:
        return pd.DataFrame(columns=['id', 'step', 'error', 'status'])
    df_fail = pd.DataFrame(failed_reports)
    for col in ['id', 'step', 'error', 'status']:
        if col not in df_fail.columns:
            df_fail[col] = None
    fail_path = os.path.join(results_dir, config['name'], 'failed_reports.tsv')
    df_fail[['id', 'step', 'error', 'status']].to_csv(fail_path, sep='\t', index=False)
    if logger is not None:
        logger.info(f"A total of {len(df_fail)} tasks failed during execution. "
                    f"Saved failure report with to {fail_path}")
    return df_fail


def step_aware_cleanup(results_dir, requested_steps=None):
    """Delete artifacts from the first requested step onward, including preprocessing.

    Steps order: ['preprocessing', 'split', 'all_solutions', 'disambiguate', 'large_chroms', 'combine']
    - Preprocessing artifacts under `directories.data_dir`:
      - `{name}_processed.tsv`
      - `{name}_processed_split.tsv`
    - Other artifacts under `results_dir` per WGD branch.
    """
    import shutil
    from spice import config, directories
    from spice.utils import get_logger
    logger = get_logger('utils')
    name = config.get('name')

    step_order = ['preprocessing', 'split', 'all_solutions', 'disambiguate', 'large_chroms', 'combine']
    if requested_steps and all(s not in step_order for s in (requested_steps or [])):
        log_debug(logger, 'No step-aware cleanup requested (no --keep-old).')
        return

    step_targets = {
        'preprocessing': [
            ('__data__', f'{name}_processed.tsv'),
            ('__data__', f'{name}_processed_split.tsv'),
        ],
        'split': [
            ('nowgd', 'chrom_data_full'),
            ('nowgd', 'chrom_data_large'),
            ('wgd', 'chrom_data_full'),
            ('wgd', 'chrom_data_large'),
            ('', 'WGD_status_major_cn.png'),
            ('', 'WGD_status_ploidy_loh.png'),
        ],
        'all_solutions': [
            ('nowgd', 'full_paths_multiple_solutions'),
            ('nowgd', 'full_paths_single_solution'),
            ('wgd', 'full_paths_multiple_solutions'),
            ('wgd', 'full_paths_single_solution'),
        ],
        'disambiguate': [
            ('nowgd', 'knn_solved_chroms'),
            ('wgd', 'knn_solved_chroms'),
        ],
        'large_chroms': [
            ('nowgd', 'mcmc_solved_chroms_large'),
            ('wgd', 'mcmc_solved_chroms_large'),
        ],
        'combine': [
            ('', 'final_events.tsv'),
            ('', 'summary.tsv'),
            ('', 'failed_reports.tsv'),
        ],
    }

    def delete_path(rel_root: str, rel_name: str):
        if rel_root == '__data__':
            data_dir = config['directories']['data_dir']
            target = os.path.join(data_dir, rel_name) if data_dir else None
        else:
            target = os.path.join(results_dir, rel_root, rel_name) if rel_root else os.path.join(results_dir, rel_name)
        if not target:
            return
        if os.path.isdir(target):
            log_debug(logger, f'Removing directory {target}')
            shutil.rmtree(target, ignore_errors=True)
        elif os.path.isfile(target):
            log_debug(logger, f'Removing file {target}')
            try:
                os.remove(target)
            except FileNotFoundError:
                pass

    start_idx = 0 if requested_steps is None else min(step_order.index(s) for s in requested_steps if s in step_order)

    targets = []
    for s in step_order[start_idx:]:
        targets.extend(step_targets.get(s, []))

    if targets:
        logger.info('Step-aware cleanup enabled (no --keep-old).')
        for rel_root, rel_name in targets:
            delete_path(rel_root, rel_name)
    else:
        logger.info('No mapped artifacts to clean for requested steps.')



def _run_batch(cur_ids, cores, desc, func, logger):
    """Run a batch of tasks either serially or in parallel."""
    n_jobs = cores if (cores is not None and cores > 1) else 1
    logger.info(f"{desc}: running on {n_jobs} core(s) for {len(cur_ids)} items")
    def _safe_func(cid):
        try:
            return func(cid)
        except Exception as e:
            logger.error(f"{desc}: failed for id '{cid}'", exc_info=False)
            return {"id": cid, "status": "failed", "error": str(e), "step": desc}
    if n_jobs == 1:
        results = []
        for i, cid in enumerate(cur_ids):
            logger.info(f'{desc}: {i+1} / {len(cur_ids)} finished ({100*i/len(cur_ids):.1f}%) - {cid}')
            results.append(_safe_func(cid))
        return results
    else:
        return Parallel(n_jobs=n_jobs)(delayed(_safe_func)(cid) for cid in cur_ids)

