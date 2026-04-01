# SPICE Configuration Reference

This document describes all parameters available in the SPICE configuration system.

SPICE uses a two-layer configuration:
1. **Default config** (`spice/objects/default_config.yaml`) — baseline values for all parameters
2. **User config** (specified with `--config`) — overrides any subset of the defaults

Each user config **must** define `name` and `directories.base_dir`. All other keys are optional.

---

## Minimal config example

```yaml
name: my_run
directories:
  base_dir: /path/to/project
input_files:
  copynumber: data/my_data.tsv
```

---

## `directories`

Controls where SPICE reads and writes files. Relative paths are resolved against `base_dir`.

| Key | Default | Description |
|-----|---------|-------------|
| `base_dir` | *(required)* | Root directory for the project; all relative paths are resolved against this |
| `data_dir` | `'data'` | Directory for input data files |
| `results_dir` | `'results'` | Directory for output results |
| `log_dir` | `'logs'` | Directory for log files |
| `plot_dir` | `'plots'` | Directory for plot output |

---

## `input_files`

Paths to input files. Relative paths are resolved against `base_dir`.

| Key | Default | Description |
|-----|---------|-------------|
| `copynumber` | *(required)* | Tab-separated copy-number input file |
| `wgd_status` | `null` | TSV with per-sample WGD status (`sample_id`, `wgd` columns). If omitted, WGD is inferred automatically |
| `xy_samples` | `null` | TSV with per-sample sex status (`sample_id`, `xy` columns). If omitted, XY is inferred from `chrY` presence |
| `sv` | `null` | Pickle file with structural variant calls for SV-constrained event matching |
| `reference_loci` | `spice/reference_loci/all_460_loci.tsv` | Reference loci file for loci assignment (TCGA 460-loci set by default) |

---

## `params` — Event Inference

### General

| Key | Default | Description |
|-----|---------|-------------|
| `logging_level` | `"INFO"` | Logging verbosity: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"` |
| `total_cn` | `False` | Set `True` for single-channel total copy-number input (uses `total_cn` column instead of `cn_a`/`cn_b`) |
| `wgd_inference_method` | `'major_cn'` | Method for inferring WGD status when no `wgd_status` file is provided. Options: `'major_cn'` (heuristic based on major CN), `'ploidy_loh'` (PCAWG-style ploidy + LOH fraction) |
| `skip_existing` | `False` | Skip computation if output files already exist |

### Event Enumeration Thresholds

| Key | Default | Description |
|-----|---------|-------------|
| `dist_limit` | `40` | Maximum number of events to consider per chromosome; chromosomes exceeding this are skipped |
| `full_path_dist_limit` | `9` | Chromosomes with more events than this threshold use MCMC instead of full-path enumeration |
| `full_path_high_mem_dist_limit` | `8` | Same threshold used for high-memory Snakemake rule (`full_path_mem_large`); set lower than `full_path_dist_limit` to reserve large memory only for the biggest jobs |
| `use_cache` | `True` | Use pre-computed lookup tables to speed up full-path enumeration (recommended) |

### KNN Disambiguation

| Key | Default | Description |
|-----|---------|-------------|
| `knn_k` | `250` | Number of nearest neighbours for KNN-based path disambiguation |
| `knn_single_width_bin` | `False` | If `True`, ignore event width when selecting KNN neighbours (faster but less accurate) |

### LOH and Timing

| Key | Default | Description |
|-----|---------|-------------|
| `all_loh_solutions` | `False` | If `True`, return all valid LOH solutions instead of the single best |
| `timing_solution_threshold` | `0.316` | Score threshold (1/√10) for accepting a timing solution |
| `skip_loh_check_for_large_chroms` | `False` | Skip LOH checks for chromosomes processed by MCMC. May reduce accuracy; useful when MCMC times out on WGD chromosomes |

### MCMC Parameters

| Key | Default | Description |
|-----|---------|-------------|
| `mcmc_max_T` | `-6` | Maximum MCMC temperature (log10 scale) |
| `mcmc_min_T` | `1` | Minimum MCMC temperature (log10 scale) |
| `mcmc_n_iterations_scale` | `200` | Scale factor for number of MCMC iterations; reduce (e.g., to `50`) to speed up at the cost of accuracy |
| `mcmc_stop_after_no_improvement` | `null` | Stop MCMC early if the score does not improve for this many consecutive iterations. `null` disables early stopping. Recommended value: `500` for faster runs |

### Time Limits (seconds, per chromosome)

| Key | Default | Description |
|-----|---------|-------------|
| `time_limit_all_solutions` | `null` | Time limit for full-path enumeration. Chromosomes that exceed this fall back to MCMC |
| `time_limit_loh_filters` | `null` | Time limit for LOH filtering step |
| `time_limit_mcmc` | `null` | Time limit for MCMC sampling. **Note:** no output is saved for chromosomes where MCMC is aborted |

### Structural Variant Matching

| Key | Default | Description |
|-----|---------|-------------|
| `sv_size_filter` | `1000` | Minimum SV size (bp) to include in matching |
| `sv_matching_threshold` | `10` | Distance threshold (genomic bins) for matching SVs to copy-number breakpoints |
| `sv_filter_dup_del` | `True` | Filter SV duplications and deletions during matching |

### Snakemake Resource Allocation

These parameters control memory and disk limits for Snakemake rule scheduling (relevant for cluster execution).

| Key | Default | Description |
|-----|---------|-------------|
| `knn_mem` | `16000` | Memory (MB) for KNN disambiguation jobs |
| `full_path_mem` | `16000` | Memory (MB) for standard full-path enumeration jobs |
| `full_path_mem_large` | `64000` | Memory (MB) for large full-path enumeration jobs (chromosomes with ≥ `full_path_high_mem_dist_limit` events) |
| `full_path_disk` | `16000` | Disk space (MB) for full-path caching |

---

## `loci_detection`

### Pipeline Control

| Key | Default | Description |
|-----|---------|-------------|
| `loci_steps` | `'fast'` | Which pipeline steps to run. Options: `'fast'` (6-step), `'default'` (full 15-step), a single step name, `'step+'` (from that step onwards), or a space-separated list of steps |
| `overwrite_preprocessing` | `False` | Force recalculation of bootstrap signals and `data_per_length_scale` even if cached files exist |

### Detection Settings

| Key | Default | Description |
|-----|---------|-------------|
| `N_loci` | `100` | Maximum number of loci to detect per run |
| `skip_up_down` | `False` | Skip TSG/OG (loss/gain) direction assignment |
| `use_original_rank` | `False` | Use the original detection rank instead of re-ranking after each pipeline step |
| `N_bootstrap` | `10` | Number of bootstrap signal samples. **For production use, set to 1000** |
| `N_kernel` | `1000` | Number of kernel samples for convolution kernel estimation. **For production use, set to 100000** |
| `detection_blocked_distance_th` | `5000000` | Minimum distance (bp) between blocked (centromeric) regions |
| `th_locus_prominence` | `1` | Prominence threshold for locus filtering |
| `remove_plateaus` | `True` | Remove events overlapping copy-number plateaus |
| `remove_chrY` | `True` | Exclude `chrY` from loci analysis |
| `drop_duplicates` | `True` | Remove duplicate event entries before processing |
| `use_observed_centromeres` | `True` | Use empirically observed centromere positions rather than reference annotation |

### Iteration Counts

These control the number of optimization iterations at each pipeline step. In most cases the defaults are appropriate; reduce them to speed up exploratory runs.

| Key | Default | Description |
|-----|---------|-------------|
| `detection_N_iterations_base` | `30` | Base iterations for the detection step |
| `detection_max_N_iterations` | `200` | Maximum iterations for detection |
| `detection_final_N_iterations` | `250` | Final-pass iterations for detection |
| `ranking_N_iterations` | `5` | Iterations for loci ranking |
| `flipping_N_iterations` | `110` | Iterations for up/down (OG/TSG) flipping |
| `flipping_N_iterations_single` | `10` | Iterations for single-locus flipping |
| `limiting_N_iterations_optim` | `100` | Iterations for the limiting optimization step |
| `optimizing_N_iterations_optimization` | `110` | Iterations for position optimization |
| `infer_widths_N_iterations` | `10` | Iterations for locus width inference |
| `merge_N_iterations_optim` | `100` | Iterations for merging optimization |
| `within_ci_N_iterations` | `100` | Iterations for within-CI filtering |
| `filter_N_iterations_optim` | `100` | Iterations for filter optimization |
| `final_limiting_N_iterations_optim` | `100` | Iterations for final limiting optimization |
| `N_bootstrap_for_widths` | `10` | Bootstrap iterations used specifically for locus width inference (must be ≤ `N_bootstrap`) |

### Loci Assignment

| Key | Default | Description |
|-----|---------|-------------|
| `loci_assignment_N_iterations` | `250` | Iterations for assignment optimization |
| `loci_assignment_within_ci_N_iterations` | `100` | Iterations for within-CI filtering during assignment |

### P-value Computation

| Key | Default | Description |
|-----|---------|-------------|
| `calculate_p_value` | `True` | Calculate p-values from scratch; if `False`, load from cache |
| `p_value_threshold` | `0.05` | Significance threshold for p-value filtering |
| `p_values_N_random` | `1000` | Number of random permutation samples for p-value calculation |
| `p_values_N_iterations` | `100` | Optimization iterations within p-value calculation |
| `post_p_value_N_iterations` | `250` | Optimization iterations after p-value-based filtering |

---

## Full default config

For reference, the complete default configuration is in `spice/objects/default_config.yaml`.
