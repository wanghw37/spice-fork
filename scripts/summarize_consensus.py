import sys
import os
import pandas as pd


def main():
    if len(sys.argv) < 4:
        print(
            "Usage: python summarize_consensus.py <output_tsv> <n_runs> <input_tsv_0> <input_tsv_1> ...",
            file=sys.stderr,
        )
        sys.exit(1)

    output_path = sys.argv[1]
    n_runs = int(sys.argv[2])
    input_paths = sys.argv[3:]

    if len(input_paths) != n_runs:
        print(
            f"Expected {n_runs} input files, got {len(input_paths)}",
            file=sys.stderr,
        )
        sys.exit(1)

    dfs = []
    for i, path in enumerate(input_paths):
        df = pd.read_csv(path, sep="\t")
        df = df[["chrom", "pos", "type", "added_events"]].copy()
        df = df.rename(columns={"added_events": f"run_{i}"})
        dfs.append(df)

    merged = dfs[0]
    for i in range(1, len(dfs)):
        merged = merged.merge(dfs[i], on=["chrom", "pos", "type"], how="inner")

    run_cols = [f"run_{i}" for i in range(n_runs)]
    consensus_mask = (merged[run_cols] > 1).all(axis=1)
    consensus = merged[consensus_mask].copy()

    consensus["avg_added_events"] = consensus[run_cols].mean(axis=1)

    out_cols = ["chrom", "pos", "type", "avg_added_events"] + run_cols
    consensus = consensus[out_cols].sort_values(["chrom", "pos"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    consensus.to_csv(output_path, sep="\t", index=False)
    print(f"Consensus: {len(consensus)} loci out of {len(merged)} passed filter")
    print(f"Written to {output_path}")


if __name__ == "__main__":
    main()
