# ICAIF v2 Public Evidence Path

This directory is the clean, paper-specific public area for `main_v6_frontier`.
It does not replace or modify the repository-level `public/` directory.

## Main Commands

Run from the repository root:

```bash
python3 paper_icaifv2/public/scripts/verify_paper_claims.py --write-report --write-json
```

The verifier reads the canonical CSV outputs, recomputes the paper-facing
numbers, and scans the source for leakage/provenance guardrails. It exits
nonzero if any blocking check fails.

To see the canonical experiment commands without running the expensive jobs:

```bash
python3 paper_icaifv2/public/scripts/run_canonical_experiments.py
```

To regenerate the canonical outputs:

```bash
python3 paper_icaifv2/public/scripts/run_canonical_experiments.py --run
```

The local runner uses `uv run --with ...` by default so the experiment
environment is explicit and does not depend on whichever `python3` happens to
be active.

For a compute node with an activated virtualenv:

```bash
python paper_icaifv2/public/scripts/run_canonical_experiments.py --run --verify --executor python --python "$(which python)"
```

On a Slurm cluster:

```bash
sbatch paper_icaifv2/public/scripts/run_canonical_on_slurm.sh
```

To rerun only selected jobs on Slurm:

```bash
sbatch --export=ALL,ONLY="ml_vs_sorts feature_speed" paper_icaifv2/public/scripts/run_canonical_on_slurm.sh
```

To export paper tables from the verified CSVs:

```bash
python3 paper_icaifv2/public/scripts/build_paper_tables.py
```

## Files

- `MANIFEST.json`: canonical scripts, result files, paper use, and caveats.
- `scripts/verify_paper_claims.py`: paper-claim and leakage verifier.
- `scripts/run_canonical_experiments.py`: dry-run or execute canonical jobs.
- `scripts/run_canonical_on_slurm.sh`: compute-node wrapper for full retraining.
- `scripts/build_paper_tables.py`: export LaTeX/Markdown tables from canonical CSVs.
- `docs/CLEANUP_NOTES.md`: what is canonical, stale, and intentionally untouched.
- `results/`: generated verification reports and computed claim summaries.

## Policy

Only numbers that pass the verifier should be used in the paper. Stale summaries
and hard-coded figure values should not be treated as evidence.
