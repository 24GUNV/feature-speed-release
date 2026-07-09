# Cleanup Notes

This is a non-destructive cleanup pass.

## Canonical Area

Use `paper_icaifv2/public/` as the public evidence path for the ICAIF v2 paper.
The repository-level `public/` directory appears to belong to an older project
and is intentionally untouched.

## Canonical Inputs

The canonical result files are listed in `paper_icaifv2/public/MANIFEST.json`.
The verifier reads those files directly and recomputes the table-level claims
from the CSVs.

## Formerly Stale Or Risky Artifacts

- `paper_icaifv2/results_summary.md` is now only a pointer to this public
  evidence path.
- `scripts/concern_experiments/expAX_statistical_tests_summary.md` was
  regenerated from the current `expAX_full_comparison.csv`.
- `generate_bound_figure.py` now uses the same equal-weight feature-set FSI
  defined in `main_v6_frontier.tex` and writes public verification CSVs.
- `generate_frontier_figure.py` now reads points from canonical CSVs and stores
  turnover in fractional units before plotting.

## Directory Hygiene

`paper_icaifv2/.gitignore` now ignores LaTeX build products and `.DS_Store`.
No existing files were deleted or moved in this pass.
