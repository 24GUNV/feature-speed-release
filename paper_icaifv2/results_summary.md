# ICAIF v2 Results Evidence Path

This file used to contain an intermediate narrative from an earlier version of
the project. It is no longer an evidence source for `main_v6_frontier`.

Use the paper-specific public verification directory instead:

- `paper_icaifv2/public/MANIFEST.json`
- `paper_icaifv2/public/results/verification_report.md`
- `paper_icaifv2/public/results/computed_claims.json`
- `paper_icaifv2/public/results/tables/`

Regenerate the verified outputs from the repository root:

```bash
python3 paper_icaifv2/public/scripts/verify_paper_claims.py --write-report --write-json
python3 paper_icaifv2/public/scripts/build_paper_tables.py
```

