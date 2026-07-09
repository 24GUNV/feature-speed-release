#!/usr/bin/env python3
"""Run or dry-run the canonical ICAIF v2 experiment pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = REPO_ROOT / "paper_icaifv2" / "public" / "MANIFEST.json"
UV_PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "xgboost",
    "lightgbm",
    "tqdm",
    "joblib",
]


def load_manifest() -> dict:
    with MANIFEST.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def command_for(script: str, executor: str, python_executable: str) -> list[str]:
    if executor == "python":
        return [python_executable, script]

    cmd = ["uv", "run"]
    for package in UV_PACKAGES:
        cmd.extend(["--with", package])
    cmd.extend(["python", script])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dry-run or execute the canonical ICAIF v2 scripts."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually run the scripts. By default this only prints commands.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional manifest ids to run, for example: --only ml_vs_sorts feature_speed",
    )
    parser.add_argument(
        "--include-needs-fix",
        action="store_true",
        help="Also include manifest entries currently marked needs_fix.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verify_paper_claims.py after the selected jobs.",
    )
    parser.add_argument(
        "--executor",
        choices=["uv", "python"],
        default="uv",
        help="Use uv-managed dependencies or the active Python environment.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use with --executor python.",
    )
    args = parser.parse_args()

    manifest = load_manifest()
    selected = []
    only = set(args.only or [])

    for item in manifest["canonical_results"]:
        if only and item["id"] not in only:
            continue
        if item.get("status") == "needs_fix" and not args.include_needs_fix:
            continue
        if not item.get("script"):
            continue
        selected.append(item)

    if only:
        missing = sorted(only - {item["id"] for item in selected})
        if missing:
            print(f"Unknown or excluded manifest ids: {', '.join(missing)}", file=sys.stderr)
            return 2

    if not selected:
        print("No scripts selected.", file=sys.stderr)
        return 2

    mode = "RUN" if args.run else "DRY RUN"
    print(f"{mode}: {len(selected)} canonical scripts")
    for item in selected:
        cmd = command_for(item["script"], args.executor, args.python)
        print(f"\n[{item['id']}] {item['paper_use']}")
        print("$ " + " ".join(cmd))
        if args.run:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    if args.verify:
        verify = REPO_ROOT / "paper_icaifv2" / "public" / "scripts" / "verify_paper_claims.py"
        cmd = [sys.executable, str(verify), "--write-report", "--write-json"]
        print("\n[verify] Paper claim verification")
        print("$ " + " ".join(str(part) for part in cmd))
        if args.run:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
