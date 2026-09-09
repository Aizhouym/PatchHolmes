#!/usr/bin/env python3
"""Phase 2 ablation: replace short CVE description with the full NVD markdown report.

Implementation: zero code change.

`scripts/run_phase2_full.py` reads `--dataset-csv` and only consumes two
columns: `cve` and `cve_description`. The runner then passes that string
verbatim into `CVEQuery(description=...)`, which the Phase 2 agent inserts
into the system prompt.

So this ablation just constructs an alternative dataset CSV where the
`cve_description` column holds the full markdown report from
`./data/nvd_cache.csv` instead of the short 100-200-char NVD
summary. Everything else (Phase 2 agent, tools, runner, LLM) is identical.

Usage
-----
    python ablation/phase2_cve_report/build_report_csv.py \\
        --sample-csv    ./data/sample_ground_truth_810.csv \\
        --nvd-cache     ./data/nvd_cache.csv \\
        --output        ./data/cve_report_810.csv

Then pass the output via `--dataset-csv` to the main runner:

    python scripts/run_phase2_full.py \\
        --phase1-jsonl  logs/phase1/main_rrf_810.jsonl \\
        --dataset-csv   data/cve_report_810.csv \\
        --output        logs/phase2/ablation/cve_report/results.jsonl \\
        --num-workers   64
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build dataset CSV with NVD markdown report as cve_description.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--sample-csv", required=True, type=Path,
        help="Source CSV with the CVE list (column 'cve'). "
             "Typically ./data/sample_ground_truth_810.csv.",
    )
    p.add_argument(
        "--nvd-cache", required=True, type=Path,
        help="NVD cache CSV with columns cve_id, markdown, error. "
             "Not redistributed in this repo (~16 MB); fetch via Favia's "
             "scraping pipeline or query the NVD API for each CVE in "
             "data/ground_truth_queries_clean.csv. "
             "Note: the pre-built data/cve_report_810.csv is already shipped, "
             "so you only need to re-run this script if you change the sample.",
    )
    p.add_argument(
        "--output", required=True, type=Path,
        help="Destination CSV (cve + cve_description columns).",
    )
    p.add_argument(
        "--carry-other-columns", action="store_true",
        help="Also copy non-essential columns (owner/repo/patch/etc.) from sample-csv "
             "into the output so the file is a drop-in replacement for the sample CSV.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load sample CVE list (preserve order)
    sample_rows = list(csv.DictReader(args.sample_csv.open()))
    sample_cves = [r["cve"].strip() for r in sample_rows if r.get("cve")]
    print(f"Sample CSV: {args.sample_csv}  ({len(sample_rows)} rows)")

    # 2. Load nvd_cache into a dict by cve_id
    print(f"Reading NVD cache: {args.nvd_cache}")
    cve_to_md: dict[str, str] = {}
    cve_to_err: dict[str, str] = {}
    with args.nvd_cache.open() as f:
        for r in csv.DictReader(f):
            cid = (r.get("cve_id") or "").strip()
            if not cid:
                continue
            cve_to_md[cid] = r.get("markdown") or ""
            cve_to_err[cid] = (r.get("error") or "").strip()
    print(f"  NVD cache entries: {len(cve_to_md):,}")

    # 3. Resolve report for each sample CVE
    missing = []
    err_recorded = []
    rows_out: list[dict] = []

    if args.carry_other_columns:
        fieldnames = ["cve_description"] + [c for c in sample_rows[0].keys() if c != "cve_description"]
    else:
        fieldnames = ["cve", "cve_description"]

    for src in sample_rows:
        cve = src["cve"].strip()
        md = cve_to_md.get(cve, "")
        err = cve_to_err.get(cve, "")
        if not md and err:
            err_recorded.append((cve, err[:80]))
        if not md:
            missing.append(cve)
            md = ""  # leave empty; runner will treat as empty description

        if args.carry_other_columns:
            row = dict(src)
            row["cve_description"] = md
        else:
            row = {"cve": cve, "cve_description": md}
        rows_out.append(row)

    # 4. Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    print(f"\n✓ Wrote {len(rows_out)} rows → {args.output}")
    print(f"  CSV columns: {fieldnames}")
    if missing:
        print(f"  ⚠ {len(missing)} CVE had no markdown report in nvd_cache "
              f"(written with empty description):")
        for c in missing[:10]:
            print(f"    {c}")
    if err_recorded:
        print(f"  ⚠ {len(err_recorded)} CVE had error in nvd_cache:")
        for c, e in err_recorded[:5]:
            print(f"    {c}: {e}")

    # 5. Quick report-length stats
    lens = [len(r["cve_description"]) for r in rows_out if r["cve_description"]]
    if lens:
        lens_sorted = sorted(lens)
        print(f"\n  cve_description (markdown) length stats over {len(lens)} non-empty rows:")
        print(f"    min={min(lens):>6}  max={max(lens):>6}  "
              f"mean={sum(lens)/len(lens):>6.0f}  "
              f"median={lens_sorted[len(lens_sorted)//2]:>6}  "
              f"p95={lens_sorted[int(len(lens_sorted)*0.95)]:>6}")


if __name__ == "__main__":
    main()
