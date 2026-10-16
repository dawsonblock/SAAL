#!/usr/bin/env python3
"""Audit a ZIP without extracting it; emit ARCHIVE_AUDIT.json and a report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from saal_rsi.authority.archive_audit_v2 import audit_zip, render_recovery_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="ZIP archive to inspect without extraction")
    parser.add_argument("--closure", help="optional reviewed source-closure JSON")
    parser.add_argument("--max-total-uncompressed", type=int, default=64 * 1024**3)
    parser.add_argument("--max-member-uncompressed", type=int, default=16 * 1024**3)
    parser.add_argument("--out", required=True, help="output directory for the two audit reports")
    args = parser.parse_args()
    closure = json.loads(Path(args.closure).read_text(encoding="utf-8")) if args.closure else None
    report = audit_zip(args.archive, closure, max_total_uncompressed=args.max_total_uncompressed,
                       max_member_uncompressed=args.max_member_uncompressed)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "ARCHIVE_AUDIT.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output / "RECOVERY_REPORT.md").write_text(render_recovery_report(report), encoding="utf-8")
    print(json.dumps({"archive_sha256": report["archive"]["sha256"], "members": report["member_count"],
                      "lfs_pointers": len(report["lfs_pointers"]), "audit": str(output / "ARCHIVE_AUDIT.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
