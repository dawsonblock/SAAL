# v0.9.6 selective-import compatibility report

Status: **BLOCKED — source archive bytes are unavailable**.

The requested archive is registered by expected SHA-256 in `../RECOVERY_INPUTS.json`, but no matching file was found in the available Downloads or Codex workspace and no matching published GitHub release asset was listed. The archive's expected digest has not been independently observed in this worktree.

No module was copied from another branch or reconstructed from its name. This preserves the v0.9.7 baseline and avoids substituting unreviewed code for the stated recovery source. `IMPORT_ALLOWLIST.json` records the proposed candidates and explicitly excludes package metadata, signing material, old closure code, release scripts, and historical evidence.

When the exact ZIP bytes are available, record its observed SHA-256, compare each candidate member to the corresponding v0.9.7 source, assign `UNCHANGED`, `NEW`, `MODIFIED`, `INCOMPATIBLE`, or `REJECTED`, and review interfaces before importing. The v6 source closure remains authoritative.
