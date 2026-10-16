# Checkpoint recovery report

Status: **content hashes verified; publisher provenance remains open**.

All six historical checkpoint paths are present as hydrated Git LFS files in the v0.9.7 baseline checkout. The six paths map to four unique objects, each 598,445,904 bytes. The verifier streamed each file, matched the bytes to its full LFS SHA-256, checked the LFS pointer blob at baseline commit `fb2e7eb000fb2a4ca50e2b5ec5034e02d3b95f4c`, and parsed each Safetensors header. Each has 138 tensors. Shared tensors match the pinned local ModernBERT-base tensor inventory; the expected masked-language-model decoder bias is replaced by a four-class classifier weight and bias.

The observed equality confirms content integrity and the recorded duplicate relationships: the Behavior Closure best and terminal paths point to the same object, and both campaigns share the same initial object. The historical v0.9.5 archive still omitted all six members; this recovery does not rewrite that archive or its manifest.

The GitHub baseline ref matched the requested baseline commit during this run, but the commit and source publisher were not authenticated against an independently administered trust root. Therefore no object is marked `PROVENANCE_VERIFIED`, and historical reproduction is not fully closed. The requested v0.9.6 ZIP is also absent, so its pointer text and candidate code cannot be compared directly.

No weights were regenerated, no LFS fetch or duplicate copy was performed, and no training campaign was launched. See `CHECKPOINT_REQUIREMENTS.json`, `VERIFIED_PAYLOADS.json`, `LFS_OBJECT_REGISTRY.json`, `CHECKPOINT_PROVENANCE.json`, and `RECOVERY_ATTEMPTS.jsonl` for the exact records.
