# Upgrade v0.2.0 -> v0.3.0 (fail closed)

## What was preserved

The synthetic assistance simulator, softmax trainer, offline LoRA script,
policy/approval logic, delegation rules, outcome verification, audit ledger,
Ed25519-qualified candidate staging and original regression suite remain.

## Breaking changes

1. Unix `GateServer` constructor now **requires** `peer_bindings={uid: frozenset({(principal, task_id)})}`.
2. `saal gate-server` requires `--peer-bindings /secure/peers.json`; optional
   `--socket-group-gid` opens access only to that dedicated group (`0660`).
3. Authority policy and peer-binding files must be private regular files owned
   by the daemon effective UID, mode `0600`; symlinks are refused. Secret files
   have the same requirement. Example config in the ZIP is not an installed file.
4. `Delegation.max_actions` defaults to 100. A capability reserves **one** action
   even at zero cost. Tokens use `claims.version=3` and include `action_limit`.
5. v2 replay SQLite schema is rejected. Drain/revoke outstanding tickets, archive
   the v2 SQLite file externally, then start with an **empty v3** DB; do not
   delete old records required by your audit obligations.
6. v3 preregistration files contain `require_replay=true`. The qualifier must
   have the **exact candidate model** at `--model`; absent or mismatched replay
   rejects qualification. Legacy demonstration configs remain compatible, but
   only `require_replay=true` satisfies the new research gate.

## Safe demo setup

```bash
# Run each command under the intended authority UID on a hardened POSIX host.
install -m 600 configs/example_delegations.json /secure/saal-delegations.json
python - <<'PYCODE'
import json, os
from pathlib import Path
p = Path('/secure/saal-peer-bindings.json')
# Replace with actual actor UID and administrator-approved task identifiers.
p.write_text(json.dumps({str(os.getuid()): [{'principal':'demo_user','task_id':'demo-1'}]}))
p.chmod(0o600)
PYCODE
saal init-key --out /secure/authority.key
saal init-key --out /secure/approval.key
saal gate-server --socket /secure/saal.sock \
  --authority-key-file /secure/authority.key \
  --human-key-file /secure/approval.key \
  --replay-db /secure/replay-v3.sqlite \
  --delegations /secure/saal-delegations.json \
  --peer-bindings /secure/saal-peer-bindings.json
```

A mapping for the authority daemon UID is only suitable for a same-UID smoke test;
for cross-UID operation configure a separate actor UID, a dedicated Unix group
and `--socket-group-gid`. The directory holding the socket must permit the actor
to traverse it, but must not expose policy files, keys, or the replay DB. Do not
misinterpret this socket service as a complete trusted executor.

## Reproducible demo (not independent)

```bash
python -m pip install -e '.[attestation]'
python -m unittest discover -s tests -q
saal train --out artifacts/model.json
saal evaluate --model artifacts/model.json \
  --prereg configs/preregistered_eval.json --out artifacts/public.json
```

See `docs/replay_qualification_v3.md` for the **real** two-environment release
process. There is no automatic promotion after qualification.
