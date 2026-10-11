"""Trusted data reviewer operation; learner should NOT possess this private key."""
import argparse
import json
from pathlib import Path
from saal.admission import issue_manifest, keygen, verify_manifest

p = argparse.ArgumentParser()
p.add_argument("command", choices=["keygen", "sign", "verify"])
p.add_argument("--private-key")
p.add_argument("--public-key")
p.add_argument("--data")
p.add_argument("--manifest")
p.add_argument("--reviewer", default="")
p.add_argument("--rationale", default="")
a = p.parse_args()
if a.command == "keygen":
    keygen(a.private_key, a.public_key)
elif a.command == "sign":
    m = issue_manifest(a.data, a.private_key, a.reviewer, a.rationale)
    Path(a.manifest).write_text(json.dumps(m, indent=2, sort_keys=True)+"\n")
else:
    valid = verify_manifest(a.data, json.loads(Path(a.manifest).read_text()), a.public_key)
    print(json.dumps({"valid": valid}))
    raise SystemExit(0 if valid else 1)
