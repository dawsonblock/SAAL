# Threat model

Protected properties:

- candidate policy code cannot activate itself;
- historical outcome records cannot silently mutate without digest changes;
- promotion binds to exact candidate and qualification evidence;
- future/holdout observations are not part of the policy context;
- malformed CAS digests cannot traverse outside the artifact store;
- policy source is statically screened before any optional dynamic loading.

Out of scope for this local reference implementation:

- kernel-level isolation from malicious native code;
- defense against compromised host/root;
- HSM-backed key custody;
- distributed consensus;
- confidential-computing attestation;
- proof that replay rankings transfer to live stochastic agent behavior.

For production, execute untrusted candidate policies in a separate uid/container/VM with no network, no secrets, read-only inputs, CPU/RAM/time quotas, and a narrow RPC interface. Keep qualification and signing on a different trust domain.
