# Release notes — v1.0.0 governed reference build

This release converts the supplied paper-only Dream-RSI snapshot into an independently implemented, executable reference control plane.

Implemented:
- replay worlds with repeated stochastic outcomes;
- prefix-only policy observations;
- budget and parallelism enforcement;
- explicit support/uncertainty scoring;
- deterministic dataset partitioning into development, selection and holdout;
- fixed-breadth and risk-aware UCB controllers;
- replay evaluation with a single explicit objective;
- repeated-seed bootstrap estimation;
- replay/live rank-correlation calibration;
- independent qualification gates;
- fresh live-canary attestation;
- Ed25519 promotion signing;
- activation-time verification of qualification, canary, signature and artifact digest;
- content-addressed artifact storage with strict digest parsing;
- append-only hash-chained SQLite audit ledger;
- static screening of candidate policy source;
- CLI demo, examples, docs and automated validation.

Not claimed:
- reproduction of the paper's reported benchmark numbers;
- equivalence to the unreleased official Dream-RSI implementation;
- a hardened operating-system sandbox;
- proof that replay improvement implies live improvement;
- general recursive self-improvement or AGI.
