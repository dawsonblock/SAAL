# Compatibility and draft-evidence migration

Historical job-v2/job-v3/job-v4 contracts and the original `model_worker.py` are unchanged. The new job-v5 and result-v4 use an additive entry point (`saal-pretrained`), native snapshots, separate tensor inventories, and three trainable checkpoints. Legacy checkpoints are never reinterpreted as ModernBERT or Qwen.

`ModelContractV5` is represented by `PretrainedModelSpec` plus the closed signed job envelope in `pretrained_contract.py`. New execution accepts only the current reviewed worker identity. Completed draft-v5 public bundles can be checked under an explicit historical worker allowlist that authenticates their archived source ZIP. That permission is verification/replay only: the current worker and durable execution registry reject an old job for new execution. Replay old evidence using its archived source and original signed job, rather than relabeling it with current source identity.

The first draft ModernBERT GPU attempt failed on nonfinite gradients. A separately admitted lower initial loss scale enabled finite updates. The first full-tuning retrieval exceeded its single-file transport deadline; it remains interrupted even though GPU training and same-host replay were logged. Chunked transfer was introduced for subsequent attempts. Qwen FP16 produced finite forward outputs but failed its first gradient check with reference kernels; FP32 is a distinct admitted numerical experiment, not an automatic retry or evidence of equivalent FP16 performance.

The initial Qwen public prompts did not explicitly enumerate decision labels. Their baseline diagnostics and failure records are preserved. The revised public generative dataset declares the decision vocabulary and has a new generator/source identity. Its before/after results must be analyzed within that revised population. Counts and oracle labels alone cannot justify comparisons across dataset revisions.

The initial and terminal checkpoints survive even if the development selector chooses step zero. Head digests, full native base identity and tokenizer/processor hashes belong to the restoration manifest. Versioned manifests and signed evidence must be regenerated deliberately only after source/evidence freeze; historical inputs remain unchanged.

No final holdout has been consumed for these pretrained models. Alpha-spending controls in inherited protected evaluation remain intact, but no operationally independent v5 assurance service has been established. A successful source package or public GPU trial does not supply that missing authority.

## Final admission and restoration repairs

The final source rejects malformed category/source/parent metadata before group lookups, bounds sampling descriptions and rationales, requires public training/development/regression partitions and caps serialized public datasets at 30 MB. Signed job/config input is read as a bounded regular file (50 MB maximum) before JSON/signature parsing. The explicit signed-job artifact limit is 50 MB, matching the embedded-dataset contract instead of admitting jobs that later fail a smaller export limit.

Checkpoint restoration validates every tensor before copying any parameter; a late invalid tensor leaves the live model unchanged. Malformed classification labels/scores count as invalid predictions without crashing confusion-matrix construction. Controller monitoring recognizes an explicit remote Python child failure even when a CLI version returns success for the surrounding remote-execution command.

These repairs change worker identity. The original learning and sequence worker ZIPs are preserved with reconstructed identities, archive hashes and `VERIFY_AND_REPLAY_ONLY` scope. Their public reports remain attributable and reconstructable; they cannot execute again under final admission. Final-source CPU/T4 compatibility jobs are separately admitted and are excluded from learning distributions. No historical qualification threshold, authority implementation or checkpoint byte is changed.
