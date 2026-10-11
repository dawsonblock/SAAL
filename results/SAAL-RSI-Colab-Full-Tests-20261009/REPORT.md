# SAAL-RSI full Colab CLI test run

**Completed on a dedicated Tesla T4, Linux x86_64, Python 3.13.15.**

- Full corrected suite: **541 passed, 1 skipped, 0 failures/errors** (542 collected).
- Additional CUDA probes: **2 passed**. AdaLoRA performed GPU parameter updates, adaptive pruning, frozen-weight verification and safetensors restoration. A small ModernBERT configuration completed a GPU forward/backward pass.
- Hash-enforced packaged Linux dependencies were downloaded, installed offline, and checked for broken requirements. Exact wheels and runtime versions are recorded.
- Dedicated session terminated; a subsequent server listing confirmed no active sessions.

## Corrections and preserved failures

The original full suite produced 538 passes, 3 TLS certificate failures and 1 skip. Python 3.13 strict certificate validation exposed missing authority-key identifiers in ephemeral test certificates. The supplied patch adds CA and leaf subject/authority identifiers and appropriate critical key usages. TLS validation, credential-denial, identity-substitution and consumed-receipt tests remain enforced.

The first added ModernBERT CUDA probe failed because its default padding token exceeded the deliberately tiny vocabulary. The corrected probe explicitly uses in-range special-token IDs. The initial setup also failed because standard venv creation could not run ensurepip; the isolated environment was then created without pip and populated using the host pip's explicit target-interpreter support. All original failure evidence is retained.

The existing 211-file source inventory was audited: 210 files matched unchanged; the sole difference is the documented TLS **test fixture** patch. Production source and the previous distributed release remain unchanged. Apply `python313-certificates.patch` from the base release root to reproduce the corrected fixture.

## Platform and claim limits

The skipped test is the real macOS sandbox-exec confidentiality/network probe. It cannot run on Linux and is not counted as passed. This run does not qualify Linux hostile-worker isolation. The suite ran on Colab; the two explicitly CUDA-specific probes ran on the T4. Most existing tests retain their configured CPU execution.

No full pretrained-model training campaign, improved accuracy, protected evaluation, independent custody, recursive improvement or external deployment is established by these tests. Research qualification remains **NOT_EVALUATED**. A watchdog bounded execution; it is not a provider billing cap. No private authority keys or protected labels were transferred.

The original test archive and corrected test archive, complete logs/XML, dependency wheel hashes, environment identity, source delta, compatibility patch, monitoring sample and cleanup confirmations are included.
