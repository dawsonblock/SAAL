# Upstream model attribution

Experiments use answerdotai/ModernBERT-base revision
`11aa1a9b1d9dd6e8cded5040fe217c9b6ef92e88`. The official
[pinned model card](https://huggingface.co/answerdotai/ModernBERT-base/blob/11aa1a9b1d9dd6e8cded5040fe217c9b6ef92e88/README.md)
states that the architecture and weights are released under Apache 2.0.
The source package contains the Apache 2.0 license text.

SAAL modifies classification heads and, depending on the admitted method,
adapters or encoder parameters. Full-tuning checkpoints necessarily contain
encoder weights. The standalone upstream snapshot/cache is not separately
bundled. This is a research modification; no upstream endorsement or independent
alignment qualification is implied.
