# SAAL-RSI public-data foundation v1

Downloaded and pinned supplementary public research data. This capsule is additive; the existing SAAL-RSI releases, pretrained cache, learning failures and Colab test evidence remain unchanged.

| Resource | Acquired | Intended scope |
|---|---|---|
| Google IFEval | All 541 public prompts and dataset card | Instruction-following diagnostics; verifier implementation is not bundled or claimed executed |
| Facebook ANLI | All nine native Parquet split files and dataset card | Native natural-language inference; no automatic SAAL decision mapping |
| AllenAI NaturalInstructions | Complete index of 1,613 task paths; 11 selected task files, README and license | Small exploratory subset, not the full task corpus or an independently sampled benchmark |
| ModernBERT and Qwen models | Exact repository references | References only; no new model weights downloaded or training admitted |

`ACQUISITION.json` binds repository commits and actual source-file SHA-256 hashes. `verify.py` checks content/path substitution and rejects public-source qualification, floating revisions and acquisition-level training claims. This is an offline integrity utility, not a replacement for existing signed training admission or qualification services. Downloaded prompts are untrusted data; dataset code was not executed.

The ModernBERT reference preserves revision `11aa1a9b1d9dd6e8cded5040fe217c9b6ef92e88`. Qwen Base and post-trained revisions were resolved independently and are recorded. Admitting them still requires exact snapshot/tokenizer/processor, worker, dependency, adapter and budget identities. The existing ModernBERT cache is not duplicated into this ZIP.

## License and scope

IFEval's dataset card declares Apache-2.0. ANLI's card declares CC BY-NC 4.0. NaturalInstructions task metadata is Apache-2.0, while task instances follow their original dataset licenses. Preserve the upstream cards and consult `TASK_LICENSE_INDEX.json`; no common commercial-use approval is inferred across the sources.

Authoritative sources: [IFEval](https://huggingface.co/datasets/google/IFEval), [ANLI](https://huggingface.co/datasets/facebook/anli), [NaturalInstructions](https://github.com/allenai/natural-instructions). Exact pins are in the acquisition record, rather than floating links.

## What remains

There are **zero independently reviewed SAAL decision records** in this capsule. `SAAL_REVIEW_TEMPLATE.csv` and `REVIEW_PROTOCOL.md` define the pending authoring/review work. No 400–800-example reviewed dataset, native-label conversion, final-holdout independence, trained improvement, Linux sandbox qualification or promotion is claimed.

The prior local Python 3.13 certificate patch and corrected GPU test/setup files are already available in `SAAL-RSI-Colab-Full-Tests-20261009.zip` on your Mac. Reattachment is unnecessary for this workspace. No new cloud session was started by acquisition.

Run offline integrity checks with `python verify.py`; run the nine boundary tests with `python -m pytest test_integrity.py -q`. Acquisition uses the existing version-pinned local Hugging Face environment; it does not alter that environment or install the unrelated datasets library. ANLI is retained in native Parquet format; loading and statistical analysis were not performed here.
