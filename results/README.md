# SAAL-RSI v0.9.4.3 additive results

Main release files remain byte-identical to main. Existing research/public-data/dynamic-LoRA evidence is retained. New records include the failed first T4 inventory attempt, the successful hash-locked T4 retry, final Linux packaging validation and clean extraction evidence. Transport chunks and the repeated394MBtest bundle are omitted; parts.json and input hashes identify them, while actual historical fixture bytes are present in main.

Fresh local suite545passes; T4 suite544passes plus1macOS-only skip; separate packaging15passes (not additional cases in the545suite). CUDA forward/backward and fresh-process probe restoration pass. No new pretrained learning experiment or empirical qualification was performed.

Run `python results/verify_results.py --pin AUTHENTICATED_CONTROLLER_PUBLIC_KEY_HEX`. This checks the main subset and signed results inventory. A bundled pin establishes self-consistency only; independent key custody/trust is unestablished. Read the root release report and reproducibility guide for scope. Corpus human reviews remain zero; independent generalization remains unestablished; external deployment remains prohibited.
