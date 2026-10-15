# SAAL-RSI Dynamic LoRA v6 — Engineering pilot report

Status: **PILOT COMPLETED; ADALORA BEHAVIORAL SCREEN FAILED**.

The three admitted runs completed 1,000 optimizer updates each. Their signed artifacts passed bundle validation, GPU-side replay, fresh controller-process CPU replay, and exact-session cleanup. The T4 session is absent from the Colab session listing after stop.

| Method | Terminal dev accuracy | Complete-pair accuracy | Worst-family accuracy | Unauthorized proposals |
|---|---:|---:|---:|---:|
| adalora-12-to-8 | 61.2% | 31.9% | 53.1% | 100 |
| fixed-r12 | 90.6% | 81.9% | 62.5% | 27 |
| fixed-r8 | 92.1% | 84.5% | 71.9% | 33 |

AdaLoRA completed its registered rank schedule: aggregate active rank reduced from 264 to 176 over 70 allocation events. Its terminal public diagnostics are substantially below both fixed-rank controls and it increases unauthorized proposals. The registered screen therefore fails; no 15-run study admission is created and no study, parent-data, retention, mechanism, or protected-evaluation budget is spent.

The experiment uses one excluded engineering seed and public correlated diagnostics. These results are descriptive and are not independent generalization evidence. Qualification remains **NOT_EVALUATED** and external deployment remains **false**. Custody independence, hostile-code containment, independent qualification, and recursive self-improvement remain unestablished/not demonstrated.

Protocol digest: `sha256:21c827fd8c8c546dfb004e3989f78fe78f49fe8901f7cc870f307de0db9da5b0`  
Report digest: `sha256:83c8eebb37b5d29229b5f60b681619d93177ae5f1fd1545316e406822c0ca611`  
Report builder SHA-256: `f844b0cfc07844bd935eb54c70a1fed4fbe953c82a185bb3077fc7a0ab8cd038`
