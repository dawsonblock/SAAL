# V4 static AdaLoRA control: engineering status

The v4 contract now admits a separately named `adalora-static-r12` control.
It uses PEFT's AdaLoRA parameterization and SVD tensors, but fixes `target_r`
to 12 and never calls `update_and_allocate()`. This isolates AdaLoRA's adapter
parameterization from its importance-driven rank reallocation. It is not the
same control as ordinary fixed-rank LoRA: its tensorization, regularizer and
optimizer behavior remain AdaLoRA-specific.

The v4 backend, rank controller and portable replay are additive modules. The
v1-v3 backend and controller source files remain unchanged so their worker
identities and historical admissions are preserved. The v4 artifact validator
requires all initialized rank components to remain active, rejects allocation
events, checks the exact AdaLoRA tensor inventory, and requires replay of the
initial, selected and terminal snapshots.

The local PEFT test exercises real optimizer updates and confirms that no rank
allocation hook is invoked. Admission, worker-staging, artifact and regression
tests run locally. No v4 Colab campaign was launched by this change; there is
no new signed campaign admission or independent reviewer attestation. The
failed study and its consumed budget remain unchanged. Qualification remains
`NOT_EVALUATED`, independent generalization remains unestablished, and external
deployment remains prohibited.
