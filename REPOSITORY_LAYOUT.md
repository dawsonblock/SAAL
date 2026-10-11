# SAAL repository branches and complete artifacts

`main` is the v0.9.4.3 reproducibility repair of the preserved v0.9.4.2 build. It retains historical model workers, checkpoints, public learning failures and provenance. The mTLS test fixture and version metadata are changed deliberately; parent-inventory.json and changes.json identify every edited parent file. reviewed-decision-corpus/ remains a 600-record,300-pair review candidate, not a completed human-reviewed corpus.

`results/saal-rsi-0943` adds prior research evidence and fresh validation records under results/. Main files remain unchanged in this additive branch. Verify main explicitly with --allow-results and verify results separately. The older results/saal-rsi-20261011 branch remains historical.

Use complete physical release assets or a hydrated Git LFS checkout. GitHub automatic source ZIPs may contain pointers. Neither a Git blob pointer nor a successful source inspection proves a checkpoint can be replayed. REPRODUCIBILITY.md documents physical verification and exact test commands.

Human reviews are zero; independent generalization is not established. No new pretrained learning campaign, independent assurance deployment or protected final evaluation was performed in the repair. Research qualification remains NOT_EVALUATED and external deployment remains prohibited. Do not interpret successful software tests as model qualification.
