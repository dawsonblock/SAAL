# Final physical ZIP verification

The archive is SAAL-RSI-v0.9.5-dev6-Full-Research-Evidence.zip, 7,470,857,286 bytes.
SHA-256: `9d3b115adcb0262354173a949ed9266812dd289247c1c2b73fff2aeb1326185b`.
It binds source commit `849760f49b1fc5d0cf8d941a3757c0e33fcf0a35` and evidence
commit `23b535a3cdf295a057e49d2d108899dddb55e137`. This later verification
commit adds detached evidence without changing the tested ZIP.

Attempt 03 completed: all 6,692 physical ZIP members were fully extracted;
6,696 logical files, 617 bound source files and 253 frozen historical files
verified. All eight canonical suites passed, 932 tests in total, with no
failures, errors or skips on pinned macOS Python 3.12. Failed-study closure
and its controller signature reproduced. The strict verifier then checked
all files again, rejected no extras, and the actual archive digest was unchanged.

Attempts 01 and 02 each passed the same 932 tests but FAILED final member
verification because verification processes created unregistered bytecode.
Their failures, hashes and logs remain intact. Attempt 03 uses explicit -B
and read-only extracted source directories to prevent isolated workers from
writing caches. It adds no cache whitelist and changes no registered source
bytes. Test fixtures and outputs remain writable outside src. The disposable
12 GB extraction was removed only after all checks passed; logs and ZIP remain.

INDEX.json and SIGNATURE.json bind this complete verification evidence to the
same controller identity as the package. This proves controller attribution
and integrity, not independent custody. The study remains FAILED/incomplete:
10 accepted runs, one interrupted, four RESERVED; qualification NOT_EVALUATED,
independent generalization NOT_ESTABLISHED, external deployment false.
