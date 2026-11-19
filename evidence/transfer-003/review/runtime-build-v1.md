# Independent runtime build review

Verdict: **SYNTHETIC_BUILD_PASS_RUNTIME_ADMISSION_BLOCKED**. Exact build manifest and all five source/test identities match. Seven supplied synthetic checks independently passed. No target model or tokenizer loaded.

Independent model-free guard probes establish failed call/reservation retention, duplicate-ID rejection, owned-child timeout, unresolved reservation rejection, missing-ledger fail-closed behavior, final-exit packet disk enforcement, and supervisor SIGINT cancellation with owned child killed/reaped. The first timeout test was too short to pass imports; that failed test receipt is preserved and the revised active-call timeout test passes.

Before target admission, implement or explicitly integrate live model parameter-state identities before/after each actor schedule. Current post-run hashes check original files, not native/candidate parameters in memory, so the numerical protocol's parameter preservation requirement is unmet. Add an injected-mutation regression.

Also freeze and validate TMPDIR/cache locations beneath EXTERNAL_ARTIFACT_ROOT in the actual launch environment; the supervisor currently inherits these locations and does not enforce the protocol routing. Reviewer synthetic runs explicitly supplied EXTERNAL_ARTIFACT_ROOT TMPDIR and disabled bytecode.

Cancellation is correctly propagated and the child is reaped, but the receipt has stop_reason=null with returncode=-9; capture an explicit cancellation/exception cause for clearer recovery. Guard identity/locks, bounded resource checks, native hook cleanup and raw-before-comparator sequence were inspected. Not every limit or concurrent-lock scenario was dynamically tested. All paths, hashes, results and limits are in runtime-build-v1.json.
