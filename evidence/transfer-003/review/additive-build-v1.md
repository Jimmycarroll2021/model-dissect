# Additive integration v1 independent static review

Verdict: BLOCKED_FOR_ADMISSION. No tokenizer admission issued. No synthetic/model execution, target tokenization or target weight loading during this review; coordinator held execution while numerical producer was active.

Exact dispatched source b07cbb864ff1dea8119ff4972c2f15b8db945b840d4024b70b47f4abe59d2661, additive manifest fdc20005258fa32da586360a1c7d138b57de8195934cbfe4521b209d751ae564 and tokenizer packet 34935fbe891b50387cf42ddff606ae6138deeba96cfbf433df1c728d0ddbff45 match. Forty non-weight packet dependencies were independently hashed and all matched; model.safetensors was deliberately not re-read while producer active. Receipt: additive-v1-static-identities.json.

## Required narrow corrections

1. science() only binds tokenization_receipt as an identity; it never validates receipt pass, protocol_sha256, or development/heldout hashes against the actual token banks. A frozen unrelated receipt and independently frozen altered bank would satisfy this path. Add executable receipt success/protocol/bank hash checks before model loading, hashing held file without deserializing its contents before selection. Add negative tests for rejected failed receipt and each mismatched linkage. Preserve held loader semantics.
2. execute() finally computes actual identity hashes in a dict comprehension. Missing or unreadable dependency causes an exception before identity-final.json is written. Thus supervisor/ledger fail closed, but per-file preservation evidence disappears. Catch per-file identity errors, persist expected/actual/error/pass status for all dependencies, then raise if any failed. A test must inject an operation failure plus an unreadable/missing identity and verify receipt survival and unsuccessful completion.

Coordinator requires both corrections and additive-v2 re-freeze before admission. Preserve current source in revision snapshot; this review is not a request to edit the frozen v1 packet in place.

## Static positive findings and remaining execution

Trace adapter runs exactly one forward with explicit all-one masks and positions, checks observations cover all twelve layers, owns cloned observations, and uses inference mode. Lazy held loader checks a selection file and selected site before deserializing held tokens. Science identity callback checks live native parameters/buffers and candidate weights/causal buffer. Guard rejects unresolved drafts, wrong phase/stage/operation/budget/environment and absent required dependency hashes; exact admission and supervisor are required. Tokenizer packet is bounded to 0 model positions, 120 seconds and 16MiB, offline local tokenizer, fixed protocol fixtures and add_special_tokens=False, no replacement on fixture rejection. Token position accounting 108 per development length plus20 per held length plus2 trace length is correct.

Author's two actual 12-layer/12-head tiny-model tests inspect all twelve science sites, matched edits, complement/no-op/restoration, actual attention@V and exception cleanup. Source inspection supports their intended coverage, but independent execution remains held and no independent test PASS is claimed here. Full corpus provenance and failure-preservation tests are not present in v1.

Atlas metadata correction independently PASS: manifest-v3 670a6e0d057e44a337e5144cf00577c5af6b20aa7e9915780afe8ec434ae06ed; preservation receipt f27016b6696d5c04d2f91f6f5e4b017ca7393a8dfaf370c097c10a3119231c3b. Diff is exactly one source-provenance paragraph replacing candidate-manifest-v2 by frozen v3 and explicit hashes. Prior document/manifest snapshots retain prior identities; atlas source/tests unchanged. Prior arithmetic/test review remains valid. Historical dependency_refresh field in v3 is copied from v2; documentation_delta explicitly records the new doc change.
