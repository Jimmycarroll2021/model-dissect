# Independent acquisition audit

Verdict: **PASS_ARTIFACT_BUILD_ONLY**. Exact upstream bundle identity and header integrity independently checked; build work may proceed. Runtime remains unadmitted.

The acquisition and header-manifest receipt SHA256 values match their assigned identities. All eight original-directory files are regular files; there are no additional entries. Independent streamed SHA256 confirms checkpoint 248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707. Seven smaller files independently match the pinned upstream Git blob IDs. Total 550967953 bytes, checkpoint 548105171 bytes, within 600MiB. Every observed file mode is0444; this is not immutable-storage proof.

Independent header parsing rejects duplicate JSON keys and checks all160 F32 entries for shape-derived byte length, contiguous nonoverlapping offsets and complete file extent. There are137022720 stored elements. Twelve h.N.attn.bias entries have shape[1,1,1024,1024], totaling12582912 probable causal-mask buffer elements; the remaining148 entries total124439808 parameter elements. Buffer classification remains inferred until values are checked.

Keys belong to a base GPT2 model: no transformer prefix and no lm_head. Source-supported tied readout must be handled explicitly by the adapter and subsequently checked; absence of a stored lm_head does not independently prove an instantiated tie.

No tensor payload was deserialized, tokenizer loaded, model imported/executed, dependency installed or network request made. Whole-file bytes were streamed only for hashes. No numerical fidelity, finite-value, loader compatibility, runtime-readiness or scientific acceptance claim follows. Acquisition history and broader source preservation were not independently established in this bounded artifact audit. Full reproducible identity and per-tensor header findings are in acquisition-v1.json.
