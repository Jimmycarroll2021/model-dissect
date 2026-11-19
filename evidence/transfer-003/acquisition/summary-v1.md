# GPT-2 acquisition and header-only inspection

PASS_ACQUISITION_HEADER_ONLY. Acquired exactly8 pinned files into `EXTERNAL_ARTIFACT_ROOT/workspaces/model-transfer-003-gpt2/artifacts/original`. Total550967953bytes; fresh streamed transfer550967953bytes; no retry, no overwrite, no existing-file reuse. Every non-LFS file matches its recorded Git blob SHA1 and size; safetensors matches its advertised LFS SHA256 and size. SHA256 recorded for every file. Checkpoint SHA256248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707.

All files mode0444; this is not immutable protection against their owner. Each file fsynced before rename and directory fsynced after completion. Available `EXTERNAL_ARTIFACT_ROOT` space before50576171008bytes and after50025160704bytes, exceeding2GiB reserve. Elapsed11.435seconds. No model or tokenizer loaded, no installation, GPU or tensor numerical examination.

Safetensors header14283bytes, payload548090880bytes;160 tensors, allF32;137022720stored elements. Contiguous, nonoverlapping offsets cover all payload bytes; every shape/dtype-derived byte count matches offsets. Header metadata format=pt. Twelve keys `h.N.attn.bias`, each[1,1,1024,1024], account for12582912elements and50331648bytes; their attention-buffer classification is a naming/architecture interpretation, not a header trainability flag. Other148tensor entries total124439808elements and497759232bytes.

Embedding entries are `wte.weight`[50257,768] and `wpe.weight`[1024,768]. No `lm_head.weight` entry and no `transformer.` prefix. A tied native output head is the expected architecture interpretation; no duplicate output tensor or runtime alias relationship was tested. Loader must explicitly map these base-model keys and validate expected missing/ignored buffer handling without silently accepting arbitrary mismatches. Header shape inspection establishes storage layout, not tensor finiteness or model execution correctness.

Identity files:
- acquisition-v1.json SHA256a2df49e96768c4a9b128447d6c295c7ccc4c4ff8c670d52aca046d16535c60b0
- safetensors-header-v1.json SHA2560c58f57dc9d870378cc1195c914ce7065e45e6eacc79fadfc5ef33dd32084181
- acquire-v1.py SHA256f6a85ad8aaba3cc115d96b14508be87b1443cd6c8587be73570a88d42e3d9892

Independent local artifact audit is still required; no runtime admission follows from this receipt.
