# Independent failed echo-run audit

Verdict: VERIFIED_IMPLEMENTATION_STORAGE_FAILURE. No causal outcome established. No new runtime admitted.

The guard terminated the producer with `Phase disk cap`, exit -15. Staged logical footprint is537,949,366bytes against536,870,912bytes, exceeding the cap by1,078,454bytes. The295 call artifacts occupy393,032,855bytes. This is an avoidable runner serialization bug, not evidence against the scientific hypothesis and not a reason to weaken the disk guard.

Every one of295 saved vectors independently matched its durable journal SHA256 and top1. The journal sums to15,332positions, ending at echo-01:scan:3:7:1. No extra unjournaled call file exists. Serial source permits at most one next52-position invocation to have started/computed without a completed durable receipt, so completed/attempted work beyond15,332 is uncertain, bounded by15,384 under this source. The full78,112reservation remains charged regardless.

The stored tensor has shape6400 and25,600logical bytes, but the52-position view retains1,331,200backing-storage bytes; the128-position case retains3,276,800bytes. `.detach().contiguous()` does not copy an already contiguous last-row view, and `torch.save` serializes backing storage. A tiny64-element synthetic independently reproduces256storage bytes for a64logical-byte view, versus64storage bytes after clone. This source-level cause is confirmed against real saved artifacts.

Minimal repair: use an owning clone for serialized final logits, assert storage_offset==0 and storage_nbytes==numel*element_size, and test52/128×6400 backing tensors, exact logical equality/digest/top1 plus serialized size. This adds no model forwards. Verify a full projected artifact budget before any later admission; retain original failed evidence and reservation.

One compact derivative of call0295 was saved only in reviewer-owned output. It is27,367bytes, logically exact, with25600storage bytes. Original SHA stayed unchanged. A bulk repair should create new files plus source-file hash/source-vector hash/destination-file hash manifest, never overwrite/delete the failed originals. Such copying cannot itself reduce the failed workspace footprint; it does not make that workspace admissible again.

Producer receipts record52/128 parity and all-head hook preservation PASS and8/8 development eligibility. One complete pair head-scan receipt exists (echo00); echo01 is partial. No selected-head receipt, heldout output or final result exists. These recorded passes have not been independently repeated on the actual model.

Budget consequence:804,208prior+78,112failed reservation=882,320 lifetime charged, leaving117,680. A fresh pair of78,112reservations needs156,224 and exceeds that remainder by38,544. Even optimistically reusing15,332completed positions, finishing the original full path plus a full independent actual repeat needs61,300+76,632=137,932positions, exceeding the remainder by20,252 before any new reservation padding. Therefore serialization repair alone cannot justify restarting or resuming the unchanged full protocol. Parent must resolve scope/budget authority explicitly; this audit grants none.

Machine evidence and all295 artifact hashes: `failure-audit-v1.json`. Reproduction code: `audit-failure-v1.py` (immutable compact-example output intentionally prevents blindly rerunning). No model/tokenizer loaded; zero model calls.
