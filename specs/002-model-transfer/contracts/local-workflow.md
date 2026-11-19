# Local transfer interface contract (proposed)

Model adapter input: batch-one token IDs, full boolean attention mask, explicit position IDs, optional real K/V cache and call-local intervention. Output: embedding, every block residual, final representation, full logits, cache plus selected capture metadata. Validate mask/position/shape bounds; do not silently infer incompatible semantics. Candidate and reference may expose different native cache layouts but normalize documented comparison sites.

Inventory/map and saved-evidence inspection commands are model-free. Forward/trace/probe commands require one exact admitted packet, unique outputs and reserved costs. Planned CLI operations: inspect, map, parity, capture, intervene, report. Implementation owner must supply actual --help and demonstrated argv; these names are design intent, not existing executable claims.

Interventions operate on activation copies/hooks with exception-safe removal. No-op and post-removal restoration match pristine outputs; controls match perturbation norm/site criteria frozen in protocol. Raw observations, derived summaries and causal claims are labelled separately. All writes use new transfer paths; no original weight mutation or legacy receipt overwrite.

For proposed Pythia: preserve all 50304 vocabulary logits and embedding rows. Tokenizer-unmapped IDs remain numeric IDs with explicit unavailable decoding labels. Padded fixtures use a frozen existing token fill plus masks/positions, never silently enlarge vocabulary.
