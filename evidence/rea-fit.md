# REA fit for tiny-model reverse engineering

Status: source research completed; no installation, model download, inference, benchmarks, or runtime tests performed. Checked 2026-11-17. GitHub tree API returned main revision `d5a942177fad030d3caf0f8d39856c116e3e5cda`; its [package manifest](https://github.com/morluto/rea/blob/d5a942177fad030d3caf0f8d39856c116e3e5cda/package.json) declares `rea-agents` 6.9.0, MIT, and Node `^22.23.4 || ^24.22.1 || >=26.12.1`. This is source identity, not a verified npm installation.

## Recommendation

Reuse REA as the agent-facing artifact inventory and evidence layer. Do not select it as the mechanism-discovery engine. For a tiny model with readable implementation, prioritize source tracing and a separately instrumented model runtime; REA can inventory compatible weights and retain the resulting declarations. Architecture inferred from tensor names/shapes remains a hypothesis until matched to implementation and execution. No speed comparison was measured.

## Documented format support and limits

[Model-artifact guide](https://github.com/morluto/rea/blob/main/docs/model-artifacts.md): `inspect_model_artifact` / CLI `inspect-model-artifact` accepts local GGUF (little-endian v2/v3; big-endian v3) and safetensors. It does not execute models, interpret weights, convert files, or support PyTorch pickle/ZIP, ONNX, TFLite or Core ML in this operation. These are container capabilities, not a supported-model-family list.

GGUF metadata and tensor declarations expose names, types, shapes, offsets and declaration-derived counts. Safetensors supplies header metadata, shapes, dtypes and ranges. Metadata arrays retain up to eight preview elements. Header hashing does not identify weight contents; full hashing streams the whole file. Some GGUF quantization layouts and numeric type IDs remain ambiguous, so byte totals can remain unknown.

MCP summary delivery retains complete evidence; `inspect_analysis_view` supports paginated metadata, tensor and type queries against retained records. Such queries do not reopen the model and imply no automatic freshness. This enables an agent to inspect selected layers without putting every tensor row into context.

## Implementation evidence

The pinned [ModelArtifactProvider](https://github.com/morluto/rea/blob/d5a942177fad030d3caf0f8d39856c116e3e5cda/src/model/ModelArtifactProvider.ts) contains real GGUF/safetensors parser dispatch, file-state checks, optional full hashing, grouped parameter totals and partial-observation handling. Its emitted coverage explicitly marks tensor data as uninterpreted. Therefore the narrow inventory operation is implemented in source, not only advertised. Parser internals were not exhaustively audited.

The pinned [ModelArtifactService](https://github.com/morluto/rea/blob/d5a942177fad030d3caf0f8d39856c116e3e5cda/src/application/model/ModelArtifactService.ts) validates requests and provider outputs, checks path/hash-scope consistency, creates Evidence and projects retained summary/view results. Its source describes a shared CLI/MCP/SDK boundary.

The pinned [CLI acceptance tests](https://github.com/morluto/rea/blob/d5a942177fad030d3caf0f8d39856c116e3e5cda/tests/acceptance/modelArtifactCli.test.ts) include authored GGUF fixtures, CLI and MCP calls, retained queries, and assertions preserving unresolved quantization layouts. These tests were inspected, not run; neither their passing status nor compatibility with the user's actual files was established.

## Broader harness fit and missing work

The [README](https://github.com/morluto/rea) advertises one CLI/MCP for native binaries, JavaScript/Electron, .NET and websites; setup supports Codex, Claude Code, Cursor, Gemini CLI and others, with workflow instructions and configuration backups. Native analysis integrates existing tools such as Ghidra/IDA/Hopper/Kuna. These broader claims were not independently exercised here.

The inspected model path provides no computation-graph reconstruction, activation capture, intervention/patching, attention-head tracing or causal circuit experiments. Those require a separate framework-aware execution/probing tool. Generic binary/runtime reverse-engineering features are not evidence of transformer interpretability support. Reusing REA avoids recreating its inventory/evidence interface; extending it into a full neural introspection system would be additional engineering, not an existing verified capability.
