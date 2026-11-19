# Feature Specification: One feature attribution validation

**Feature Branch**: No branch created; this workspace has no Git repository at its root.  
**Created**: 2026-11-19  
**Status**: Scientific design locally frozen; execution blocked on exact identity, capability binding and independent admission review.  
**Input**: Use one hosted attribution graph to discover one feature-level hypothesis, freeze its evaluation before interventions, then test it with the existing evidence discipline. No installation or model execution is authorized by this specification.

## User Scenarios & Testing

### User Story 1 - Trace a hypothesis to an observed graph (Priority: P1)

Jimmy can inspect a concrete graph and understand exactly which feature-level claim it motivates, including what the graph cannot establish.

**Independent Test**: A reviewer can resolve the saved graph, prompt, model and feature identity without guessing any identifiers.

**Acceptance Scenarios**:

1. Given an accessible hosted graph, when a candidate is selected, then exactly one graph and one falsifiable feature claim are recorded with its selection rationale and discovery exposure.
2. Given missing graph data or an unresolved feature identity, when the record is delivered, then it explicitly remains blocked and contains no invented feature or causal finding.

### User Story 2 - Freeze the causal test (Priority: P1)

Jimmy can distinguish an exploratory observation from a test whose success and failure rules were fixed before inspecting intervention or held-out results.

**Independent Test**: A separate reviewer verifies the immutable protocol, data identities, numeric criteria and exposure timeline before any causal scoring.

**Acceptance Scenarios**:

1. Given a discovered feature, when a protocol is frozen, then the claim, effect direction, eligible data, control arms, numeric thresholds, budget and decision rule are complete and versioned.
2. Given an altered threshold or exposed held-out set, when an amendment is proposed, then the previous record is preserved and the changed claim cannot retain its original preregistration status.

### User Story 3 - Obtain an independently repeatable bounded conclusion (Priority: P2)

Jimmy receives SUPPORTED, REFUTED or INCOMPLETE for the frozen claim, with the evidence and limits necessary to assess it.

**Independent Test**: A separate executor repeats the exact admitted experiment and independently recomputes its decision from retained data.

**Acceptance Scenarios**:

1. Given all validity gates pass, when every primary criterion passes in both executions, then the scoped claim is SUPPORTED.
2. Given valid complete measurements fail a frozen primary criterion, when this is independently repeated, then the claim is REFUTED at the declared criterion, without asserting the feature is useless.
3. Given missing identity, inadequate data, invalid intervention, failed fidelity, missing repeat or exhausted budget, then the result is INCOMPLETE, never a scientific negative.

### Edge Cases

- A hosted graph exposes a feature label but omits tokenizer revision, transcoder release or token indices.
- Graph truncation hides paths; a large graph edge is mistaken for an intervention effect.
- The original and replacement computation disagree even though each runs successfully.
- Feature reconstruction error is large, or an error term inadvertently cancels an edit.
- A selected feature is inactive on evaluation prompts; baseline normalization is ill-conditioned.
- Control features cannot be matched using frozen criteria; a restart exposes held-out scores.
- Hosted execution cannot return raw evidence or allow an independent repeat.

## Requirements

### Functional Requirements

- **FR-001**: Preserve one graph's raw record and accessible view, discovery input and available generation settings; explicitly list unknown provenance.
- **FR-002**: Bind the single candidate to exact feature identity, token positions and output contrast before evaluation.
- **FR-003**: Separate discovery exposure from frozen evaluation inputs; never use evaluation scores to select the feature or thresholds.
- **FR-004**: Require numerical fidelity, reconstruction-error and intervention-locality gates before interpreting a causal effect.
- **FR-005**: Freeze ablation, donor patch, rescue, no-op, restoration and appropriate specificity controls, with exact numeric acceptance rules, before executing any intervention probe.
- **FR-006**: Preserve complete outcomes, exclusions, failed gates and resource accounting; do not weaken criteria or substitute favorable fixtures.
- **FR-007**: Require a separate reviewer and actual independent exact-candidate repeat before a scientific conclusion is accepted.
- **FR-008**: Keep previous accepted head-level work immutable and distinguish reusable evidence practices from missing feature-level execution support.
- **FR-009**: Use no installation, training or new model run under this drafting assignment. Execution needs a concrete independently reviewed capability and admission packet.

### Key Entities

- **Discovery record**: A graph, input, provenance, exposure log and selected feature.
- **Frozen protocol**: One claim, inputs, operations, controls, thresholds and decision rule bound to an immutable manifest.
- **Execution receipt**: Exact candidate and environment, raw evidence, actual resource usage and terminal status.
- **Independent verdict**: Review and repeat evidence associated with the same frozen protocol and candidate.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Exactly one selected feature and one causal claim, or an explicit blocked record with zero causal findings asserted.
- **SC-002**: Zero unresolved identity, threshold, control or data fields when declaring a protocol execution-ready or admitted. A locally frozen scientific design may explicitly retain unresolved execution binding fields only while marked EXECUTION_BLOCKED; it is not a complete execution-ready preregistration.
- **SC-003**: Every required arm and every admitted evaluation item has retained evidence and a reported outcome; omissions yield INCOMPLETE.
- **SC-004**: One independent repeat and independently recomputed decision for the exact admitted candidate before accepting a scientific verdict.
- **SC-005**: Zero changes to previously accepted scientific results and zero installations or target executions in this preparation phase.

## Assumptions

- Hosted graph access and exact identities are dependencies, not established capabilities.
- A feature is defined by its dictionary/transcoder release, layer and feature index, not by its natural-language label alone.
- The prior head-level harness supplies design patterns but is not a proven feature intervention engine.
- A valid negative is useful. Claim scope is limited to the frozen corpus and intervention; no complete-circuit or semantic-understanding claim is intended.

## Experiment contract

See [preregistration-template.md](preregistration-template.md) and [integration-boundary.md](integration-boundary.md). These are preparatory documents; [selected-claim-v1.md](selected-claim-v1.md) now binds one observed graph node and freezes numeric scientific criteria and the text corpus. Exact execution identities and independent final review remain missing; no complete executable preregistration or execution readiness is claimed. The existing root Spec Kit pointer remains unchanged under the parent's ownership boundary.
