# model-dissect Constitution

## Core Principles

### I. Freeze before you measure
Claims, prompts or problem sets, controls and pass marks are committed to git before the run they judge. Anything changed after seeing results is labelled as after the fact.

### II. Held-out stays held-out
Whoever chooses an edit, prompt or threshold never sees the test set. Test sets are scored once, after the choice is frozen.

### III. Controls or it did not happen
Every measured effect is reported next to a baseline and at least one like-for-like control. Every run is reported, including failures and incomplete ones.

### IV. Reproduce exactly
A result counts when a second run gives the same numbers and the artefact (edit, corpus, code) is identified by hash. Before trusting a new runner, it must reproduce a known number.

### V. Smallest thing that answers the question
One change in progress. Smoke-test every configuration end to end before a long run. No machinery that is not needed by the current step.

## Lanes
- **Exploratory:** plain scripts under `exploratory/`; results labelled exploratory, never cited as confirmed.
- **Confirm:** a one-page frozen spec, one run, one independent rerun.

## Safety
No credentials in files, prompts or commands. Model-generated code runs only in a sandbox with no network. Check GPU ownership before long jobs. Mutable data, models and caches live under `/mnt/ai`, not in git.

## Governance
Specs live in `specs/NNN-name/` and are driven with Spec Kit (spec, plan, tasks). This constitution overrides convenience; amend it by commit with a stated reason.

**Version**: 1.0.0 | **Ratified**: 2026-11-25 | **Last Amended**: 2026-11-25
