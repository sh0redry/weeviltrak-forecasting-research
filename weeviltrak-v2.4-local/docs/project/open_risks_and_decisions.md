# Open Risks and Decisions

## Current risks

| Risk | Evidence / impact | Required decision or action |
| --- | --- | --- |
| Historical data lineage | Excel and Redshift are not one-to-one; confirmed 2020/2023/2024 date-swap batches exist | Data owner to confirm canonical source and correction process. |
| Placeholder dates | Many Redshift records use `12/31` | Exclude from direct event-date evaluation and do not train as observed events. |
| Stage/Phase transition | Raw IDs changed meaning in 2026 | Keep canonical mapping mandatory; never use raw IDs as business semantics. |
| Recent-label coverage | Three-year policy may lack direct labels for all outputs | Coverage gate blocks retraining; approve reuse of last artifact or new data plan. |
| Weather/cache dependency | Training and prediction require causal, complete engineered weather history | Validate cache coverage before model or release conclusions. |
| Artifact provenance | Local outputs and notebooks can drift from approved artifacts | Use immutable release bundle and manifest hashes. |

## Decisions already made

- v2.4 is the current default; v2.5 is deprecated compatibility only.
- Phase 1 is a direct v2.4 target with midpoint-derived historical labels.
- Phase 2 maps to Stage 3 / Phase 2.
- Recent three-year training window and coverage gate remain policy.
- Eleven curated manual events are runtime-only and never written to Redshift.

## Before changing a production policy

Record the question, data snapshot, code/config version, evaluation population,
metrics, approver, and rollback plan. Update the handoff, architecture, data
contract, and release runbook in the same change when applicable.
