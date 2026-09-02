# Documentation Guide

This directory is the engineering documentation surface. The current production
configuration is **v2.4** (`app/config/weeviltrak_v2.4.yml`). Historical plans
are retained for provenance but must not be used as production instructions.

## Start here

1. [New engineer onboarding guide](project/new_engineer_onboarding_guide.md) —
   project explanation, recommended reading/code path, safe first-week plan,
   and engineering boundaries.
2. [2026 project handoff](project/2026_project_handoff.md) — current state,
   onboarding order, ownership boundaries, and open work.
3. [v2.4 architecture](architecture/weeviltrak_model_process_overview.md) —
   data flow, targets, features, Base/termination layers, and outputs.
4. [Historical data contract and risks](data/historical_data_contract_and_risks.md)
   — Redshift/Excel/S3 roles and known data-quality constraints.
5. [v2.4 training and release runbook](operations/v24_training_release_runbook.md)
   — safe training, validation, artifact, and release workflow.
6. [Open risks and decisions](project/open_risks_and_decisions.md) — items that
   need an owner or a data-team decision.

## Documentation map

| Directory | Status and purpose |
| --- | --- |
| `architecture/` | **Current:** engineering architecture and data flow. |
| `data/` | **Current:** data contracts, lineage, quality findings, and safety rules. |
| `models/` | **Current + historical:** version registry and design notes. See [model versions](models/model_versions.md). |
| `operations/` | **Current + historical:** runbooks, release, and backtesting procedures. Read titles carefully. |
| `project/` | **Current + historical:** handoff, risks, protocols, and project context. |
| `investigations/` | **Evidence:** point-in-time diagnostic work; not production policy. |
| `archive/implementation/` | **Historical:** superseded implementation plans; never use as current instructions. |

## Client delivery versus engineering operation

[`manual/`](../manual/README.md) is the client-facing v2.4 delivery surface.
It covers immutable artifact bundles, local pickle inference, daily input/output
contracts, and deployment warm-up. It does not replace the engineering runbook:
the app pipeline needs Redshift/S3/weather access and is responsible for
training, backtesting, and release creation.

## Maintenance rule

When a change affects a production v2.4 target, feature schema, data source,
termination policy, artifact format, or release workflow, update the relevant
current document above in the same change. Mark obsolete documents as
historical rather than silently leaving them to appear current.
