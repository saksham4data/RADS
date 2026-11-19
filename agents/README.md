# RADS agent team

Instructions for the agents that execute `docs/PLAN.md`. Every agent follows `agents/COMMON.md` plus its own role file.

## Team

| Agent | Registered name | Instruction file | Phases |
|---|---|---|---|
| Manager | `rads-manager` | `MANAGER.md` | All. Coordinates and records; writes no code |
| QA | `rads-qa` | `QA.md` | Checks every sub-phase; fixes nothing |
| Data developer | `rads-data-dev` | `DATA_DEV.md` | P2, P3.1, P3.2, P3.4 |
| Evaluation developer | `rads-eval-dev` | `EVAL_DEV.md` | P1, P4, P9.3 |
| Model developer | `rads-model-dev` | `MODEL_DEV.md` | P3.3, P5, P6, P9.1, P9.2 |
| Systems developer | `rads-systems-dev` | `SYSTEMS_DEV.md` | P7, P8, P10.1 |
| Writer | `rads-writer-dev` | `WRITER_DEV.md` | P10.2, P10.3, documentation upkeep |

The registered definitions live in `.claude/agents/rads-*.md`. That folder is gitignored, so it exists only on this machine. Each definition tells the agent to read the files below before doing anything.

## Reading order for every agent

1. `docs/PLAN.md`
2. `docs/TRACKER.md`
3. `agents/COMMON.md`, then the agent's own file in this folder
4. `docs/RADS_STRATEGY.md`
5. `docs/research/decisions.md`, `docs/research/experiment_history.md`, `docs/research/dataset_lineage.md`
6. Files named under "Inputs" in the assigned sub-phase

## Cycle for one sub-phase

```text
Manager: check dependencies and gates
   │
   ▼
Manager: write the brief in TRACKER.md section 3, status IN PROGRESS
   │
   ▼
Dev agent: read the required files, do the work, return a completion report
   │
   ▼
Manager: log the report in TRACKER.md section 4, status IN QA
   │
   ▼
QA: re-run the evidence, return PASS or FAIL
   │
   ├── PASS ─► Manager: status DONE, update registry, tell the user it is ready to commit
   │
   └── FAIL ─► Manager: new brief with the findings, same dev agent (after 3 FAILs: escalate to the user)
```

At a user gate, the manager stops and asks the user before anything that depends on it.

## How to start

- In Claude Code, ask: "Run the RADS manager" or "Use the rads-manager agent to continue the plan." The manager reads the tracker and continues from the current state.
- To run one sub-phase by hand, launch the named dev agent with: "Do sub-phase P<x.y>. Read the reading-order files and your brief in docs/TRACKER.md section 3 first." Then launch `rads-qa` on the same sub-phase.
- If the `rads-*` agents are not listed, restart Claude Code once. The `.claude/agents/` folder was created during a running session.

## Rules that never change

- Agents never commit, push or change branches. Saksham reviews and commits.
- No test data is used outside the evaluation steps named in the plan.
- Every number comes from a logged run.
- No emojis, no filler.
