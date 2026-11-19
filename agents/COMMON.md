# Rules for every RADS agent

These rules apply to the manager, QA and every dev agent. Your role file (`agents/<ROLE>.md`) adds to them; it never relaxes them.

## 1. Read before you work

Before any action, read in this order:
1. `docs/PLAN.md`
2. `docs/TRACKER.md`
3. `agents/COMMON.md` (this file), then your own `agents/<ROLE>.md`
4. `docs/RADS_STRATEGY.md`
5. `docs/research/decisions.md`, `docs/research/experiment_history.md`, `docs/research/dataset_lineage.md`
6. Every file named under "Inputs" in your sub-phase and in your assignment brief

List the files you read at the top of your completion report.

`docs/archive/` contains superseded documents. Do not follow them. Read one only if your brief points to it.

## 2. Scope

- Work only on the sub-phase in your assignment brief (`docs/TRACKER.md` section 3).
- Create or modify only the files the brief allows. If you need another file, stop and ask the manager.
- Do not edit `docs/PLAN.md`. Propose changes to the manager, who asks the user.
- Do not start the next sub-phase on your own.

## 3. Data integrity

1. **Source-video separation.** Every clip, window and fold from one source video stays in one split. Use the source-ID rule written in P2.2.
2. **No test data outside evaluation.** Never read test-split labels or test videos in training, feature selection, threshold choice or debugging. Only evaluation scripts run on test splits, and only in the sub-phases that say so (P4.1, P4.2, P5.7, P8.2, P9.2 final table, P9.3).
3. **One test run per final model.** If a test result disappoints, do not tune and re-run. Report it, and the manager raises a user gate.
4. **Do not train on `data/clips/`.** The P01/P02 clips are invalid by design (decision D008).
5. **`data/raw/` is read-only.** Never modify, move or re-encode raw videos or their metadata.

## 4. Evidence

- **Every number you report comes from a command you actually ran.** Give the command and the key output.
- **Every metric is saved as a run** in the results registry (`results/<run_id>/run.json`, defined in P1.4), with a run ID of the form `R-<YYYYMMDD>-<nn>`.
- **Never estimate, round up, or invent a result.** If you could not run or verify something, write "not verified" and say why.
- **Published numbers** (for example the ACCIDENT baselines) are labelled as published, with the source.
- **Report failures and missed targets plainly.**

## 5. Reproducibility

- Fix random seeds and record them.
- Save the full config next to every output, together with the git commit (`git rev-parse HEAD`) and whether the working tree had uncommitted changes.
- Version model files and caches: `v1`, `v2`, and so on. Never overwrite an earlier version.
- Record library versions for any run that produces a model or a metric.

## 6. Version control

- **Never** run `git commit`, `git push`, `git checkout`, `git switch`, `git reset`, `git rebase`, `git stash` or `git merge`, and never create or delete a branch or tag. The user commits after review.
- Read-only git commands (`status`, `diff`, `log`, `show`, `rev-parse`, `check-ignore`) are allowed.

## 7. Deleting things

Delete nothing outside your own outputs from the current sub-phase, unless the manager has written approval for that path in `docs/TRACKER.md`.

## 8. Code standards

- Match the style of the surrounding code: naming, comment density, structure.
- Add tests in `tests/` for new logic. `python -m pytest tests` must pass before you report completion.
- Keep runtime code (`rads/`) free of experiment-only code. Experiments live in `scripts/` or notebooks.
- One implementation per concept. Feature extraction used in training and in the live runtime must be the same code.

## 9. Writing style

- Plain technical English. No emojis. No filler, no marketing words, no exclamation marks.
- State facts with their source. Mark assumptions as assumptions.
- Dates as YYYY-MM-DD.

## 10. Long and heavy jobs

- **This machine has a 4-core CPU and no GPU.**
- **Jobs over about 2 minutes** run in the background with a log file. Report the log path.
- **Jobs that need a GPU** (track caching, detector fine-tuning) are prepared as Kaggle notebooks for the user to run. Write the exact steps for the user.

## 11. Completion report

Return this to the manager when you finish. The manager copies it into the work log.

```text
Sub-phase: P<x.y>   Agent: <ROLE>   Attempt: <n>   Date: YYYY-MM-DD
Files read: <list>
Done: <what was done, in a few sentences>
Files created or modified: <list>
Commands run: <command> -> <key output>
Acceptance criteria:
  1. <criterion from PLAN.md> -> MET / NOT MET -> <evidence>
  2. ...
Run IDs: <list or none>
Tests: python -m pytest tests -> <result>
Open issues: <list or none>
```

## 12. When to stop and escalate

Stop and report to the manager, without guessing, when:
- your brief is unclear or contradicts `docs/PLAN.md`;
- an acceptance criterion cannot be met;
- a result contradicts the plan or an earlier recorded result;
- you need a file outside your allowed set;
- something would require test data outside an evaluation step;
- a tool, dataset or download is unavailable.
