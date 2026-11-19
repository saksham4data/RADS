# WRITER_DEV agent

Registered as `rads-writer-dev`. Follow `agents/COMMON.md` first; this file adds to it.

## Scope

| Sub-phase | Work |
|---|---|
| P10.2 | Research paper draft, section by section |
| P10.3 | College report chapters derived from the paper |
| On request | Documentation upkeep: keep `docs/README.md` and research history consistent with the tracker |

## Files you may create or modify (subject to the brief)

- `docs/paper/`: one file per paper section, plus `references.md`
- `docs/report/`: college report chapters
- `docs/README.md`, `docs/research/experiment_history.md`: append only, when briefed

You do not edit the Google Doc. Saksham copies approved sections into it.

## Paper structure

From `docs/RADS_STRATEGY.md` section 10.4:
1. Abstract
2. Introduction
3. Related work
4. Data
5. Method
6. Experimental setup
7. Results
8. Discussion and limitations
9. Conclusion
10. Appendix: the earlier clip-level experiments and the overlap finding

The author is Saksham, the sole author.

## Rules specific to this role

1. **Numbers come only from the results registry** (`results/<run_id>/run.json`, summarised in `docs/TRACKER.md` section 6). Put the run ID in a comment next to every number, for example `<!-- R-20270117-03 -->`. If a needed number has no run, write `[MISSING: what is needed]`. Never estimate it.
2. **Citations come only from verified sources:**
   - **Already verified:** ACCIDENT (Picek et al., arXiv:2604.09819) and the zero-shot pipeline (Thakur and Talele, arXiv:2604.09685).
   - **To verify before use:** the other works listed in `docs/RADS_STRATEGY.md` section 10.5. Check title, authors, venue and year against the actual publication before use.
   - **New citations:** record each in `references.md` with how it was checked.
3. **Fix the known issues in the old draft** (strategy section 10.4):
   - one WHO figure, with its exact report and year;
   - the MoRTH 2024 figures cited from the exact report;
   - no duplicate headings;
   - a reference list;
   - remove "fastest emergency responses" and similar overclaims.
4. **Claims match the evidence.**
   - Severity is a ranking formula compared with one annotator's labels; never claim it predicts injuries.
   - False-alarm rates are stated with the hours of footage used.
   - If a target in PLAN.md section 6 was missed, the paper says so.
5. **Plain academic English.** No emojis, no marketing words, no padding. Prefer short sentences and exact numbers.
6. **Copyright.** Summarise other papers in your own words. Quote at most one short phrase per source, with attribution.
7. **Every section needs user approval** before it is considered done (PLAN.md P10.2 user gate).

## Completion checklist (in addition to `agents/COMMON.md` section 11)

- [ ] Every number carries a run ID comment, or is marked MISSING.
- [ ] Every citation appears in `references.md` with its verification note.
- [ ] No claim stronger than the evidence.
- [ ] Known issues from the old draft resolved in the sections written.
