---
name: pr-summary
description: >-
  Write a PR summary in the user's flavor — fixed section outline (Summary, Context,
  Assumptions and Design Decisions, Changes, Scope / Non-Goals), ASD-STE100 Simplified
  Technical English, no meta commentary or document references. Auto-reads the git diff
  and commits against the base branch and writes PR_<number>_summary.md at the repo root.
  Use when the user asks for a PR summary or PR description.
---

# PR summary

Write a pull-request summary from the actual git changes. Follow the outline, the writing style,
and the content rules below exactly.

## 1. Gather the change context from git

Read the real changes. Do not summarize from memory or from the conversation alone.

1. Find the base branch:
   - Try `git remote show origin | sed -n 's/.*HEAD branch: //p'`.
   - If that fails, use `main`.
2. Read the commits and the diff on the current branch:
   - `git log --oneline <base>..HEAD`
   - `git diff --stat <base>...HEAD` first, then `git diff <base>...HEAD` and read the key hunks.
3. Determine the PR number for the output filename, in this order:
   1. `gh pr view --json number -q .number` — use it if a PR exists for the branch.
   2. Else estimate the next number from `gh pr list --state all --limit 1 --json number -q '.[0].number'`
      and add 1. Confirm the number with the user before you write the file.
   3. Else, if `gh` is unavailable, use the branch name.

## 2. Write the output file

Write the summary to `PR_<number>_summary.md` at the repo root. If step 3 fell back to the branch
name, write `PR_<branch>_summary.md` instead.

Use this exact outline. Keep the headings and their order.

```
## Summary

## Context

## Assumptions and Design Decisions

## Changes

## Scope / Non-Goals
```

Section content:

- **Summary:** State what the PR changes and why it matters. Two to four sentences.
- **Context:** State the problem that prompted the change. State where the change fits in the
  larger system.
- **Assumptions and Design Decisions:** One bullet per item. Start each bullet with `Assumption:`
  or `Design decision:`.
- **Changes:** One bullet per meaningful change. Derive each bullet from the diff.
- **Scope / Non-Goals:** One bullet per boundary. State what the PR does not do.

## 3. Writing style — ASD-STE100 Simplified Technical English

- Write short sentences. Keep one idea in each sentence.
- Use the active voice. Name the actor. Example: "The verifier parses the design."
- Use the simple present or the simple past tense. Do not use perfect or progressive tenses.
- Use concrete verbs, such as parses, checks, builds, solves, and calculates. Do not use
  nominalizations.
- Keep the articles ("the", "a"). Do not remove words to make the text shorter.
- Use one term for one thing. Do not use synonyms for the same concept.
- Do not use slang, idioms, or hedging words, such as "basically", "in order to", or "leverage".

## 4. Content rules — no meta commentary, no document references

- Do not reference documents, tickets, units, or task numbers.
- Do not narrate the test history as a story ("the previous tests supported…" as the subject).
- Do not explain or describe the summary itself.
- State the work. Describe where it fits. Write nothing else.

### Before and after

Do not write:

> Unit 8 assembles the product scoring path… The previous tests supported incomplete pipeline
> stages and allowed `NotImplementedError`. Units 1–7 are now available, so the tests must require
> the complete verifier behavior.

Write instead:

> The verifier converts a model completion into a deterministic score. It parses the design,
> expands the geometry, checks the design rules, builds the loads, solves the structure, calculates
> member capacity, and calculates the reward. The previous tests allowed incomplete stages and
> `NotImplementedError`. The new tests require complete scoring behavior and confirm that the
> verifier does not depend on the RL training stack.

## 5. Gold-standard example

A finished summary in this style:

```markdown
## Summary

This PR adds end-to-end tests for the complete verifier scoring path. It verifies each reward-ladder exit and confirms that a valid completion reaches the solved rung. These tests help prevent changes that break scoring behavior.

## Context

The verifier converts a model completion into a deterministic score. It parses the design, expands the geometry, checks the design rules, builds the loads, solves the structure, calculates member capacity, and calculates the reward. The previous tests allowed incomplete stages and `NotImplementedError`. The new tests require complete scoring behavior and confirm that the verifier does not depend on the RL training stack.

## Assumptions and Design Decisions

- Assumption: The shared test design remains valid for the design-rule checks and the real solver.
- Assumption: A solved design reaches rung 3, but it does not have to be feasible or low cost.
- Design decision: Inject a solver failure at the verifier boundary to test rung 2 without an actual solver fault.
- Design decision: Test the import contract in a separate Python process that rejects training-package imports.

## Changes

- Replace temporary tests for incomplete stages with assertions that all verifier stages complete.
- Add end-to-end tests for parse failure at rung 0, design-rule failure at rung 1, solver failure at rung 2, and a solved result at rung 3.
- Check the reason, score, solver inputs, and populated reward fields for the applicable exits.
- Strengthen the dependency test so that `trussRL.verifier` cannot import the training package or common RL libraries.
- Remove duplicate reward-constructor tests and permissive `NotImplementedError` handling from this integration test file.

## Scope / Non-Goals

- This PR changes verifier tests only. It does not change scoring, solver, capacity, or reward logic.
- This PR does not add calibration or training behavior, and it does not change training dependencies.
```

## 6. After you write the file

Tell the user the filename you wrote. Do not print the full summary again in chat.
