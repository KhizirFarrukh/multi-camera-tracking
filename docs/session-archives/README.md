# Session archives

One file per working session, in date order. Each is a detailed record of what
was asked, what was built, what went wrong, and why each non-obvious decision
went the way it did.

These are **history, not instructions**. For the current state of the project,
read [`../../HANDOFF.md`](../../HANDOFF.md) first — an archive describes what was
true on the day it was written, and a file it names may since have moved.

| session | covers | commits |
|---|---|---|
| [2026-09-19-stage-11.md](2026-09-19-stage-11.md) | stage 11 — vehicle detection and single-camera tracking; inherits a compacted account of stages 07–10 | `b841dde`, `dcab65b` |

## Why these exist

The handoff pack answers *where am I and what do I do next*. It deliberately
does not carry the reasoning that produced each decision, or the list of things
that were tried and failed, because a document that carried all of it would stop
being readable.

That reasoning is worth keeping anyway. Most of the sharpest comments in this
codebase exist because the first version was wrong in a way that looked right,
and those episodes are what these files record. A later session that is about to
re-make a decision can check here whether it has already been made and unmade.

## What belongs in one

- The user's instructions, verbatim.
- Every failure encountered and what actually fixed it — especially the ones
  where the first diagnosis was wrong.
- Decisions taken, with the alternative that was rejected.
- Deviations from the stage prompt, and what would close each one.
- The verification numbers as they stood at the end.

## What does not

Anything that belongs in the code. A decision worth acting on lives in a
docstring, a comment beside the line it explains, or `DECISIONS.md` — not only
here, where nobody reads it in time.
