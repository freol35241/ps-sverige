# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Engineering rigour

**Make use of a proper engineering process, apply engineering judgement and verify continuously.**

- Make sure you understand the full assignment and that you have everything you need to solve it. Raise any concerns.
- Research extensively to find state of the art methdologies, alternative approaches and methods from other areas that might be applicable. Do NOT choose the typical solution just because its typical.
- Present and discuss alternative methodologies with the human operator. 
- Approach a large task incrementally, break it down into independently solvable and verifiable parts.
- Verify in steps of incremental complexity. Start by simplistic back-of-a-napkin cases that are easily verified. Symbolic math is a good way of ensuring equation/logic correctness (SymPy). If required, build simple simulators to provide input data that can be used for verification.
- Record and expose assumptions.
- Ask for human input if ambiguous. Use visual artifacts extensively to communicate with the human.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Lab diary

A running lab diary is maintained at `diary/`. One markdown file per day. A file may contain multiple entries. Update it when:
- A design decision is made or deferred
- A modelling experiment is run and results are noted
- A new component or idea is introduced
- Something surprising is observed in the data or model behaviour

Keep entries brief and dated. The diary is a thinking tool, not a polished document.