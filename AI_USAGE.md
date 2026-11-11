# AI Usage, Team 24

The brief asks us to declare the AI coding assistants we used and what we used them for.

## Kavan Siddesh Kumar
**Tool:** Claude Code (Claude Opus 5.5, Anthropic).

**What it did:**
- Read the brief and the Part 1 files, and helped organize the repo layout, the work split and the shared file formats (`TEAM_WORKPLAN.md`, `PARTNER_LLM_PROMPT.md`).
- Wrote and refined code in `src/`, `tools/` and `tests/`, and generated Kavan's notebooks from it.
- Wrote the shared golden-set scorer (`src/p1_score.py`, `notebooks/p1_eval.ipynb`).
- Drafted `eval/p1_test_prompts.csv`, `eval/p3_scenarios.json` and `eval/p3_red_team_cases.json`.
- Debugged errors with offline tests that use fake models.
- Drafted documentation, markdown wording and draft report text, which Kavan rewrites in his own words.

**What it did not do:**
- It did not do any human evaluation. Ratings and pass/fail judgments come from people.
- It did not decide personal contributions.
- It did not run the GPU work. Kavan runs the notebooks.

**What Kavan did:** [TO BE WRITTEN BY KAVAN: what you reviewed, changed, ran and verified, and what you rewrote in your own words]

**Things it got wrong, and how they were found:**
- The first table parser read values like `3.0` as `30.0`, because PyMuPDF's table extraction scrambled the characters. Printing the parsed rows showed it, and the parser now reads each cell from word positions.
- With the pinned PyMuPDF 1.25.5, the parser treated a table's caption row as a year column. Tables 1-2 to 1-4 produced no facts, and every Table 1-1 row was labelled "World". Kerned captions also hid Table A-3. The offline test failed on China's 2009 growth, and printing the parsed facts showed why. Both are fixed.

## Muhammed Faiq Salman
**Tool:** [TO BE WRITTEN BY MUHAMMED: tool name and version]

**What it did:** [TO BE WRITTEN BY MUHAMMED]

**What it did not do:** [TO BE WRITTEN BY MUHAMMED]
