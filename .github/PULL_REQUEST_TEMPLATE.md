<!--
Thanks for sending a PR. A few reminders from CONTRIBUTING.md so review stays
fast:

- Branch name prefix: `codex/<short-topic>`.
- Imperative, under-72-char PR title.
- One logical change per PR (bundling two is OK if the description says so).
- Do NOT add `Co-Authored-By:` trailers or "Generated with ..." footers —
  attribution on this repo is owner-only.
-->

## Summary

<!-- 1-3 bullets: what changed and why. -->

## Test plan

<!--
Paste the commands you actually ran locally. At minimum, for any code change:

- [ ] `ruff format --check src/ tests/ scripts/`
- [ ] `ruff check src/ tests/ scripts/`
- [ ] `pytest tests/ -q --ignore=tests/e2e`

For viewer changes, also spot-check:
- [ ] `python -m http.server` from repo root and open the touched viewer URL.

For bundled-demo-splat changes, follow the `docs/scenes-list.json` + picker +
`tests/test_pages_assets.py` triplet in CONTRIBUTING.md § "Bundled demo splats".
-->

## Out of scope / follow-ups

<!-- Optional: anything you deliberately didn't touch so the reviewer doesn't
     ask. Pointers to follow-up issues welcome. -->
