# Contributing

Thanks for your interest in CompileAgent.

## Filing issues

We use GitHub Issues for all bug reports, feature requests, and
discussion. Pick the right template — they're under
[`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE/):

- **Bug report** — something is broken. Engine bug? Webapp bug? Bot
  behaviour bug?
- **Feature request** — new card-effect feature, new training knob, new
  UI affordance.

A good bug report usually includes:

- **What you tried**, exact reproduction steps.
- **What you expected** vs **what actually happened**.
- For webapp bugs: the game id (it's in the URL, e.g.
  `/games/<uuid>`) — games are deterministic replays of the saved
  action list and we can reproduce server-side from the row.
- For engine / training bugs: the run timestamp (`runs/<ts>/`), the
  seed, and the iteration if applicable. Engine games are deterministic
  given `(GameConfig.seed, action list)`.

## Pull requests

Branch protection is enabled on `main`: every change goes through a PR
that the repo owner reviews and merges. Force-pushes and direct pushes
to `main` are blocked.

Workflow:

1. Fork the repo (or create a topic branch if you have write access).
2. Make your change in a small, focused commit.
3. Run the test + build checks locally before opening the PR:

   ```bash
   # Engine + NN tests
   python -m pytest tests/ -q

   # Webapp typecheck + production build
   cd webapp
   npx tsc --noEmit
   npm run build
   ```

4. Open the PR. Use a clear title that finishes the sentence "This PR
   will …". Reference the issue it closes if there is one.
5. The repo owner reviews — be ready to iterate on review comments.
6. Once approved, the owner merges.

## Style + scope

- **Match the surrounding style.** Both the Python and TS engines use
  small, focused modules with sparingly-used comments that explain *why*
  rather than *what*. New comments should follow that pattern.
- **Tests before fixes for engine bugs.** If you fix a card effect, add
  a test in `tests/test_engine.py` that would have caught the bug. The
  test suite is the contract.
- **No silent behaviour drift in card effects.** Compile has official
  errata (the Codex, last revised 16 Dec 2024). If a card text changes,
  update `ERRATA` in `src/compile_engine/cards.py` *and* the TS mirror in
  `webapp/lib/compile/cards.ts`, both with a comment citing the source.
- **One concern per PR.** A bot bug fix should not also rename
  variables across the codebase.

## Development setup

See the [README quick-start](README.md#quick-start) — the short version:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -m pytest tests/ -q

cd webapp
cp .env.development.local.example .env.development.local   # then fill in
npx drizzle-kit migrate
npm install
npm run dev    # DEV_MODE=true auto-logs you in as user "dev"
```

If you don't have a Neon account, ask the repo owner for a dev branch
URL or stand up your own Postgres locally and set `DATABASE_URL`
accordingly. The Drizzle schema lives in `webapp/lib/db/schema.ts` and
migrations are generated with `npx drizzle-kit generate`.

## Security

If you find an issue that affects user accounts, session security, or
data isolation, **don't** open a public issue. Email the repo owner
directly (find them on the GitHub profile) so the fix can ship before
disclosure.
