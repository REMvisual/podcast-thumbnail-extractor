# CLAUDE.md — PodLab | Thumbnail Extractor

This is the PUBLIC release repo. Its contents are portable to any user's checkout location and are published to a GitHub remote.

## Rules

1. **No hardcoded absolute paths.** All runtime paths must be relative to the package root or env-var-overridable. Pre-commit: `grep -rnE 'C:[\\/]|D:[\\/]|/Users/[A-Za-z]+|%USERPROFILE%[\\/][A-Za-z]'` over shipped files must return zero hits.
2. **This repo has a GitHub remote.** `git push origin master` is expected.
3. **Commit messages are public.** Do NOT include internal tracking IDs. Use conventional commits: `feat:`, `fix:`, `docs:`, `chore:`, `test:`.
4. **Privacy gate:** before every commit, grep staged files against your personal-identifier denylist (keep that list local/gitignored, not in this repo). Zero hits required.
5. **Python 3.10+.** `torch` and `torchvision` require it.
6. **Tests:** `pytest` from the repo root. Keep them fast; no network or filesystem side effects beyond `tmp_path`.
7. **DO NOT edit files outside this folder** when working in a parent workspace — this repo is self-contained and portable.

## Runtime path env vars (portability)

| Variable | Default | Purpose |
|---|---|---|
| `THUMBNAIL_EXTRACTOR_UPLOAD_DIR` | `./uploads` | Video upload scratch |
| `THUMBNAIL_EXTRACTOR_OUTPUT_DIR` | `./outputs` | Extracted thumbnail output |
| `THUMBNAIL_EXTRACTOR_MODEL_DIR` | `./models` | CNN model storage |
| `THUMBNAIL_EXTRACTOR_CONFIG_PATH` | `./config/categories.json` | Category config |
| `THUMBNAIL_EXTRACTOR_PORT` | `5000` | Flask port |
