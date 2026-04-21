# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-21

### Added
- Initial public release.
- Face, screen (UI + art), and everything detection modes.
- Heuristic quality scoring (sharpness + brightness + contrast).
- Scene-boundary adaptive frame sampling + near-duplicate deduplication.
- Background-removal via rembg for clean cutouts.
- Config-driven category system — add/remove/train categories from the web UI (`/#config`).
- Per-category CNN training with SSE live progress.
- Public-data-trained starter models (UI + Art) distributed via GitHub Releases.
- One-click Windows install via `install.bat`.
- Linux best-effort install via `install.sh`.
- Env-var overridable runtime paths for portability.

### Security
- No telemetry, no cloud calls.
- Starter models trained on CC0 / public-domain data only.
- Strings-scanned model binaries for leak-free release.

[0.1.0]: https://github.com/remvisuals/podcast-thumbnail-extractor/releases/tag/v0.1.0
