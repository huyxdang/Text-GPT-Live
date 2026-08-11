# Progress log

## Task: Make the public repository clone-ready

**Status:** complete
**Started:** 2026-08-11 12:20 ICT

### Scope

- Make a clean clone understandable, runnable, and testable without hidden local files.
- Align public documentation and configuration with the released g1 model.
- Verify GitHub and Hugging Face release artifacts and remove only proven obsolete paths.

### 2026-08-11 12:20 - Baseline audit

**Status:** complete

**Completed**

- Verified the deterministic runtime starts and returns healthy session responses.
- Tested a tracked-files-only clone.
- Inspected the public model and dataset repositories.

**Evidence**

- Clean-clone test result: 440 passed, 77 failed, 5 errors.
- Failures referenced unpublished sample fixtures, the data specification, and ignored compiled JSONL files.
- Hugging Face model storage was 8.67 GB and included a complete 5.1 GB model plus an orphan 3.2 GB shard.

**Decisions**

- Keep lightweight tests self-contained and make corpus-dependent checks skip clearly until the full build runs.
- Provide a deterministic no-model development path and a separate Apple Silicon MLX release path.

### 2026-08-11 12:40 - Clean-clone test boundary repaired

**Status:** complete

**Completed**

- Restored four small hand-authored sample corpora and the system-prompt specification required by compiler tests.
- Added `requirements-dev.txt` with the test and dataset-build dependencies.
- Made only four full-corpus repair checks skip with an actionable build command when generated train/dev files are absent.

**Evidence**

- Clean-tree suite before a corpus build: 515 passed, 4 skipped.
- Fixture directories are byte-identical to their tracked development sources; tests now use the current `docs/data.md` specification.

**Decisions**

- Keep the 123 MB compiled train/dev corpus out of Git; it remains reproducible and published separately.
- Check in the small stable fixture inputs used by unit-level compiler tests.

### 2026-08-11 13:05 - Public runtime and documentation aligned

**Status:** complete

**Completed**

- Removed the abandoned V5 editor route, client, writer provider, runtime state, tests, and Anthropic dependency.
- Replaced public `SMOL_*` configuration with current `LOCAL_*` and `SYSTEM_TLS` names; historical remote checkpoint names remain unchanged for provenance.
- Added the Apple Silicon MLX quick start, exact three-stage training commands, current architecture/data/training docs, and a historical banner on the pre-training acceptance plan.
- Added Python 3.11 GitHub Actions CI.
- Removed the operator-specific Tinker checkpoint embedded in the stage-3 builder; it now resolves stage 2 from `data/tinker/run_state.json` or explicit CLI arguments.

**Evidence**

- `python3 -m compileall -q app scripts train datagen tests`: passed.
- `git diff --check`: passed.
- The documented deterministic server started, `/health` returned 200, and `/api/sessions` created a session.
- Removed more than 2,100 lines from obsolete/conflicting surfaces while preserving g1 execution and training.

**Decisions**

- Keep V4/V6 and the mixed historical training module for now: current g1 training helpers still import that lineage, so safe removal requires a dedicated extraction.
- Keep the seven-action g1 contract as the current release documentation; preserve the original five-demo plan only as an explicitly archived artifact.

### 2026-08-11 13:35 - Reproduction and release artifacts verified

**Status:** complete

**Completed**

- Installed only `requirements-dev.txt` in an isolated environment.
- Rebuilt the complete stage-1 corpus from tracked authored sources.
- Re-ran the entire suite with generated train/dev files present.
- Verified the Hugging Face `model.safetensors` hash matches the locally evaluated checkpoint and loads without the orphan shard.
- Created Hugging Face backup tag `pre-cleanup-2026-08-11` and opened model PR #1 containing the orphan-shard deletion.

**Evidence**

- Full build: 6,688 rows from 6,740 inputs; 6,013 train, 675 dev; zero train/dev overlap; five over-limit rows removed.
- Full isolated suite after build: 519 passed in 9.58s.
- Hugging Face model SHA-256: `0e8c3abc70a643a168eedcebbf9606884c029b0908bd5a1c8fe39381f7603369`.
- The orphan `model-00001-of-00002.safetensors` is 3,215,610,503 bytes and is referenced by no index entry.

**Next**

- Merge Hugging Face model PR #1 only after explicit approval for the public deletion.

**Blockers**

- The Hub cleanup PR is safely staged but its public deletion requires explicit approval.

### 2026-08-11 14:05 - Final release verification

**Status:** complete

**Completed**

- Independent release review completed with no remaining Critical or Important findings.
- Created the release commit and verified it from a separate clean clone.

**Evidence**

- Fresh clone before corpus build: 515 passed, 4 expected skips.
- Fresh clone full build: 6,688 rows; 6,013 train, 675 dev; zero overlap; maximum 62,223 target tokens.
- The five previously skipped corpus checks all passed after the build.
- GitHub Actions run `31463541407` passed on hosted Python 3.11.

**Blockers**

- None for the GitHub repository release.
