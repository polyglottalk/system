# Contributing to PolyglotTalk

Thank you for your interest in contributing! This document covers branching strategy, commit conventions, code standards, and the pull request workflow.

---

## Table of Contents

1. [Branch Strategy](#branch-strategy)
2. [Commit Conventions](#commit-conventions)
3. [Development Setup](#development-setup)
4. [Adding New Features](#adding-new-features)
5. [Code Style](#code-style)
6. [Testing Requirements](#testing-requirements)
7. [Pull Request Process](#pull-request-process)
8. [Issue Tracking](#issue-tracking)

---

## Branch Strategy

The repository uses a `main` / `develop` / feature-branch model.

```
main
 └── develop
      ├── feature/add-opus-mt-fallback
      ├── fix/tts-meta-tensor-crash
      ├── experiment/benchmark-asr-models
      └── chore/update-dependencies
```

| Branch | Purpose | Merges into |
|---|---|---|
| `main` | Stable, released code. Never commit directly. | — |
| `develop` | Integration branch. All finished features land here first. | `main` (via release PR) |
| `feature/*` | New functionality. Branched from `develop`. | `develop` |
| `fix/*` | Bug fixes. Branched from `develop`. | `develop` |
| `experiment/*` | Benchmarks, research scripts, exploratory work. | `develop` |
| `chore/*` | Dependency updates, CI config, docs. | `develop` |
| `hotfix/*` | Critical production fixes only. Branched from `main`. | `main` + `develop` |

### Rules

- **Never push directly to `main` or `develop`** — always open a pull request.
- Branch names must be lowercase and hyphen-separated: `feature/multi-language-support`, not `feature/MultiLanguage`.
- Keep branches short-lived. Merge or close within a sprint.
- Rebase your branch on `develop` before opening a PR to avoid merge conflicts:
  ```bash
  git fetch origin
  git rebase origin/develop
  ```

---

## Commit Conventions

PolyglotTalk follows [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).

### Format

```
<type>(<scope>): <short summary>

[optional body]

[optional footer(s)]
```

### Types

| Type | When to use |
|---|---|
| `feat` | A new feature visible to users or downstream components |
| `fix` | A bug fix |
| `perf` | A change that improves performance without changing behaviour |
| `refactor` | Code restructuring — no feature change, no bug fix |
| `test` | Adding or correcting tests only |
| `docs` | Documentation changes only |
| `chore` | Build scripts, dependency bumps, CI changes |
| `experiment` | Research scripts, benchmark code, exploratory notebooks |
| `revert` | Reverts a previous commit |

### Scopes

Use the component name as the scope:

`asr`, `translator`, `tts`, `pipeline`, `audio`, `config`, `models`, `tests`, `ci`, `deps`

### Examples

```
feat(translator): add MarianMT fallback for unsupported language pairs

fix(tts): defer MMS-TTS model loading to run() to prevent initialization issues

perf(asr): set OMP_NUM_THREADS=2 to reduce context-switching overhead

test(pipeline): add e2e mock test asserting 2+ translated outputs within 15s

docs(readme): add WSL2 PulseAudio setup instructions

chore(deps): bump faster-whisper to 1.2.0
```

### Rules

- Summary line: **imperative mood**, **no full stop**, max **72 characters**.
- Body: explain *why*, not *what*. Wrap at 72 characters.
- Reference issues in the footer: `Closes #42`, `Fixes #17`.
- One logical change per commit — avoid mixing unrelated changes.

---

## Development Setup

### Prerequisites

- Python 3.11.x
- [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`
- System packages:
  ```bash
  sudo apt-get install -y libportaudio2 portaudio19-dev pulseaudio libpulse0 alsa-utils
  ```

### Create a virtual environment

```bash
# Using uv (recommended)
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt

# Using pip
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Download models

```bash
python setup_models.py
```

---

## Adding New Features

1. **Open or find an issue** describing the feature before writing code.

2. **Branch from `develop`**:
   ```bash
   git fetch origin
   git checkout develop
   git pull origin develop
   git checkout -b feature/<short-description>
   ```

3. **Implement** — keep changes focused. One feature per branch.

4. **Add tests** in `tests/` before marking the PR ready (see [Testing Requirements](#testing-requirements)).

5. **Update documentation** — if the feature changes behaviour, update `README.md` or `IMPLEMENTATION_PLAN.md` accordingly.

6. **Rebase and push**:
   ```bash
   git fetch origin
   git rebase origin/develop
   git push origin feature/<short-description>
   ```

7. **Open a pull request** into `develop` (see [Pull Request Process](#pull-request-process)).

---

## Code Style

- **Formatter**: [`ruff format`](https://docs.astral.sh/ruff/) — line length 100.
- **Linter**: `ruff check` — no warnings allowed in CI.
- **Type hints**: required for all public functions and class methods.
- **Docstrings**: Google-style for all public classes and functions.

Run locally before pushing:

```bash
ruff format .
ruff check .
```

### Project-specific conventions

- All queue items must be **dataclasses** (`AudioChunk`, `TextSegment`, `TranslatedSegment`) carrying at minimum `chunk_id: int` and `timestamp: float`.
- Use `queue.put(item, timeout=1.0)` inside a loop that checks `stop_event` — never blocking `put()` calls.
- All threads must be **daemon threads** so `Ctrl+C` exits cleanly.
- Model loading belongs in the thread's `run()` method, never in `__init__()`, to avoid HuggingFace meta-tensor crashes.
- Environment variable overrides (`OMP_NUM_THREADS`, `CT2_INTER_THREADS`) must be set in `config.py` before any imports.

---

## Testing Requirements

All contributions must include or update the relevant tests under `tests/`.

| Test file | What it covers |
|---|---|
| `test_audio_capture.py` | Mic recording — WAV size and RMS |
| `test_asr.py` | Transcription of a known WAV clip |
| `test_translator.py` | Translation output contains expected characters |
| `test_tts.py` | TTS synthesis produces a non-zero 24 kHz WAV |
| `test_context.py` | Context window logic, edge cases, fuzzy trim |
| `test_pipeline_e2e.py` | Full pipeline with mocked audio input |

### Running tests

```bash
pytest tests/ -v
```

### Coverage expectation

- New code paths must be covered by at least one test.
- All existing tests must continue to pass — no regressions.

### Benchmark / experiment scripts

Place under `benchmarks/` and use `experiment/*` branches. These are not part of the standard test suite but must not break on import.

---

## Pull Request Process

1. **Title** — follow the same `type(scope): summary` format as commit messages.
2. **Description** — fill in:
   - What changed and why.
   - How to test it manually.
   - Any known limitations or follow-up work.
3. **Link the issue** — use `Closes #N` in the PR description.
4. **Pass CI** — all tests and linting checks must be green before requesting review.
5. **One approval required** — at least one reviewer must approve before merging.
6. **Squash or rebase merge** — no merge commits into `develop`. Squash if the branch has noisy WIP commits; rebase if each commit is already clean and meaningful.
7. **Delete the branch** after merging.

### PR checklist

```
- [ ] Branched from `develop`, not `main`
- [ ] Commits follow Conventional Commits format
- [ ] `ruff format` and `ruff check` pass locally
- [ ] All existing tests pass (`pytest tests/ -v`)
- [ ] New tests added for new behaviour
- [ ] README / docs updated if behaviour changed
- [ ] Issue linked in PR description
```

---

## Issue Tracking

- **Bug reports**: include Python version, OS, full traceback, and steps to reproduce.
- **Feature requests**: describe the problem being solved, not just the solution.
- **Label conventions**:

  | Label | Meaning |
  |---|---|
  | `bug` | Something is broken |
  | `enhancement` | New feature or improvement |
  | `experiment` | Research / benchmarking task |
  | `good first issue` | Suitable for first-time contributors |
  | `blocked` | Waiting on an external dependency or decision |
  | `wontfix` | Out of scope or intentionally not addressed |

---

## Questions?

Open a discussion or issue — we're happy to help.
