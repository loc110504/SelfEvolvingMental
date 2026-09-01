# Repository Guidelines



Keep this file as small as possible. Key information only.

## Project Structure &amp; Module Organization

This repository contains a preserved AgentMental baseline plus an early PsyVEC
implementation:

- `README.md` contains the project introduction.
- `main-5p.pdf` is a local PDF artifact and is currently untracked.
- `AgentMental/` is the pinned, pristine upstream baseline.
- `src/psyvec/` contains production code, beginning with model/backend ports.
- `configs/` contains versioned non-secret configuration defaults.
- `tests/` contains offline characterization and unit tests.

If implementation code is added, organize it by responsibility (for example, `src/` for production code and `tests/` for tests) and update this guide and the README with the new layout.

## Build, Test, and Development Commands

There are no project-specific build, test, or development scripts at present. For the current repository, useful checks are:

```sh
git status --short       # Review changed and untracked files
sed -n '1,160p' README.md # Inspect the project documentation
```

When adding tooling, record the canonical commands here (for example, `npm test`, `pytest`, or `make build`) and keep them runnable from the repository root.

## Coding Style &amp; Naming Conventions

No language or formatter is currently configured. New code should use four-space indentation unless its ecosystem convention requires otherwise, descriptive names, and small focused modules. Use lowercase kebab-case for standalone filenames where practical (for example, `document-parser.py`); follow the standard naming conventions of the selected language. Add a formatter or linter configuration alongside any substantial codebase addition.

## Testing Guidelines

Tests use Python's standard-library `unittest`. New behavior should include
focused tests under `tests/`, with names describing the expected outcome, such
as `test_rejects_empty_document`. Run them with:

```sh
python3 -m unittest discover -s tests -p 'test_*.py'
ruff check src tests/unit
mypy --strict src/psyvec
```

## Commit &amp; Pull Request Guidelines

The only existing commit is `first commit`, so no established message convention can be inferred. Use concise imperative subjects (for example, `Add PDF metadata parser`) and keep each commit focused.

Pull requests should explain the change, identify relevant files, describe validation performed, and link an issue when one exists. For document or visual changes, include a rendered preview or screenshots when useful. Do not commit secrets, credentials, or unnecessarily large generated files.

## Document and Asset Handling

Treat PDFs and other generated artifacts as deliberate deliverables: verify that they open correctly and confirm whether they should be tracked before committing. Keep source material separate from generated output once the project grows, and update `README.md` when adding a user-facing artifact.
