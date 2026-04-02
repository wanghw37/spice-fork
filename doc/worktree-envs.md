# Per-Worktree Conda Environments with direnv

This repository supports a development workflow where each Git worktree owns
its own prefix conda environment at `./.conda-env`.

This avoids a common failure mode with editable installs: if multiple worktrees
share one conda environment, `pip install -e .` can leave the environment
pointing at the wrong source tree.

## Why this layout

- `main` and each linked worktree get isolated Python/package state
- `spice` resolves to the code in the current worktree, not a sibling tree
- environments are disposable and easy to rebuild
- `direnv` auto-activates the right env when you enter a worktree

## Prerequisites

- conda or mamba available in your shell
- `direnv` installed
- `eval "$(direnv hook bash)"` added to `~/.bashrc`

Example install commands for `direnv`:

```bash
# Ubuntu/Debian
sudo apt-get install direnv

# macOS
brew install direnv
```

## First-time setup for a worktree

From the root of the target worktree:

```bash
scripts/bootstrap_worktree_env.sh
direnv allow
```

This will:

- create `./.conda-env` if missing
- install SPICE editable from the current worktree
- create a local `.envrc` from `.envrc.example` if missing

## Faster setup via a shared base env

If you already have a heavy base environment with common dependencies, clone it
into the worktree before reinstalling SPICE editable:

```bash
scripts/bootstrap_worktree_env.sh --clone-from /abs/path/to/base-env
direnv allow
```

The bootstrap script always reinstalls editable SPICE from the current
worktree, so cloned environments do not keep stale editable mappings.

## Validation

After entering the worktree:

```bash
which python
python -c "import spice, inspect; print(inspect.getsourcefile(spice))"
```

The reported source path should live under the current worktree root.

## Creating a new linked worktree

```bash
git worktree add .worktrees/my-feature -b my-feature
cd .worktrees/my-feature
scripts/bootstrap_worktree_env.sh
direnv allow
```

## Rebuilding a broken worktree env

From the worktree root:

```bash
conda env remove --prefix ./.conda-env
rm -rf .direnv .envrc
scripts/bootstrap_worktree_env.sh
direnv allow
```

## Rules of thumb

- Never run `pip install -e .` from one worktree into another worktree's env
- Keep `.envrc`, `.direnv/`, and `.conda-env/` untracked
- Treat each worktree env as local and disposable
