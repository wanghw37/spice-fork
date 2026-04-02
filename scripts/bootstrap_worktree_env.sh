#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Bootstrap a per-worktree conda prefix environment for SPICE.

Usage:
  scripts/bootstrap_worktree_env.sh [--python VERSION] [--clone-from PREFIX]
                                   [--skip-editable] [--skip-envrc]

Options:
  --python VERSION     Python version for a fresh env (default: 3.10)
  --clone-from PREFIX  Clone an existing conda prefix env instead of creating
                       a fresh one. The editable SPICE install is always
                       reinstalled from the current worktree afterwards.
  --skip-editable      Do not run "python -m pip install -e <worktree_root>".
  --skip-envrc         Do not create .envrc from .envrc.example.
  -h, --help           Show this help message.
EOF
}

python_version="3.10"
clone_from=""
install_editable=1
write_envrc=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      python_version="${2:?missing value for --python}"
      shift 2
      ;;
    --clone-from)
      clone_from="${2:?missing value for --clone-from}"
      shift 2
      ;;
    --skip-editable)
      install_editable=0
      shift
      ;;
    --skip-envrc)
      write_envrc=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "This script must be run inside a Git worktree." >&2
  exit 1
fi

worktree_root="$(git rev-parse --show-toplevel)"
env_prefix="${worktree_root}/.conda-env"
template_path="${worktree_root}/.envrc.example"
envrc_path="${worktree_root}/.envrc"

conda_bin="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "${conda_bin}" ]]; then
  echo "conda not found in PATH. Install or initialize conda first." >&2
  exit 1
fi

if [[ ! -d "${env_prefix}" ]]; then
  if [[ -n "${clone_from}" ]]; then
    echo "Cloning conda env from ${clone_from} to ${env_prefix}"
    "${conda_bin}" create --prefix "${env_prefix}" --clone "${clone_from}" -y
  else
    echo "Creating fresh conda env at ${env_prefix} (python=${python_version})"
    "${conda_bin}" create --prefix "${env_prefix}" "python=${python_version}" -y
  fi
else
  echo "Conda env already exists at ${env_prefix}; reusing it."
fi

if [[ "${install_editable}" -eq 1 ]]; then
  echo "Installing editable SPICE from ${worktree_root}"
  "${env_prefix}/bin/python" -m pip install -e "${worktree_root}"
fi

if [[ "${write_envrc}" -eq 1 ]]; then
  if [[ ! -f "${template_path}" ]]; then
    echo "Template not found at ${template_path}" >&2
    exit 1
  fi
  if [[ ! -f "${envrc_path}" ]]; then
    cp "${template_path}" "${envrc_path}"
    echo "Created ${envrc_path} from template."
  else
    echo "${envrc_path} already exists; leaving it unchanged."
  fi
fi

cat <<EOF

Bootstrap complete for:
  ${worktree_root}

Next steps:
  1. Install direnv if needed and add its shell hook to ~/.bashrc
  2. cd "${worktree_root}"
  3. direnv allow
  4. Verify with:
       which python
       python -c "import spice, inspect; print(inspect.getsourcefile(spice))"
EOF
