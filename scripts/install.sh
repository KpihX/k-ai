#!/usr/bin/env bash

set -euo pipefail

CYAN='\033[1;36m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
DIM='\033[2m'
RESET='\033[0m'

info() { echo -e "${CYAN}[k-ai]${RESET} $*"; }
ok() { echo -e "${GREEN}  OK${RESET} $*"; }
warn() { echo -e "${YELLOW}  WARN${RESET} $*"; }
fail() { echo -e "${RED}  ERR${RESET} $*"; }
step() { echo -e "\n${CYAN}━━━ $* ━━━${RESET}"; }

HAS_TTY_INPUT="false"
if [[ -t 0 ]]; then
  HAS_TTY_INPUT="true"
fi

ask_yes_no() {
  local prompt="$1"
  local default="${2:-y}"
  local suffix="[Y/n]"
  if [[ "${default}" == "n" ]]; then
    suffix="[y/N]"
  fi
  if [[ ! -t 0 ]]; then
    [[ "${default}" == "y" ]] && return 0 || return 1
  fi
  read -r -p "$(echo -e "${CYAN}[k-ai]${RESET} ${prompt} ${suffix} ")" reply
  reply="${reply:-$default}"
  [[ "${reply}" =~ ^[Yy]$ ]]
}

prompt_line() {
  local prompt="$1"
  local reply=""
  if [[ ! -t 0 ]]; then
    printf '%s' ""
    return 0
  fi
  read -r -p "$(echo -e "${CYAN}[k-ai]${RESET} ${prompt} ")" reply
  printf '%s' "${reply}"
}

expand_path() {
  python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser())' "$1"
}

ensure_venv() {
  local venv_path="$1"
  if [[ ! -x "${venv_path}/bin/python" ]]; then
    python3 -m venv "${venv_path}"
  fi
}

install_uv_binary() {
  local installer_url="$1"
  local installer_shell=""
  if command -v curl >/dev/null 2>&1; then
    installer_shell="curl -LsSf '${installer_url}' | sh"
  elif command -v wget >/dev/null 2>&1; then
    installer_shell="wget -qO- '${installer_url}' | sh"
  else
    warn "Neither curl nor wget is available, so uv cannot be installed automatically."
    return 1
  fi
  sh -c "${installer_shell}"
  export PATH="${HOME}/.local/bin:${PATH}"
  command -v uv >/dev/null 2>&1
}

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_INSTALL_PROFILE="${PROJECT_DIR}/install/install.yaml"
DEFAULT_PROFILE_LOADER_VENV="${HOME}/.cache/k-ai-installer-profile"
INSTALL_PROFILE_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--path)
      if [[ $# -gt 1 && ! "$2" =~ ^- ]]; then
        INSTALL_PROFILE_ARG="$2"
        shift 2
      else
        INSTALL_PROFILE_ARG="defaults"
        shift
      fi
      ;;
    *)
      fail "Unknown argument: $1"
      echo "Usage: ./scripts/install.sh [-p|--path [defaults|/path/to/install.yaml]]"
      exit 1
      ;;
  esac
done

INSTALL_PROFILE="${DEFAULT_INSTALL_PROFILE}"
if [[ -n "${INSTALL_PROFILE_ARG}" && "${INSTALL_PROFILE_ARG}" != "defaults" ]]; then
  INSTALL_PROFILE="${INSTALL_PROFILE_ARG}"
fi

PROFILE_LOADER_PYTHON="python3"
INSTALL_BACKEND=""
BOOTSTRAP_VENV=""
BOOTSTRAP_BIN_DIR=""
SELECTED_PACKAGES=()
EDITOR_CHOICE=""

run_profile_python() {
  "${PROFILE_LOADER_PYTHON}" "$@"
}

run_managed_python() {
  if [[ "${INSTALL_BACKEND}" == "uv" ]]; then
    uv run python "$@"
  else
    "${BOOTSTRAP_VENV}/bin/python" "$@"
  fi
}

run_managed_pytest() {
  if [[ "${INSTALL_BACKEND}" == "uv" ]]; then
    uv run pytest -q "$@"
  else
    "${BOOTSTRAP_VENV}/bin/python" -m pytest -q "$@"
  fi
}

run_managed_kai() {
  if [[ "${INSTALL_BACKEND}" == "uv" ]]; then
    uv run k-ai "$@"
  else
    "${BOOTSTRAP_VENV}/bin/python" -m k_ai.main "$@"
  fi
}

step "Checking prerequisites"

if ! command -v python3 >/dev/null 2>&1; then
  fail "python3 not found"
  exit 1
fi

PY_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PY_MAJOR="${PY_VERSION%%.*}"
PY_MINOR="${PY_VERSION##*.}"
if [[ "${PY_MAJOR}" -lt 3 || "${PY_MINOR}" -lt 12 ]]; then
  fail "Python ${PY_VERSION} found, need >= 3.12"
  exit 1
fi
ok "Python ${PY_VERSION}"

if [[ ! -f "${INSTALL_PROFILE}" ]]; then
  fail "Install profile not found: ${INSTALL_PROFILE}"
  exit 1
fi
ok "Install profile ${INSTALL_PROFILE}"

if python3 -c 'import yaml' >/dev/null 2>&1; then
  ok "System Python already provides PyYAML for loading install profiles"
else
  PROFILE_LOADER_VENV="$(expand_path "${DEFAULT_PROFILE_LOADER_VENV}")"
  ensure_venv "${PROFILE_LOADER_VENV}"
  "${PROFILE_LOADER_VENV}/bin/python" -m pip install -q --upgrade pip pyyaml
  PROFILE_LOADER_PYTHON="${PROFILE_LOADER_VENV}/bin/python"
  ok "Prepared isolated profile loader env at ${PROFILE_LOADER_VENV}"
fi

if command -v bun >/dev/null 2>&1; then
  ok "bun $(bun --version 2>/dev/null)"
elif command -v npm >/dev/null 2>&1; then
  ok "npm $(npm --version 2>/dev/null)"
else
  warn "Neither bun nor npm found. QMD auto-install will be skipped."
fi

step "Loading installation profile"

eval "$(
  INSTALL_PROFILE="${INSTALL_PROFILE}" run_profile_python - <<'PY'
import os
import shlex
from pathlib import Path

import yaml

path = Path(os.environ["INSTALL_PROFILE"]).expanduser()
data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

def q(value: str) -> str:
    return shlex.quote(str(value))

interactive = data.get("interactive", {}) or {}
bootstrap = data.get("bootstrap", {}) or {}
runtime_store = data.get("runtime_store", {}) or {}
runtime_git = data.get("runtime_git", {}) or {}
capabilities = data.get("capabilities", {}) or {}
editor = data.get("editor", {}) or {}
python_sandbox = data.get("python_sandbox", {}) or {}
qmd = data.get("qmd", {}) or {}
verification = data.get("verification", {}) or {}

default_packages = []
for item in python_sandbox.get("default_packages", []) or []:
    text = str(item or "").strip()
    if text and text not in default_packages:
        default_packages.append(text)

print(f"INSTALL_INTERACTIVE={q(str(bool(interactive.get('enabled', True))).lower())}")
print(f"INSTALL_ASK_DEFAULT_PACKAGES={q(str(bool(interactive.get('ask_default_python_packages_one_by_one', True))).lower())}")
print(f"INSTALL_ALLOW_EXTRA_PACKAGES={q(str(bool(interactive.get('allow_extra_python_packages', True))).lower())}")
print(f"BOOTSTRAP_PREFER_UV={q(str(bool(bootstrap.get('prefer_uv', True))).lower())}")
print(f"BOOTSTRAP_OFFER_UV_INSTALL={q(str(bool(bootstrap.get('offer_uv_install', True))).lower())}")
print(f"BOOTSTRAP_INSTALL_UV_BY_DEFAULT={q(str(bool(bootstrap.get('install_uv_by_default', True))).lower())}")
print(f"BOOTSTRAP_ALLOW_PIP_FALLBACK={q(str(bool(bootstrap.get('allow_pip_fallback', True))).lower())}")
print(f"BOOTSTRAP_VENV_DIR={q(bootstrap.get('bootstrap_venv', '~/.k-ai/bootstrap-cli'))}")
print(f"BOOTSTRAP_BIN_DIR={q(bootstrap.get('bin_dir', '~/.local/bin'))}")
print(f"BOOTSTRAP_UV_INSTALLER_URL={q(bootstrap.get('uv_installer_url', 'https://astral.sh/uv/install.sh'))}")
print(f"K_AI_DIR={q(runtime_store.get('home_dir', '~/.k-ai'))}")
print(f"RUNTIME_GIT_ENABLED={q(str(bool(runtime_git.get('enabled', True))).lower())}")
print(f"RUNTIME_GIT_AUTO_COMMIT_ON_EXIT={q(str(bool(runtime_git.get('auto_commit_on_chat_exit', True))).lower())}")
print(f"RUNTIME_GIT_COMMIT_PREFIX={q(runtime_git.get('commit_prefix', 'chat:'))}")
print(f"RUNTIME_GIT_INITIAL_COMMIT_MESSAGE={q(runtime_git.get('initial_commit_message', 'init: k-ai runtime store'))}")
print(f"CAP_EXA={q(str(bool(capabilities.get('exa', True))).lower())}")
print(f"CAP_PYTHON={q(str(bool(capabilities.get('python', True))).lower())}")
print(f"CAP_SHELL={q(str(bool(capabilities.get('shell', True))).lower())}")
print(f"CAP_QMD={q(str(bool(capabilities.get('qmd', True))).lower())}")
print(f"EDITOR_PREFERRED={q(editor.get('preferred', ''))}")
print(f"EDITOR_OFFER_MICRO={q(str(bool(editor.get('offer_micro_install', True))).lower())}")
print(f"EDITOR_FALLBACK={q(editor.get('fallback', 'nano'))}")
print(f"PYTHON_SANDBOX_ENABLED={q(str(bool(python_sandbox.get('enabled', True))).lower())}")
print(f"PYTHON_SANDBOX_DIR={q(python_sandbox.get('sandbox_dir', '~/.k-ai/sandbox'))}")
print(f"PYTHON_DEFAULT_PACKAGES={q(' '.join(default_packages))}")
print(f"QMD_INSTALL_IF_MISSING={q(str(bool(qmd.get('install_if_missing', True))).lower())}")
print(f"QMD_COLLECTION_NAME={q(qmd.get('collection_name', 'k-ai'))}")
print(f"QMD_COLLECTION_MASK={q(qmd.get('collection_mask', '**/*.{jsonl,json}'))}")
print(f"QMD_CONTEXT_SUMMARY={q(qmd.get('context_summary', 'k-ai session history.'))}")
print(f"VERIFY_PREFER_MAKE={q(str(bool(verification.get('prefer_make_check', True))).lower())}")
print(f"VERIFY_ENABLED={q(str(bool(verification.get('enabled', True))).lower())}")
print(f"VERIFY_RUN_DOCTOR={q(str(bool(verification.get('run_doctor', True))).lower())}")
PY
)"

K_AI_DIR="$(expand_path "${K_AI_DIR}")"
SESSIONS_DIR="${K_AI_DIR}/sessions"
MEMORY_FILE="${K_AI_DIR}/MEMORY.json"
CONFIG_FILE="${K_AI_DIR}/config.yaml"
HOOK_FILE="${K_AI_DIR}/.git/hooks/post-commit"
SANDBOX_DIR="$(expand_path "${PYTHON_SANDBOX_DIR}")"
BOOTSTRAP_VENV="$(expand_path "${BOOTSTRAP_VENV_DIR}")"
BOOTSTRAP_BIN_DIR="$(expand_path "${BOOTSTRAP_BIN_DIR}")"
RUNTIME_GITIGNORE_TEMPLATE="${PROJECT_DIR}/install/.gitignore.runtime"

step "Selecting installation backend"

if [[ "${BOOTSTRAP_PREFER_UV}" != "true" ]]; then
  if [[ "${BOOTSTRAP_ALLOW_PIP_FALLBACK}" != "true" ]]; then
    fail "The install profile disables uv and also disables the isolated pip fallback."
    exit 1
  fi
  INSTALL_BACKEND="bootstrap"
  warn "Install profile prefers the isolated bootstrap backend instead of uv"
elif command -v uv >/dev/null 2>&1; then
  INSTALL_BACKEND="uv"
  ok "uv $(uv --version 2>/dev/null | head -1)"
else
  warn "uv is not currently available."
  info "Preferred path: install uv now and use it for dependency sync and command shims."
  info "Fallback path: keep uv absent and use an isolated bootstrap virtualenv at ${BOOTSTRAP_VENV}."

  if [[ "${BOOTSTRAP_OFFER_UV_INSTALL}" == "true" ]]; then
    INSTALL_UV_DEFAULT="n"
    if [[ "${BOOTSTRAP_INSTALL_UV_BY_DEFAULT}" == "true" ]]; then
      INSTALL_UV_DEFAULT="y"
    fi
    if ask_yes_no "Install uv now? yes = preferred managed install, no = isolated pip fallback without touching system Python." "${INSTALL_UV_DEFAULT}"; then
      if install_uv_binary "${BOOTSTRAP_UV_INSTALLER_URL}"; then
        INSTALL_BACKEND="uv"
        ok "uv installed successfully"
      else
        warn "uv installation failed; continuing with the fallback decision path."
      fi
    fi
  fi

  if [[ -z "${INSTALL_BACKEND}" ]]; then
    if [[ "${BOOTSTRAP_ALLOW_PIP_FALLBACK}" != "true" ]]; then
      fail "uv is unavailable and pip fallback is disabled by the install profile."
      exit 1
    fi
    INSTALL_BACKEND="bootstrap"
    warn "Using isolated bootstrap virtualenv fallback at ${BOOTSTRAP_VENV}"
  fi
fi

step "Preparing managed Python environment"

cd "${PROJECT_DIR}"

if [[ "${INSTALL_BACKEND}" == "uv" ]]; then
  uv sync --dev
  uv tool install --editable "${PROJECT_DIR}" --force >/dev/null
  ok "Editable CLI installed with uv"
else
  ensure_venv "${BOOTSTRAP_VENV}"
  "${BOOTSTRAP_VENV}/bin/python" -m pip install -q --upgrade pip setuptools wheel
  "${BOOTSTRAP_VENV}/bin/python" -m pip install -q -e "${PROJECT_DIR}" pytest pytest-asyncio
  ok "Editable CLI installed in isolated bootstrap env"

  mkdir -p "${BOOTSTRAP_BIN_DIR}"
  cat > "${BOOTSTRAP_BIN_DIR}/k-ai" <<EOF
#!/usr/bin/env bash
exec "${BOOTSTRAP_VENV}/bin/python" -m k_ai.main "\$@"
EOF
  chmod +x "${BOOTSTRAP_BIN_DIR}/k-ai"
  ok "Installed k-ai launcher at ${BOOTSTRAP_BIN_DIR}/k-ai"

  case ":${PATH}:" in
    *":${BOOTSTRAP_BIN_DIR}:"*) ;;
    *)
      warn "${BOOTSTRAP_BIN_DIR} is not in PATH for this shell. Add it if you want 'k-ai' to resolve globally."
      ;;
  esac
fi

step "Preparing runtime store"

mkdir -p "${SESSIONS_DIR}"
if [[ ! -f "${MEMORY_FILE}" ]]; then
  mkdir -p "$(dirname "${MEMORY_FILE}")"
  printf '{"version": 1, "entries": []}\n' > "${MEMORY_FILE}"
  ok "Created ${MEMORY_FILE}"
else
  ok "Memory file already exists"
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
  run_managed_python -c 'from k_ai.config import ConfigManager; print(ConfigManager.get_default_yaml(), end="")' > "${CONFIG_FILE}"
  ok "Created ${CONFIG_FILE}"
else
  ok "Config file already exists"
fi

step "Choosing default editor"

if [[ -n "${EDITOR_PREFERRED}" ]] && command -v "${EDITOR_PREFERRED%% *}" >/dev/null 2>&1; then
  EDITOR_CHOICE="${EDITOR_PREFERRED}"
  ok "Using preferred editor from install profile: ${EDITOR_CHOICE}"
elif command -v micro >/dev/null 2>&1; then
  EDITOR_CHOICE="micro"
  ok "micro already available"
elif [[ "${EDITOR_OFFER_MICRO}" == "true" ]] && [[ "${INSTALL_INTERACTIVE}" == "true" ]] && [[ "${HAS_TTY_INPUT}" == "true" ]]; then
  info "Editor choices:"
  info "  - yes: install and use micro as the default editor"
  info "  - no: keep the system editor chain (K_AI_EDITOR / VISUAL / EDITOR / ${EDITOR_FALLBACK})"
  if ask_yes_no "Install micro as the default editor for k-ai config editing?" "y"; then
    if command -v apt-get >/dev/null 2>&1; then
      if command -v sudo >/dev/null 2>&1; then
        sudo apt-get update && sudo apt-get install -y micro
      else
        apt-get update && apt-get install -y micro
      fi
      if command -v micro >/dev/null 2>&1; then
        EDITOR_CHOICE="micro"
        ok "micro installed"
      else
        warn "micro installation did not produce a usable binary"
      fi
    else
      warn "apt-get not available; cannot auto-install micro"
    fi
  fi
fi

if [[ "${RUNTIME_GIT_ENABLED}" == "true" ]] && ! command -v git >/dev/null 2>&1; then
  warn "git is not available. Runtime git tracking will be disabled for this installation."
  RUNTIME_GIT_ENABLED="false"
fi

if [[ -z "${EDITOR_CHOICE}" ]]; then
  if [[ -n "${K_AI_EDITOR:-}" ]]; then
    EDITOR_CHOICE="${K_AI_EDITOR}"
  elif [[ -n "${VISUAL:-}" ]]; then
    EDITOR_CHOICE="${VISUAL}"
  elif [[ -n "${EDITOR:-}" ]]; then
    EDITOR_CHOICE="${EDITOR}"
  else
    EDITOR_CHOICE="${EDITOR_FALLBACK:-nano}"
  fi
fi

if [[ "${INSTALL_INTERACTIVE}" == "true" ]]; then
  step "Choosing initial tool capabilities"
  info "Capability choices:"
  info "  - exa: public web search through Exa"
  info "  - python: sandboxed Python execution plus sandbox package management"
  info "  - shell: sandboxed shell command execution"
  info "  - qmd: indexed history and knowledge retrieval tools"
  info "Protected approval-admin tools are not changed here; they remain YAML-only if you want to alter them manually."

  if ask_yes_no "Enable Exa web search capability?" "$([[ "${CAP_EXA}" == "true" ]] && echo y || echo n)"; then
    CAP_EXA="true"
  else
    CAP_EXA="false"
  fi
  if ask_yes_no "Enable Python sandbox capability?" "$([[ "${CAP_PYTHON}" == "true" ]] && echo y || echo n)"; then
    CAP_PYTHON="true"
  else
    CAP_PYTHON="false"
  fi
  if ask_yes_no "Enable shell sandbox capability?" "$([[ "${CAP_SHELL}" == "true" ]] && echo y || echo n)"; then
    CAP_SHELL="true"
  else
    CAP_SHELL="false"
  fi
  if ask_yes_no "Enable QMD knowledge capability?" "$([[ "${CAP_QMD}" == "true" ]] && echo y || echo n)"; then
    CAP_QMD="true"
  else
    CAP_QMD="false"
  fi
fi

CONFIG_FILE="${CONFIG_FILE}" \
EDITOR_CHOICE="${EDITOR_CHOICE}" \
K_AI_DIR="${K_AI_DIR}" \
SESSIONS_DIR="${SESSIONS_DIR}" \
MEMORY_FILE="${MEMORY_FILE}" \
SANDBOX_DIR="${SANDBOX_DIR}" \
RUNTIME_GIT_ENABLED="${RUNTIME_GIT_ENABLED}" \
RUNTIME_GIT_AUTO_COMMIT_ON_EXIT="${RUNTIME_GIT_AUTO_COMMIT_ON_EXIT}" \
RUNTIME_GIT_COMMIT_PREFIX="${RUNTIME_GIT_COMMIT_PREFIX}" \
CAP_EXA="${CAP_EXA}" \
CAP_PYTHON="${CAP_PYTHON}" \
CAP_SHELL="${CAP_SHELL}" \
CAP_QMD="${CAP_QMD}" \
QMD_COLLECTION_NAME="${QMD_COLLECTION_NAME}" \
run_managed_python - <<'PY' >/dev/null
import os

from k_ai.config import ConfigManager


def as_bool(name: str) -> bool:
    return os.environ[name].strip().lower() == "true"


cm = ConfigManager(override_path=os.environ["CONFIG_FILE"])
cm.set("config.persist_path", os.environ["CONFIG_FILE"])
cm.set("memory.internal_file", os.environ["MEMORY_FILE"])
cm.set("sessions.directory", os.environ["SESSIONS_DIR"])
cm.set("config.editor", os.environ["EDITOR_CHOICE"])
cm.set("runtime_git.enabled", as_bool("RUNTIME_GIT_ENABLED"))
cm.set("runtime_git.auto_commit_on_chat_exit", as_bool("RUNTIME_GIT_AUTO_COMMIT_ON_EXIT"))
cm.set("runtime_git.commit_prefix", os.environ["RUNTIME_GIT_COMMIT_PREFIX"])
cm.set("tools.exa.enabled", as_bool("CAP_EXA"))
cm.set("tools.python.enabled", as_bool("CAP_PYTHON"))
cm.set("tools.python.sandbox_dir", os.environ["SANDBOX_DIR"])
cm.set("tools.shell.enabled", as_bool("CAP_SHELL"))
cm.set("tools.qmd.enabled", as_bool("CAP_QMD"))
cm.set("tools.qmd.session_collection", os.environ["QMD_COLLECTION_NAME"])
cm.save_active_yaml(os.environ["CONFIG_FILE"])
PY
ok "Configured config.editor=${EDITOR_CHOICE}"

if [[ "${RUNTIME_GIT_ENABLED}" == "true" ]]; then
  if [[ -f "${RUNTIME_GITIGNORE_TEMPLATE}" ]]; then
    cp "${RUNTIME_GITIGNORE_TEMPLATE}" "${K_AI_DIR}/.gitignore"
    ok "Installed managed runtime .gitignore in ${K_AI_DIR}"
  else
    warn "Runtime gitignore template not found: ${RUNTIME_GITIGNORE_TEMPLATE}"
  fi

  step "Initializing ~/.k-ai git repo"

  if [[ ! -d "${K_AI_DIR}/.git" ]]; then
    git -C "${K_AI_DIR}" init -q
    git -C "${K_AI_DIR}" config user.name "k-ai runtime"
    git -C "${K_AI_DIR}" config user.email "runtime@k-ai.local"
    git -C "${K_AI_DIR}" add .
    if git -C "${K_AI_DIR}" diff --cached --quiet --exit-code; then
      ok "Initialized git repo in ${K_AI_DIR} (nothing to commit)"
    else
      git -C "${K_AI_DIR}" commit -q -m "${RUNTIME_GIT_INITIAL_COMMIT_MESSAGE}"
      ok "Initialized git repo in ${K_AI_DIR}"
    fi
  else
    git -C "${K_AI_DIR}" config user.name "k-ai runtime"
    git -C "${K_AI_DIR}" config user.email "runtime@k-ai.local"
    git -C "${K_AI_DIR}" add .
    if git -C "${K_AI_DIR}" diff --cached --quiet --exit-code; then
      ok "Git repo already present in ${K_AI_DIR}; runtime state already clean"
    else
      git -C "${K_AI_DIR}" commit -q -m "${RUNTIME_GIT_INITIAL_COMMIT_MESSAGE}"
      ok "Git repo already present in ${K_AI_DIR}; runtime state committed"
    fi
  fi
else
  warn "Runtime git tracking disabled by install profile"
fi

step "Preparing Python sandbox"

if [[ "${PYTHON_SANDBOX_ENABLED}" != "true" || "${CAP_PYTHON}" != "true" ]]; then
  warn "Python sandbox setup skipped because the install profile or chosen capability disabled it"
else
  if [[ ! -x "${SANDBOX_DIR}/bin/python" ]]; then
    python3 -m venv "${SANDBOX_DIR}"
    ok "Sandbox created at ${SANDBOX_DIR}"
  else
    ok "Sandbox already present"
  fi

  read -r -a PROFILE_PACKAGES <<< "${PYTHON_DEFAULT_PACKAGES}"
  if [[ "${INSTALL_INTERACTIVE}" == "true" && "${INSTALL_ASK_DEFAULT_PACKAGES}" == "true" && "${#PROFILE_PACKAGES[@]}" -gt 0 ]]; then
    info "Default sandbox package choices:"
    info "  - yes: install the proposed package into the dedicated Python sandbox"
    info "  - no: skip it for now"
    for pkg in "${PROFILE_PACKAGES[@]}"; do
      if ask_yes_no "Install sandbox package '${pkg}'?" "y"; then
        SELECTED_PACKAGES+=("${pkg}")
      fi
    done
  else
    SELECTED_PACKAGES=("${PROFILE_PACKAGES[@]}")
  fi

  if [[ "${INSTALL_INTERACTIVE}" == "true" && "${INSTALL_ALLOW_EXTRA_PACKAGES}" == "true" ]]; then
    info "Extra sandbox packages:"
    info "  - type one package name per prompt to add it"
    info "  - press Enter on an empty line when you are done"
    while true; do
      extra_pkg="$(prompt_line "Extra sandbox package to add (empty to continue):")"
      extra_pkg="$(printf '%s' "${extra_pkg}" | xargs 2>/dev/null || true)"
      if [[ -z "${extra_pkg}" ]]; then
        break
      fi
      SELECTED_PACKAGES+=("${extra_pkg}")
    done
  fi

  if [[ "${#SELECTED_PACKAGES[@]}" -gt 0 ]]; then
    mapfile -t UNIQUE_PACKAGES < <(printf '%s\n' "${SELECTED_PACKAGES[@]}" | awk 'NF && !seen[$0]++')
    "${SANDBOX_DIR}/bin/pip" install -q "${UNIQUE_PACKAGES[@]}"
    ok "Sandbox packages installed: ${UNIQUE_PACKAGES[*]}"
    PACKAGES_JSON="$(printf '%s\n' "${UNIQUE_PACKAGES[@]}" | run_managed_python - <<'PY'
import json
import sys

items = [line.strip() for line in sys.stdin if line.strip()]
print(json.dumps(items))
PY
)"
    PACKAGES_JSON="${PACKAGES_JSON}" run_managed_python -c '
import json
import os
from k_ai.config import ConfigManager

cm = ConfigManager(override_path=r"'"${CONFIG_FILE}"'")
cm.set("tools.python.default_packages", json.loads(os.environ["PACKAGES_JSON"]))
cm.save_active_yaml(r"'"${CONFIG_FILE}"'")
' >/dev/null
  else
    warn "No sandbox packages selected; the dedicated Python sandbox was created without extra packages."
  fi
fi

step "Setting up QMD"

if [[ "${CAP_QMD}" == "true" && "${QMD_INSTALL_IF_MISSING}" == "true" ]] && ! command -v qmd >/dev/null 2>&1; then
  if command -v bun >/dev/null 2>&1; then
    bun install -g qmd >/dev/null
  elif command -v npm >/dev/null 2>&1; then
    npm install -g qmd >/dev/null
  fi
fi

if [[ "${CAP_QMD}" != "true" ]]; then
  warn "QMD capability disabled by install choices"
elif command -v qmd >/dev/null 2>&1; then
  ok "QMD available"
  if qmd collection show "${QMD_COLLECTION_NAME}" >/dev/null 2>&1; then
    ok "QMD collection '${QMD_COLLECTION_NAME}' already exists"
  else
    qmd collection add "${SESSIONS_DIR}" --name "${QMD_COLLECTION_NAME}" --mask "${QMD_COLLECTION_MASK}"
    qmd context add "${SESSIONS_DIR}" "${QMD_CONTEXT_SUMMARY}"
    ok "Created QMD collection '${QMD_COLLECTION_NAME}'"
  fi
else
  warn "QMD unavailable. History search tools will remain limited until QMD is installed."
fi

step "Installing runtime git hook"

if [[ "${RUNTIME_GIT_ENABLED}" == "true" ]]; then
  mkdir -p "$(dirname "${HOOK_FILE}")"
  cat > "${HOOK_FILE}" <<'HOOK'
#!/usr/bin/env bash
set -euo pipefail
if command -v qmd >/dev/null 2>&1; then
  (qmd update >/dev/null 2>&1 || true)
fi
HOOK
  chmod +x "${HOOK_FILE}"
  ok "Installed ${HOOK_FILE}"
else
  warn "Runtime git hook skipped because runtime git tracking is disabled"
fi

step "Running verification"

if [[ "${VERIFY_ENABLED}" == "true" ]]; then
  if [[ "${VERIFY_PREFER_MAKE}" == "true" && "${INSTALL_BACKEND}" == "uv" ]] && [[ -f "${PROJECT_DIR}/Makefile" ]] && command -v make >/dev/null 2>&1; then
    make check
  else
    python3 -m py_compile src/k_ai/*.py src/k_ai/tools/*.py src/k_ai/ui/*.py test/*.py
    run_managed_pytest
  fi

  if [[ "${VERIFY_RUN_DOCTOR}" == "true" ]]; then
    run_managed_kai doctor || warn "Doctor reported issues. Review the output above."
  fi
else
  warn "Verification disabled by install profile"
fi

echo
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${GREEN}  k-ai installation complete${RESET}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "  Install profile: ${CYAN}${INSTALL_PROFILE}${RESET}"
echo -e "  Install docs:    ${CYAN}${PROJECT_DIR}/install/README.md${RESET}"
echo -e "  Backend:         ${CYAN}${INSTALL_BACKEND}${RESET}"
if [[ "${INSTALL_BACKEND}" == "bootstrap" ]]; then
  echo -e "  Launcher:        ${CYAN}${BOOTSTRAP_BIN_DIR}/k-ai${RESET}"
fi
echo -e "  Runtime config:  ${CYAN}${CONFIG_FILE}${RESET}"
echo -e "  Chat:            ${CYAN}k-ai chat${RESET}"
echo -e "  Check:           ${CYAN}make check${RESET} ${DIM}(preferred when uv is available)${RESET}"
echo -e "  Status:          ${CYAN}k-ai chat${RESET} then ${CYAN}/status${RESET}"
