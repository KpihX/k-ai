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

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_INSTALL_PROFILE="${PROJECT_DIR}/install/install.yaml"
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

if command -v uv >/dev/null 2>&1; then
  ok "uv $(uv --version 2>/dev/null | head -1)"
else
  fail "uv not found. Install uv first: https://docs.astral.sh/uv/"
  exit 1
fi

if [[ ! -f "${INSTALL_PROFILE}" ]]; then
  fail "Install profile not found: ${INSTALL_PROFILE}"
  exit 1
fi
ok "Install profile ${INSTALL_PROFILE}"

if command -v bun >/dev/null 2>&1; then
  ok "bun $(bun --version 2>/dev/null)"
elif command -v npm >/dev/null 2>&1; then
  ok "npm $(npm --version 2>/dev/null)"
else
  warn "Neither bun nor npm found. QMD auto-install will be skipped."
fi

step "Syncing Python dependencies"

cd "${PROJECT_DIR}"
uv sync --dev
uv tool install --editable "${PROJECT_DIR}" --force >/dev/null
ok "Editable CLI installed"

step "Loading installation profile"

eval "$(
  INSTALL_PROFILE="${INSTALL_PROFILE}" uv run python - <<'PY'
import os, shlex, yaml
from pathlib import Path

path = Path(os.environ["INSTALL_PROFILE"]).expanduser()
data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

def q(value: str) -> str:
    return shlex.quote(str(value))

interactive = data.get("interactive", {}) or {}
runtime_store = data.get("runtime_store", {}) or {}
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
print(f"K_AI_DIR={q(runtime_store.get('home_dir', '~/.k-ai'))}")
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
print(f"VERIFY_RUN_DOCTOR={q(str(bool(verification.get('run_doctor', True))).lower())}")
PY
)"

K_AI_DIR="$(python3 -c 'from pathlib import Path; import os; print(Path(os.environ["K_AI_DIR"]).expanduser())' 2>/dev/null)"
SESSIONS_DIR="${K_AI_DIR}/sessions"
SANDBOX_DIR="$(python3 -c 'from pathlib import Path; import os; print(Path(os.environ["PYTHON_SANDBOX_DIR"]).expanduser())' 2>/dev/null)"
MEMORY_FILE="${K_AI_DIR}/MEMORY.json"
CONFIG_FILE="${K_AI_DIR}/config.yaml"
HOOK_FILE="${K_AI_DIR}/.git/hooks/post-commit"
EDITOR_CHOICE=""
SELECTED_PACKAGES=()

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
  uv run python -c 'from k_ai.config import ConfigManager; print(ConfigManager.get_default_yaml(), end="")' > "${CONFIG_FILE}"
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
elif [[ "${EDITOR_OFFER_MICRO}" == "true" ]] && [[ "${INSTALL_INTERACTIVE}" == "true" ]] && ask_yes_no "Install micro as the default editor for k-ai config editing?" "y"; then
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

uv run python -c '
from k_ai.config import ConfigManager
cm = ConfigManager(override_path=r"'"${CONFIG_FILE}"'")
cm.set("config.editor", r"'"${EDITOR_CHOICE}"'")
cm.set("tools.python_exec.sandbox_dir", r"'"${SANDBOX_DIR}"'")
cm.save_active_yaml(r"'"${CONFIG_FILE}"'")
' >/dev/null
ok "Configured config.editor=${EDITOR_CHOICE}"

if [[ ! -f "${K_AI_DIR}/.gitignore" ]]; then
  printf 'sandbox/\n*.tmp\n' > "${K_AI_DIR}/.gitignore"
  ok "Created ${K_AI_DIR}/.gitignore"
fi

step "Initializing ~/.k-ai git repo"

if [[ ! -d "${K_AI_DIR}/.git" ]]; then
  git -C "${K_AI_DIR}" init -q
  git -C "${K_AI_DIR}" add -A
  git -C "${K_AI_DIR}" commit -q -m "init: k-ai runtime store" || true
  ok "Initialized git repo in ${K_AI_DIR}"
else
  ok "Git repo already present in ${K_AI_DIR}"
fi

step "Preparing Python sandbox"

if [[ "${PYTHON_SANDBOX_ENABLED}" != "true" ]]; then
  warn "Python sandbox disabled by install profile"
else
  if [[ ! -x "${SANDBOX_DIR}/bin/python" ]]; then
    python3 -m venv "${SANDBOX_DIR}"
    ok "Sandbox created at ${SANDBOX_DIR}"
  else
    ok "Sandbox already present"
  fi

  read -r -a PROFILE_PACKAGES <<< "${PYTHON_DEFAULT_PACKAGES}"
  if [[ "${INSTALL_INTERACTIVE}" == "true" && "${INSTALL_ASK_DEFAULT_PACKAGES}" == "true" ]]; then
    for pkg in "${PROFILE_PACKAGES[@]}"; do
      if ask_yes_no "Install sandbox package '${pkg}'?" "y"; then
        SELECTED_PACKAGES+=("${pkg}")
      fi
    done
  else
    SELECTED_PACKAGES=("${PROFILE_PACKAGES[@]}")
  fi

  if [[ "${INSTALL_INTERACTIVE}" == "true" && "${INSTALL_ALLOW_EXTRA_PACKAGES}" == "true" ]]; then
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
    PACKAGES_JSON="$(printf '%s\n' "${UNIQUE_PACKAGES[@]}" | uv run python - <<'PY'
import json, sys
items = [line.strip() for line in sys.stdin if line.strip()]
print(json.dumps(items))
PY
)"
    PACKAGES_JSON="${PACKAGES_JSON}" uv run python -c '
import json
import os
from k_ai.config import ConfigManager
cm = ConfigManager(override_path=r"'"${CONFIG_FILE}"'")
cm.set("tools.python_exec.default_packages", json.loads(os.environ["PACKAGES_JSON"]))
cm.save_active_yaml(r"'"${CONFIG_FILE}"'")
' >/dev/null
  else
    warn "No sandbox packages selected; python_exec sandbox was created without extra packages."
  fi
fi

step "Setting up QMD"

if [[ "${QMD_INSTALL_IF_MISSING}" == "true" ]] && ! command -v qmd >/dev/null 2>&1; then
  if command -v bun >/dev/null 2>&1; then
    bun install -g qmd >/dev/null
  elif command -v npm >/dev/null 2>&1; then
    npm install -g qmd >/dev/null
  fi
fi

if command -v qmd >/dev/null 2>&1; then
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

step "Running verification"

if [[ "${VERIFY_PREFER_MAKE}" == "true" ]] && [[ -f "${PROJECT_DIR}/Makefile" ]] && command -v make >/dev/null 2>&1; then
  make check
else
  python3 -m py_compile src/k_ai/*.py src/k_ai/tools/*.py src/k_ai/ui/*.py test/*.py
  uv run pytest -q
fi

if [[ "${VERIFY_RUN_DOCTOR}" == "true" ]]; then
  uv run k-ai doctor || warn "Doctor reported issues. Review the output above."
fi

echo
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${GREEN}  k-ai installation complete${RESET}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "  Install profile: ${CYAN}${INSTALL_PROFILE}${RESET}"
echo -e "  Install docs:    ${CYAN}${PROJECT_DIR}/install/README.md${RESET}"
echo -e "  Runtime config:  ${CYAN}${CONFIG_FILE}${RESET}"
echo -e "  Chat:            ${CYAN}k-ai chat${RESET}"
echo -e "  Check:           ${CYAN}make check${RESET} ${DIM}(preferred)${RESET}"
echo -e "  Status:          ${CYAN}k-ai chat${RESET} then ${CYAN}/status${RESET}"
