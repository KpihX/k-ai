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

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
K_AI_DIR="${HOME}/.k-ai"
SESSIONS_DIR="${K_AI_DIR}/sessions"
SANDBOX_DIR="${K_AI_DIR}/sandbox"
MEMORY_FILE="${K_AI_DIR}/MEMORY.json"
CONFIG_FILE="${K_AI_DIR}/config.yaml"
HOOK_FILE="${K_AI_DIR}/.git/hooks/post-commit"

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

step "Preparing ~/.k-ai"

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

if [[ ! -x "${SANDBOX_DIR}/bin/python" ]]; then
  python3 -m venv "${SANDBOX_DIR}"
  "${SANDBOX_DIR}/bin/pip" install -q numpy sympy scipy pandas matplotlib seaborn scikit-learn
  ok "Sandbox created with scientific packages"
else
  ok "Sandbox already present"
fi

step "Setting up QMD"

if ! command -v qmd >/dev/null 2>&1; then
  if command -v bun >/dev/null 2>&1; then
    bun install -g qmd >/dev/null
  elif command -v npm >/dev/null 2>&1; then
    npm install -g qmd >/dev/null
  fi
fi

if command -v qmd >/dev/null 2>&1; then
  ok "QMD available"
  if qmd collection show k-ai >/dev/null 2>&1; then
    ok "QMD collection 'k-ai' already exists"
  else
    qmd collection add "${SESSIONS_DIR}" --name k-ai --mask '**/*.{jsonl,json}'
    qmd context add "${SESSIONS_DIR}" \
      "k-ai session history — JSONL message logs (user/assistant/tool exchanges) and session metadata. Search past conversations, decisions, code, context."
    ok "Created QMD collection 'k-ai'"
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

uv run pytest -q
uv run k-ai doctor || warn "Doctor reported issues. Review the output above."

echo
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${GREEN}  k-ai installation complete${RESET}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "  Chat:        ${CYAN}k-ai chat${RESET}"
echo -e "  Status:      ${CYAN}k-ai chat${RESET} then ${CYAN}/status${RESET}"
echo -e "  Config save: ${CYAN}/config save${RESET}"
echo -e "  Purge:       ${CYAN}./scripts/purge.sh${RESET}"
