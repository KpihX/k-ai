#!/usr/bin/env bash

set -euo pipefail

RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
CYAN='\033[1;36m'
RESET='\033[0m'

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
K_AI_DIR="${K_AI_RUNTIME_DIR:-${HOME}/.k-ai}"
PURGE_YES=0
QMD_COLLECTION_NAME=""

info() { echo -e "${CYAN}[k-ai]${RESET} $*"; }
ok() { echo -e "${GREEN}  OK${RESET} $*"; }
warn() { echo -e "${YELLOW}  WARN${RESET} $*"; }
fail() { echo -e "${RED}  ERR${RESET} $*"; }

PURGE_QMD=0
PURGE_UV_TOOL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --qmd) PURGE_QMD=1; shift ;;
    --uv-tool) PURGE_UV_TOOL=1; shift ;;
    --yes|-y) PURGE_YES=1; shift ;;
    --runtime-dir)
      shift
      if [[ $# -eq 0 ]]; then
        fail "--runtime-dir requires a path"
        exit 1
      fi
      K_AI_DIR="$1"
      shift
      ;;
    *)
      fail "Unknown option: $1"
      echo "Usage: ./scripts/purge.sh [--yes] [--runtime-dir PATH] [--qmd] [--uv-tool]"
      exit 1
      ;;
  esac
done

expand_path() {
  python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser())' "$1"
}

resolve_distribution_name() {
  python3 - <<PY
from pathlib import Path
import tomllib

path = Path(r"${PROJECT_DIR}") / "pyproject.toml"
try:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    print(data.get("project", {}).get("name", "kpihx-ai"))
except Exception:
    print("kpihx-ai")
PY
}

resolve_qmd_collection_name() {
  local config_path="$1"
  python3 - <<PY
from pathlib import Path
import re

path = Path(r"${config_path}")
if not path.exists():
    print("k-ai")
    raise SystemExit
text = path.read_text(encoding="utf-8")
match = re.search(r"^[ \\t]*session_collection:[ \\t]*[\"']?([^\"'\\n#]+)", text, re.MULTILINE)
print(match.group(1).strip() if match else "k-ai")
PY
}

K_AI_DIR="$(expand_path "${K_AI_DIR}")"
if [[ -z "${K_AI_DIR}" || "${K_AI_DIR}" == "/" || "${K_AI_DIR}" == "${HOME}" || "${K_AI_DIR}" == "${PROJECT_DIR}" ]]; then
  fail "Refusing to purge unsafe runtime path: ${K_AI_DIR}"
  exit 1
fi

if [[ -f "${K_AI_DIR}/config.yaml" ]]; then
  QMD_COLLECTION_NAME="$(resolve_qmd_collection_name "${K_AI_DIR}/config.yaml")"
else
  QMD_COLLECTION_NAME="k-ai"
fi

echo -e "${RED}This will remove runtime data from ${K_AI_DIR}.${RESET}"
echo "Included by default:"
echo "  - sessions"
echo "  - memory"
echo "  - sandbox"
echo "  - persisted config"
echo "  - runtime git repo/hooks"
if [[ "${PURGE_QMD}" -eq 1 ]]; then
  echo "  - QMD collection '${QMD_COLLECTION_NAME}' if available"
fi
if [[ "${PURGE_UV_TOOL}" -eq 1 ]]; then
  echo "  - editable uv tool install for k-ai"
fi
if [[ "${PURGE_YES}" -eq 1 ]]; then
  answer="y"
elif [[ ! -t 0 ]]; then
  warn "Non-interactive purge aborted. Re-run with --yes to confirm."
  exit 0
else
  printf "Continue? [y/N] "
  read -r answer
  if [[ ! "${answer}" =~ ^[Yy]$ ]]; then
    warn "Aborted."
    exit 0
  fi
fi

if [[ -d "${K_AI_DIR}" ]]; then
  rm -rf "${K_AI_DIR}"
  ok "Removed ${K_AI_DIR}"
else
  warn "${K_AI_DIR} not present"
fi

if [[ "${PURGE_QMD}" -eq 1 ]] && command -v qmd >/dev/null 2>&1; then
  if qmd collection show "${QMD_COLLECTION_NAME}" >/dev/null 2>&1; then
    qmd collection remove "${QMD_COLLECTION_NAME}" || warn "Failed to remove QMD collection '${QMD_COLLECTION_NAME}'"
    ok "Removed QMD collection '${QMD_COLLECTION_NAME}'"
  else
    warn "QMD collection '${QMD_COLLECTION_NAME}' not present"
  fi
fi

if [[ "${PURGE_UV_TOOL}" -eq 1 ]]; then
  DIST_NAME="$(resolve_distribution_name)"
  if command -v uv >/dev/null 2>&1 && uv tool list 2>/dev/null | grep -q "^${DIST_NAME} "; then
    uv tool uninstall "${DIST_NAME}" || warn "Failed to uninstall uv tool ${DIST_NAME}"
    ok "Removed uv tool install (${DIST_NAME})"
  else
    warn "uv tool install for ${DIST_NAME} not found"
  fi
fi

info "Project checkout in ${PROJECT_DIR} was kept intact."
