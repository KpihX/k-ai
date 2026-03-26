#!/usr/bin/env bash

set -euo pipefail

RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
CYAN='\033[1;36m'
RESET='\033[0m'

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
K_AI_DIR="${HOME}/.k-ai"

info() { echo -e "${CYAN}[k-ai]${RESET} $*"; }
ok() { echo -e "${GREEN}  OK${RESET} $*"; }
warn() { echo -e "${YELLOW}  WARN${RESET} $*"; }
fail() { echo -e "${RED}  ERR${RESET} $*"; }

PURGE_QMD=0
PURGE_UV_TOOL=0

for arg in "$@"; do
  case "$arg" in
    --qmd) PURGE_QMD=1 ;;
    --uv-tool) PURGE_UV_TOOL=1 ;;
    *)
      fail "Unknown option: $arg"
      echo "Usage: ./scripts/purge.sh [--qmd] [--uv-tool]"
      exit 1
      ;;
  esac
done

echo -e "${RED}This will remove runtime data from ${K_AI_DIR}.${RESET}"
echo "Included by default:"
echo "  - sessions"
echo "  - memory"
echo "  - sandbox"
echo "  - persisted config"
echo "  - runtime git repo/hooks"
if [[ "${PURGE_QMD}" -eq 1 ]]; then
  echo "  - QMD collection 'k-ai' if available"
fi
if [[ "${PURGE_UV_TOOL}" -eq 1 ]]; then
  echo "  - editable uv tool install for k-ai"
fi
printf "Continue? [y/N] "
read -r answer
if [[ ! "${answer}" =~ ^[Yy]$ ]]; then
  warn "Aborted."
  exit 0
fi

if [[ -d "${K_AI_DIR}" ]]; then
  rm -rf "${K_AI_DIR}"
  ok "Removed ${K_AI_DIR}"
else
  warn "${K_AI_DIR} not present"
fi

if [[ "${PURGE_QMD}" -eq 1 ]] && command -v qmd >/dev/null 2>&1; then
  if qmd collection show k-ai >/dev/null 2>&1; then
    qmd collection remove k-ai || warn "Failed to remove QMD collection 'k-ai'"
    ok "Removed QMD collection 'k-ai'"
  else
    warn "QMD collection 'k-ai' not present"
  fi
fi

if [[ "${PURGE_UV_TOOL}" -eq 1 ]]; then
  if uv tool list 2>/dev/null | grep -q '^k-ai '; then
    uv tool uninstall k-ai || warn "Failed to uninstall uv tool k-ai"
    ok "Removed uv tool install"
  else
    warn "uv tool install for k-ai not found"
  fi
fi

info "Project checkout in ${PROJECT_DIR} was kept intact."
