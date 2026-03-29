import os
import stat
import subprocess
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSTALL_SCRIPT = PROJECT_ROOT / "scripts" / "install.sh"
PURGE_SCRIPT = PROJECT_ROOT / "scripts" / "purge.sh"


def _base_env(home: Path, extra_path: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    env["XDG_CONFIG_HOME"] = str(home / ".config")
    env["XDG_DATA_HOME"] = str(home / ".local" / "share")
    env["XDG_CACHE_HOME"] = str(home / ".cache")
    env["UV_CACHE_DIR"] = str(home / ".uv-cache")
    if extra_path is not None:
        env["PATH"] = f"{extra_path}:{env['PATH']}"
    return env


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_install_script_configures_custom_runtime_root(tmp_path):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    profile_path = tmp_path / "install-audit.yaml"
    profile = yaml.safe_load((PROJECT_ROOT / "install" / "install.yaml").read_text(encoding="utf-8"))
    profile["interactive"]["enabled"] = False
    profile["verification"]["enabled"] = False
    profile["verification"]["prefer_make_check"] = False
    profile["verification"]["run_doctor"] = False
    profile["qmd"]["install_if_missing"] = False
    profile["qmd"]["collection_name"] = "audit-kai"
    profile["capabilities"]["qmd"] = False
    profile["python_sandbox"]["enabled"] = False
    profile["runtime_store"]["home_dir"] = "~/.k-ai-audit"
    profile["editor"]["offer_micro_install"] = False
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")

    env = _base_env(home)
    result = subprocess.run(
        [str(INSTALL_SCRIPT), "-p", str(profile_path)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    runtime_root = home / ".k-ai-audit"
    config_path = runtime_root / "config.yaml"
    assert config_path.exists()

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert data["config"]["persist_path"] == str(config_path)
    assert data["memory"]["path"] == str(runtime_root / "MEMORY.md")
    assert data["sessions"]["directory"] == str(runtime_root / "sessions")
    assert data["tools"]["qmd"]["session_collection"] == "audit-kai"
    assert data["tools"]["mcp"]["enabled"] is True
    assert data["mcp"]["servers"]["filesystem"]["command"] == "mcp-server-filesystem"
    assert data["runtime_git"]["enabled"] is True
    assert (runtime_root / ".git").exists()
    log = subprocess.run(
        ["git", "-C", str(runtime_root), "log", "--oneline", "-1"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert log.returncode == 0
    assert "init: k-ai runtime store" in log.stdout


def test_purge_script_aborts_cleanly_without_yes_in_noninteractive_mode(tmp_path):
    home = tmp_path / "home"
    runtime_root = home / ".k-ai"
    runtime_root.mkdir(parents=True)
    (runtime_root / "config.yaml").write_text("config: {}\n", encoding="utf-8")

    env = _base_env(home)
    result = subprocess.run(
        [str(PURGE_SCRIPT)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert runtime_root.exists()
    assert "Non-interactive purge aborted" in result.stdout


def test_purge_script_uses_custom_runtime_dir_and_dynamic_uv_package_name(tmp_path):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    runtime_root = home / ".runtime-store"
    runtime_root.mkdir(parents=True)
    (runtime_root / "config.yaml").write_text(
        "tools:\n  qmd:\n    session_collection: custom-qmd\n",
        encoding="utf-8",
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls_file = tmp_path / "uv-calls.txt"
    _write_executable(
        bin_dir / "uv",
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "{calls_file}"
if [[ "$1 $2" == "tool list" ]]; then
  printf 'kpihx-ai v0.2.0\\n- k-ai\\n'
elif [[ "$1 $2" == "tool uninstall" ]]; then
  exit 0
else
  exit 1
fi
""",
    )

    env = _base_env(home, extra_path=bin_dir)
    result = subprocess.run(
        [str(PURGE_SCRIPT), "--yes", "--runtime-dir", str(runtime_root), "--uv-tool"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    assert not runtime_root.exists()
    calls = calls_file.read_text(encoding="utf-8")
    assert "tool list" in calls
    assert "tool uninstall kpihx-ai" in calls


def test_purge_script_uses_collection_name_from_runtime_config(tmp_path):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    runtime_root = home / ".runtime-store"
    runtime_root.mkdir(parents=True)
    (runtime_root / "config.yaml").write_text(
        "tools:\n  qmd:\n    session_collection: bespoke-qmd\n",
        encoding="utf-8",
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls_file = tmp_path / "qmd-calls.txt"
    _write_executable(
        bin_dir / "qmd",
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "{calls_file}"
if [[ "$1 $2 $3" == "collection show bespoke-qmd" ]]; then
  exit 0
fi
if [[ "$1 $2 $3" == "collection remove bespoke-qmd" ]]; then
  exit 0
fi
exit 1
""",
    )

    env = _base_env(home, extra_path=bin_dir)
    result = subprocess.run(
        [str(PURGE_SCRIPT), "--yes", "--runtime-dir", str(runtime_root), "--qmd"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    calls = calls_file.read_text(encoding="utf-8")
    assert "collection show bespoke-qmd" in calls
    assert "collection remove bespoke-qmd" in calls
