# k-ai Installation Profiles

This directory contains the installation-time defaults used by [`scripts/install.sh`](../scripts/install.sh).

Quick navigation:

- main project README: [`../README.md`](../README.md)
- local docs hub: [`../docs/README.md`](../docs/README.md)
- live docs site: [kpihx.github.io/k-ai-docs](https://kpihx.github.io/k-ai-docs/)

Main file:

- [`install.yaml`](install.yaml)

Use cases:

- keep the default interactive experience:
  `./scripts/install.sh`
- explicitly use the default profile:
  `./scripts/install.sh -p defaults`
- use a custom profile:
  `./scripts/install.sh --path /path/to/my-install.yaml`

What `install.yaml` controls:

- uv bootstrap policy and fallback behavior
- installer interactivity
- runtime-store git tracking defaults
- initial capability defaults (`exa`, `python`, `shell`, `qmd`)
- editor choice defaults
- python sandbox path
- default sandbox packages proposed one by one
- whether extra packages can be added interactively
- QMD setup defaults
- final verification behavior

Design rule:

- `install/install.yaml` controls installation
- `~/.k-ai/config.yaml` controls runtime

The installer persists the final editor choice and selected default sandbox
packages into the runtime config so the installed environment remains coherent
after first boot.

Runtime git behavior:

- the installer copies [`install/.gitignore.runtime`](.gitignore.runtime) into `~/.k-ai/.gitignore`
- it initializes a local git repo in `~/.k-ai/` when runtime git tracking is enabled
- it runs the first `git add .` and initial commit after install
- later, interactive chat exits can auto-commit `config.yaml`, `MEMORY.json`, and `sessions/*`

Bootstrap behavior:

- preferred path: use `uv` when it is already available
- if `uv` is missing, the installer can propose installing it automatically
- if the user refuses `uv`, the installer falls back to an isolated virtualenv
  dedicated to `k-ai`
- this fallback does not install packages into the general system Python

Interactive philosophy:

- each meaningful prompt spells out the available cases
- capability families are chosen explicitly during install
- package selection is explicit, one proposed package at a time
- extra sandbox packages are collected in a loop until the user sends an empty line
