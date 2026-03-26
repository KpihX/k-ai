# k-ai Installation Profiles

This directory contains the installation-time defaults used by [install.sh](/home/kpihx/Work/AI/k_ai/scripts/install.sh).

Main file:

- [install.yaml](/home/kpihx/Work/AI/k_ai/install/install.yaml)

Use cases:

- keep the default interactive experience:
  `./scripts/install.sh`
- explicitly use the default profile:
  `./scripts/install.sh -p defaults`
- use a custom profile:
  `./scripts/install.sh --path /path/to/my-install.yaml`

What `install.yaml` controls:

- installer interactivity
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
