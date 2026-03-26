.DEFAULT_GOAL := help

.PHONY: help install purge test check build publish push status

help:
	@echo "Available targets:"
	@echo "  install  - run scripts/install.sh"
	@echo "  purge    - run scripts/purge.sh"
	@echo "  test     - run pytest"
	@echo "  check    - py_compile + pytest"
	@echo "  build    - build package"
	@echo "  publish  - publish package with uv"
	@echo "  status   - git status --short"
	@echo "  push     - push current branch to all remotes"

install:
	@./scripts/install.sh

purge:
	@./scripts/purge.sh

test:
	@uv run pytest -q

check:
	@python3 -m py_compile src/k_ai/*.py src/k_ai/tools/*.py src/k_ai/ui/*.py test/*.py
	@uv run pytest -q

build:
	@uv build

publish: build
	@uv publish

status:
	@git status --short

push:
	@branch="$$(git branch --show-current)"; \
	for remote in $$(git remote); do \
		echo "==> pushing $$branch to $$remote"; \
		git push "$$remote" "$$branch"; \
	done
