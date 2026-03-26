.DEFAULT_GOAL := help

.PHONY: help install purge test check build publish push push-docs publish-docs release status

ZSH_LOGIN = zsh -lc

help:
	@echo "Available targets:"
	@echo "  install  - run scripts/install.sh"
	@echo "  purge    - run scripts/purge.sh"
	@echo "  test     - run pytest"
	@echo "  check    - py_compile + pytest"
	@echo "  build    - build package"
	@echo "  publish  - publish package with uv"
	@echo "  push-docs - push the standalone docs repo to all remotes"
	@echo "  publish-docs - publish the standalone docs repo"
	@echo "  release  - check + build + publish + push + push-docs"
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
	@rm -rf dist
	@uv build

publish: build
	@$(ZSH_LOGIN) 'if ! env | grep -q "^UV_PUBLISH_TOKEN="; then \
		echo "UV_PUBLISH_TOKEN is not available in the login shell. Check bw-env / shell bootstrap."; \
		exit 1; \
	fi; \
	uv publish --check-url https://pypi.org/simple'

push-docs:
	@$(MAKE) -C docs push

publish-docs:
	@$(MAKE) -C docs publish

status:
	@git status --short

push:
	@branch="$$(git branch --show-current)"; \
	for remote in $$(git remote); do \
		echo "==> pushing $$branch to $$remote"; \
		git push "$$remote" "$$branch"; \
	done

release: check build publish push push-docs
