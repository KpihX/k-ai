# Makefile for k-ai

.PHONY: all push

# ==============================================================================
# GIT COMMANDS
# ==============================================================================

# Push to all remotes defined in .git/config
push:
	@git remote | xargs -L1 git push

# ==============================================================================
# DEVELOPMENT
# ==============================================================================

# Install in editable mode for development
dev:
	uv tool install --editable .

# ==============================================================================
# BUILD & PUBLISH
# ==============================================================================

build:
	uv build

publish: build
	uv publish

.DEFAULT_GOAL := all
all:
	@echo "Available targets: push, dev, build, publish"
