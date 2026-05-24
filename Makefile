PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
PYTHONPATH := src
RUFF ?= $(if $(wildcard .venv/bin/ruff),.venv/bin/ruff,ruff)
COVERAGE ?= $(if $(wildcard .venv/bin/coverage),.venv/bin/coverage,coverage)
BANDIT ?= $(if $(wildcard .venv/bin/bandit),.venv/bin/bandit,bandit)
BASEDPYRIGHT ?= $(if $(wildcard .venv/bin/basedpyright),.venv/bin/basedpyright,basedpyright)
PIP_AUDIT ?= $(if $(wildcard .venv/bin/pip-audit),.venv/bin/pip-audit,pip-audit)
VULTURE ?= $(if $(wildcard .venv/bin/vulture),.venv/bin/vulture,vulture)
DEPTRY ?= $(if $(wildcard .venv/bin/deptry),.venv/bin/deptry,deptry)
LINT_IMPORTS ?= $(if $(wildcard .venv/bin/lint-imports),.venv/bin/lint-imports,lint-imports)
SEMGREP ?= $(if $(wildcard .venv/bin/semgrep),.venv/bin/semgrep,semgrep)
UV_CACHE_DIR ?= .uv-cache
# Container engine: docker preferred, podman as drop-in replacement.
# Override with `make ... DOCKER=podman` or by setting DOCKER in the environment.
DOCKER ?= $(shell command -v docker 2>/dev/null || command -v podman 2>/dev/null)
SHELLCHECK ?= shellcheck
SHFMT ?= shfmt
RUFF_TARGETS := . $(wildcard src/project_code_intelligence/code_profiles/*.py)

SHELL_FILES := \
	pci-doctor \
	pci-embedding-bench \
	pci-embedding-server \
	pci-fastembed-server \
	pci-index \
	pci-ingest-code \
	pci-llama-embed \
	pci-mcp \
	pci-mcp-smoke \
	docker/llamacpp-cuda/entrypoint.sh \
	docker/llamacpp-rocm/entrypoint.sh

PACKAGED_COMPOSE_ASSETS := \
	bin/pci-embedding-server:pci-embedding-server \
	docker/build-context/LICENSE:docker/build-context/LICENSE \
	docker/build-context/README.md:docker/build-context/README.md \
	docker/build-context/pyproject.toml:docker/build-context/pyproject.toml \
	docker/fastembed/Dockerfile:docker/fastembed/Dockerfile \
	docker/llamacpp-cuda/Dockerfile:docker/llamacpp-cuda/Dockerfile \
	docker/llamacpp-cuda/entrypoint.sh:docker/llamacpp-cuda/entrypoint.sh \
	docker/llamacpp-rocm/Dockerfile:docker/llamacpp-rocm/Dockerfile \
	docker/llamacpp-rocm/entrypoint.sh:docker/llamacpp-rocm/entrypoint.sh \
	docker/pgvector/init-extensions.sql:docker/pgvector/init-extensions.sql \
	scripts/select_llamacpp_rocm_bundle.py:scripts/select_llamacpp_rocm_bundle.py

.PHONY: help check lint format-check format shellcheck shell-format-check shell-format test coverage dead-code dependency-check architecture-check semgrep-check quality-strict typecheck security security-audit deps-audit doctor integration-smoke scan scan-dry-run mcp-smoke embedding-bench amd-rocm-bundle compose-check tool-install tool-uninstall

# Preserve the historical default (`make` runs the full quality gate); `help` is opt-in.
.DEFAULT_GOAL := check

help: ## Show this help and exit
	@awk 'BEGIN {FS = ":[^#]*## "; print "Targets (default: check):"} /^[a-zA-Z][a-zA-Z0-9_-]+:[^=]*## / {printf "  %-22s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

check: format-check shell-format-check lint shellcheck dead-code dependency-check architecture-check semgrep-check coverage typecheck security deps-audit compose-check ## Run the full local quality gate

lint: ## Lint Python with ruff
	$(RUFF) check $(RUFF_TARGETS)

format-check: ## Verify Python is ruff-formatted
	$(RUFF) format --check $(RUFF_TARGETS)

format: ## Auto-format Python with ruff
	$(RUFF) format $(RUFF_TARGETS)

shellcheck: ## Lint shell scripts with shellcheck
	@if command -v $(SHELLCHECK) >/dev/null 2>&1; then \
		$(SHELLCHECK) $(SHELL_FILES); \
	else \
		echo "warning: shellcheck not found; skipping shell lint" >&2; \
	fi

shell-format-check: ## Verify shell scripts are shfmt-formatted
	@if command -v $(SHFMT) >/dev/null 2>&1; then \
		$(SHFMT) -d $(SHELL_FILES); \
	else \
		echo "warning: shfmt not found; skipping shell format check" >&2; \
	fi

shell-format: ## Auto-format shell scripts with shfmt
	@if command -v $(SHFMT) >/dev/null 2>&1; then \
		$(SHFMT) -w $(SHELL_FILES); \
	else \
		echo "warning: shfmt not found; skipping shell formatting" >&2; \
	fi

test: ## Run unit tests
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -v

coverage: ## Run tests with coverage report
	$(COVERAGE) erase
	PYTHONPATH=$(PYTHONPATH) $(COVERAGE) run -m unittest discover -s tests -v
	$(COVERAGE) report -m --sort=cover

dead-code: ## Find unused code with vulture
	$(VULTURE) src tests scripts vulture_whitelist.py --min-confidence 0 --sort-by-size

dependency-check: ## Find unused or undeclared deps with deptry
	$(DEPTRY) src scripts

architecture-check: ## Enforce import boundaries with lint-imports
	$(LINT_IMPORTS) --config pyproject.toml

semgrep-check: ## Run semgrep static analysis with project rules
	$(SEMGREP) scan --config semgrep.yml --error --strict --metrics=off --disable-version-check --quiet src scripts tests

quality-strict: dead-code dependency-check architecture-check semgrep-check coverage ## Run the strict optional quality gates

typecheck: ## Type-check with basedpyright
	$(BASEDPYRIGHT) --warnings

security: ## Run bandit security scan
	$(BANDIT) -c pyproject.toml -r src tests scripts --severity-level all --confidence-level all

security-audit: ## Run bandit including nosec-suppressed findings
	$(BANDIT) -c pyproject.toml -r src tests scripts --severity-level all --confidence-level all --ignore-nosec

deps-audit: ## Audit Python dependencies with pip-audit
	@set -eu; \
	if ! command -v uv >/dev/null 2>&1; then \
		echo "warning: uv not found; skipping dependency audit" >&2; \
	elif [ -x "$(PIP_AUDIT)" ] || command -v $(PIP_AUDIT) >/dev/null 2>&1; then \
		tmpfile=$$(mktemp); \
		trap 'rm -f "$$tmpfile"' EXIT INT TERM; \
		UV_CACHE_DIR="$(UV_CACHE_DIR)" uv export --frozen --no-emit-project --no-emit-workspace --format requirements-txt --quiet > "$$tmpfile"; \
		$(PIP_AUDIT) -r "$$tmpfile" --require-hashes --disable-pip --strict; \
	else \
		echo "warning: pip-audit not found; skipping dependency audit" >&2; \
	fi

doctor: ## Run pci-doctor diagnostics
	./pci-doctor

integration-smoke: ## Bring up pgvector and run integration smoke test
	$(DOCKER) compose up -d --wait --wait-timeout 60 pgvector
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/integration_smoke.py

scan: ## Index this repo into the local DB with pci-index
	./pci-index .

scan-dry-run: ## Preview indexing without writing to the DB
	./pci-index . --dry-run

mcp-smoke: ## Run pci-mcp-smoke end-to-end probes
	./pci-mcp-smoke

embedding-bench: ## Run the embedding endpoint benchmark
	./pci-embedding-bench

amd-rocm-bundle: ## Print the llama.cpp ROCm bundle selection for this GPU
	$(PYTHON) scripts/select_llamacpp_rocm_bundle.py --format env

compose-check: ## Validate docker-compose.yml and bundled copy
	@if ! diff -q docker-compose.yml src/project_code_intelligence/docker-compose.yml >/dev/null 2>&1; then \
		echo "error: docker-compose.yml and src/project_code_intelligence/docker-compose.yml are out of sync" >&2; \
		diff docker-compose.yml src/project_code_intelligence/docker-compose.yml >&2; \
		exit 1; \
	fi
	@for mapping in $(PACKAGED_COMPOSE_ASSETS); do \
		package_path=$${mapping%%:*}; \
		root_path=$${mapping#*:}; \
		bundled_path="src/project_code_intelligence/$$package_path"; \
		if ! diff -q "$$root_path" "$$bundled_path" >/dev/null 2>&1; then \
			echo "error: $$root_path and $$bundled_path are out of sync" >&2; \
			diff "$$root_path" "$$bundled_path" >&2; \
			exit 1; \
		fi; \
	done
	@if [ -n "$(DOCKER)" ] && $(DOCKER) compose version >/dev/null 2>&1; then \
		$(DOCKER) compose config --quiet; \
	else \
		echo "warning: no container engine (docker or podman) with compose support found; skipping compose validation" >&2; \
	fi

tool-install: ## Install or upgrade the pci-* binaries with uv
	uv tool install --python "$$(command -v python3)" . --reinstall
	@bindir=$$(uv tool dir --bin 2>/dev/null) || bindir=""; \
	if [ -z "$$bindir" ]; then exit 0; fi; \
	case ":$$PATH:" in \
	  *":$$bindir:"*) exit 0 ;; \
	esac; \
	printf '\n  %s is not on your PATH (uv tool install shims live there).\n' "$$bindir" >&2; \
	if [ -t 0 ] && [ -t 2 ]; then \
	  printf '  Run `uv tool update-shell` now to add it? [y/N] ' >&2; \
	  read -r reply; \
	  case "$$reply" in \
	    [Yy]|[Yy][Ee][Ss]) uv tool update-shell ;; \
	    *) printf '  Skipped. Run `uv tool update-shell` (or add %s to PATH manually) when ready.\n' "$$bindir" >&2 ;; \
	  esac; \
	else \
	  printf '  Run `uv tool update-shell` (or add it to PATH manually) so the pci-* commands are found.\n' >&2; \
	fi

tool-uninstall: ## Clean local services/cache and uninstall the pci-* binaries
	@bindir=$$(uv tool dir --bin 2>/dev/null) || bindir=""; \
	doctor=$$(command -v pci-doctor 2>/dev/null) || doctor=""; \
	if [ -z "$$doctor" ] && [ -n "$$bindir" ] && [ -x "$$bindir/pci-doctor" ]; then doctor="$$bindir/pci-doctor"; fi; \
	if [ -n "$$doctor" ]; then \
	  "$$doctor" --clean; \
	else \
	  printf 'warning: pci-doctor not found; skipping local service/cache cleanup before uninstall\n' >&2; \
	fi
	uv tool uninstall project-code-intelligence
