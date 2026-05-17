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

.PHONY: check lint format-check format shellcheck shell-format-check shell-format test coverage dead-code dependency-check architecture-check semgrep-check quality-strict typecheck security security-audit deps-audit doctor integration-smoke scan scan-dry-run mcp-smoke embedding-bench amd-rocm-bundle compose-check compose-up compose-cpu compose-npu compose-amdgpu compose-nvidia compose-down tool-install

check: format-check shell-format-check lint shellcheck test dead-code dependency-check architecture-check semgrep-check coverage typecheck security deps-audit compose-check

lint:
	$(RUFF) check $(RUFF_TARGETS)

format-check:
	$(RUFF) format --check $(RUFF_TARGETS)

format:
	$(RUFF) format $(RUFF_TARGETS)

shellcheck:
	@if command -v $(SHELLCHECK) >/dev/null 2>&1; then \
		$(SHELLCHECK) $(SHELL_FILES); \
	else \
		echo "warning: shellcheck not found; skipping shell lint" >&2; \
	fi

shell-format-check:
	@if command -v $(SHFMT) >/dev/null 2>&1; then \
		$(SHFMT) -d $(SHELL_FILES); \
	else \
		echo "warning: shfmt not found; skipping shell format check" >&2; \
	fi

shell-format:
	@if command -v $(SHFMT) >/dev/null 2>&1; then \
		$(SHFMT) -w $(SHELL_FILES); \
	else \
		echo "warning: shfmt not found; skipping shell formatting" >&2; \
	fi

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -v

coverage:
	$(COVERAGE) erase
	PYTHONPATH=$(PYTHONPATH) $(COVERAGE) run -m unittest discover -s tests -v
	$(COVERAGE) report -m --sort=cover

dead-code:
	$(VULTURE) src tests scripts vulture_whitelist.py --min-confidence 0 --sort-by-size

dependency-check:
	$(DEPTRY) src scripts

architecture-check:
	$(LINT_IMPORTS) --config pyproject.toml

semgrep-check:
	$(SEMGREP) scan --config semgrep.yml --error --strict --metrics=off --disable-version-check --quiet src scripts tests

quality-strict: dead-code dependency-check architecture-check semgrep-check coverage

typecheck:
	$(BASEDPYRIGHT) --warnings

security:
	$(BANDIT) -c pyproject.toml -r src tests scripts --severity-level all --confidence-level all

security-audit:
	$(BANDIT) -c pyproject.toml -r src tests scripts --severity-level all --confidence-level all --ignore-nosec

deps-audit:
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

doctor:
	./pci-doctor

integration-smoke: compose-up
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/integration_smoke.py

scan:
	./pci-index .

scan-dry-run:
	./pci-index . --dry-run

mcp-smoke:
	./pci-mcp-smoke

embedding-bench:
	./pci-embedding-bench

amd-rocm-bundle:
	$(PYTHON) scripts/select_llamacpp_rocm_bundle.py --format env

compose-check:
	@if ! diff -q docker-compose.yml src/project_code_intelligence/docker-compose.yml >/dev/null 2>&1; then \
		echo "error: docker-compose.yml and src/project_code_intelligence/docker-compose.yml are out of sync" >&2; \
		diff docker-compose.yml src/project_code_intelligence/docker-compose.yml >&2; \
		exit 1; \
	fi
	@if [ -n "$(DOCKER)" ] && $(DOCKER) compose version >/dev/null 2>&1; then \
		$(DOCKER) compose config --quiet; \
	else \
		echo "warning: no container engine (docker or podman) with compose support found; skipping compose validation" >&2; \
	fi

compose-up:
	$(DOCKER) compose up -d --wait --wait-timeout 60 pgvector

compose-cpu:
	$(DOCKER) compose --profile cpu up -d --build

compose-npu:
	$(DOCKER) compose --profile npu up -d

compose-amdgpu:
	$(DOCKER) compose --profile amdgpu up -d --build

compose-nvidia:
	$(DOCKER) compose --profile nvidia up -d --build

compose-down:
	$(DOCKER) compose down

tool-install:
	uv tool install . --reinstall
