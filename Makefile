PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
PYTHONPATH := src
RUFF ?= $(if $(wildcard .venv/bin/ruff),.venv/bin/ruff,ruff)
COVERAGE ?= $(if $(wildcard .venv/bin/coverage),.venv/bin/coverage,coverage)
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

.PHONY: check lint format-check format shellcheck shell-format-check shell-format test coverage typecheck security security-audit doctor integration-smoke scan scan-dry-run mcp-smoke embedding-bench amd-rocm-bundle compose-check compose-up compose-cpu compose-npu compose-amdgpu compose-nvidia compose-down tool-install

check: format-check shell-format-check lint shellcheck test typecheck security compose-check

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
	PYTHONPATH=$(PYTHONPATH) $(COVERAGE) run -m unittest discover -s tests -v
	$(COVERAGE) report -m

typecheck:
	basedpyright --warnings

security:
	bandit -r src tests scripts --severity-level all --confidence-level all

security-audit:
	bandit -r src tests scripts --severity-level all --confidence-level all --ignore-nosec

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
	@if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then \
		docker compose config --quiet; \
	else \
		echo "warning: docker compose not found; skipping compose validation" >&2; \
	fi

compose-up:
	docker compose up -d pgvector

compose-cpu:
	docker compose --profile cpu up -d --build

compose-npu:
	docker compose --profile npu up -d

compose-amdgpu:
	docker compose --profile amdgpu up -d --build

compose-nvidia:
	docker compose --profile nvidia up -d --build

compose-down:
	docker compose down

tool-install:
	uv tool install . --reinstall
