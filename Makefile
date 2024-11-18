SHELL := /bin/bash

PYTHON_VERSION ?= 3.10
PYTHON ?= python$(PYTHON_VERSION)
PIP ?= $(PYTHON) -m pip
PIPENV ?= $(PYTHON) -m pipenv
PIPENV_PYTHON = $(PIPENV) run python
PIPENV_PIP = $(PIPENV_PYTHON) -m pip
PWD = $(shell pwd)

ifeq ($(origin H2O_LLM_STUDIO_WORKDIR), environment)
    WORKDIR := $(H2O_LLM_STUDIO_WORKDIR)
else
    WORKDIR := $(shell pwd)
endif

ifeq ($(LOG_LEVEL), $(filter $(LOG_LEVEL), debug trace))
    PW_DEBUG = DEBUG=pw:api
else
    PW_DEBUG =
endif

.PHONY: pipenv
pipenv:
	$(PIP) install pip==24.2
	$(PIP) install pipenv==2024.0.1

.PHONY: setup
setup: pipenv
	$(PIPENV) install --verbose --python $(PYTHON_VERSION)
	-$(PIPENV_PIP) install flash-attn==2.6.1 --no-build-isolation --upgrade --no-cache-dir

.PHONY: setup-dev
setup-dev: pipenv
	$(PIPENV) install --verbose --dev --python $(PYTHON_VERSION)
	-$(PIPENV_PIP) install flash-attn==2.6.1 --no-build-isolation --upgrade --no-cache-dir
	$(PIPENV) run playwright install


clean-env:
	$(PIPENV) --rm

.PHONY: style
style: reports pipenv
	@echo -n > reports/flake8_errors.log
	@echo -n > reports/mypy_errors.log
	@echo -n > reports/mypy.log
	@echo

	-$(PIPENV) run flake8 | tee -a reports/flake8_errors.log
	@if [ -s reports/flake8_errors.log ]; then exit 1; fi

	-$(PIPENV) run mypy . --check-untyped-defs | tee -a reports/mypy.log
	@if ! grep -Eq "Success: no issues found in [0-9]+ source files" reports/mypy.log ; then exit 1; fi

.PHONY: format
format: pipenv
	$(PIPENV) run isort .
	$(PIPENV) run black .

.PHONY: isort
isort: pipenv
	$(PIPENV) run isort .

.PHONY: black
black: pipenv
	$(PIPENV) run black .

.PHONY: test
test: reports
	@bash -c 'set -o pipefail; export PYTHONPATH=$(PWD); \
	$(PIPENV) run pytest -v --junitxml=reports/junit.xml \
	--import-mode importlib \
	--html=./reports/pytest.html \
	--cov=llm_studio \
	--cov-report term \
	--cov-report html:./reports/coverage.html \
    -o log_cli=true -o log_level=INFO -o log_file=reports/tests.log \
    tests/* 2>&1 | tee reports/tests.log'

.PHONY: shell
shell:
	$(PIPENV) shell
