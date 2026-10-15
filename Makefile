SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
INSTALL_STAMP := $(VENV)/.editable-installed
PROJECT_META := $(wildcard pyproject.toml setup.py setup.cfg requirements*.txt)

# Extra CLI arguments can be passed to almost every execution target:
#   make simulate SCHEDULER=carbon ARGS="--max-delay-hours 6 --limit 5000"
ARGS ?=

# -----------------------------------------------------------------------------
# Common experiment parameters
# -----------------------------------------------------------------------------
SCHEDULER ?= fcfs
WORKLOAD ?= data/processed/pm100_clean.parquet
PREDICTIONS ?= data/job_predictions/test_predictions.parquet
RIDGE_PREDICTIONS_DIR ?= data/job_predictions/ridge_baseline
CARBON_CACHE ?= data/carbon_intensity/actual/actual.json
FORECAST_ARCHIVE ?= data/carbon_intensity/snapshots/test_boosted_ridge.json
CONTENTION_WORKLOAD ?= data/job_table.parquet
POWER_CAP_MW ?= 0.68
ALL_SCHEDULERS := fcfs easy power-cap carbon carbon-scaled-delay carbon-power-cap carbon-forecast carbon-forecast-scaled-delay replay

CARBON_START ?= 2020-04-30T00:00:00Z
CARBON_END ?= 2020-11-02T00:00:00Z

.PHONY: help venv install reinstall print-config \
        test test-one \
        carbon-download carbon-history carbon-history-offline \
        carbon-exogenous carbon-exogenous-offline carbon-evaluate carbon-select \
        carbon-snapshots carbon-walkforward carbon-figures carbon-boosted \
        carbon-scheduling-impact \
        job-search job-train job-train-ridge job-scheduling-impact job-compare \
        simulate simulate-all simulate-predicted simulate-forecast baselines baselines-full \
        tradeoff tradeoff-predicted tradeoff-contention \
        ablation milp improved-scheduler qos-carbon qos-carbon-report robustness robustness-report \
        clean clean-pycache clean-venv

help: ## Mostra i target disponibili
	@printf "Uso: make <target> [ARGS='...'] [VAR=value]\n\n"
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_.-]+:.*## / {printf "  %-26s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf "\nEsempio:\n  make simulate SCHEDULER=carbon ARGS='--max-delay-hours 6 --limit 5000'\n"

print-config: ## Mostra le variabili principali del Makefile
	@printf "PYTHON=%s\n" "$(PYTHON)"
	@printf "VENV=%s\n" "$(VENV)"
	@printf "SCHEDULER=%s\n" "$(SCHEDULER)"
	@printf "WORKLOAD=%s\n" "$(WORKLOAD)"
	@printf "PREDICTIONS=%s\n" "$(PREDICTIONS)"
	@printf "CARBON_CACHE=%s\n" "$(CARBON_CACHE)"
	@printf "FORECAST_ARCHIVE=%s\n" "$(FORECAST_ARCHIVE)"
	@printf "POWER_CAP_MW=%s\n" "$(POWER_CAP_MW)"
	@printf "ALL_SCHEDULERS=%s\n" "$(ALL_SCHEDULERS)"
	@printf "ARGS=%s\n" "$(ARGS)"

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
$(VENV_PYTHON):
	$(PYTHON) -m venv $(VENV)

venv: $(VENV_PYTHON) ## Crea l'ambiente virtuale .venv

$(INSTALL_STAMP): $(VENV_PYTHON) $(PROJECT_META)
	$(VENV_PIP) install -e .
	@touch $@

install: $(INSTALL_STAMP) ## Installa il progetto in editable mode nella .venv

reinstall: ## Forza la reinstallazione editable
	@rm -f $(INSTALL_STAMP)
	$(MAKE) install

.PHONY: thesis

thesis:
	cd thesis && lualatex main.tex
	cd thesis && $(HOME)/bin/biber main
	cd thesis && lualatex main.tex
	cd thesis && lualatex main.tex

# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------
test: install ## Esegue tutti i check unittest (tests/check_*.py)
	$(VENV_PYTHON) -m unittest discover -s tests -p "check_*.py"

test-one: install ## Esegue un singolo test: make test-one TEST=check_hpc_sim_simulator.py
	@test -n "$(TEST)" || { echo "Uso: make test-one TEST=check_hpc_sim_simulator.py"; exit 2; }
	$(VENV_PYTHON) tests/$(TEST)

# -----------------------------------------------------------------------------
# Carbon-intensity data and forecasting
# -----------------------------------------------------------------------------
carbon-download: install ## Scarica la serie Electricity Maps 2020 (START/END sovrascrivibili)
	$(VENV_PYTHON) scripts/download_carbon_intensity.py \
		--start $(CARBON_START) \
		--end $(CARBON_END) \
		$(ARGS)

carbon-history: install ## Costruisce/aggiorna lo storico carbon intensity
	$(VENV_PYTHON) scripts/prepare_carbon_history.py $(ARGS)

carbon-history-offline: install ## Ricostruisce lo storico usando solo le cache locali
	$(VENV_PYTHON) scripts/prepare_carbon_history.py --offline $(ARGS)

carbon-exogenous: install ## Scarica e prepara i segnali esogeni della rete
	$(VENV_PYTHON) -m carbon_intensity.exogenous $(ARGS)

carbon-exogenous-offline: install ## Ricostruisce i segnali esogeni dalla cache locale
	$(VENV_PYTHON) -m carbon_intensity.exogenous --offline $(ARGS)

carbon-evaluate: install ## Valuta i modelli di carbon-intensity forecasting
	$(VENV_PYTHON) -m carbon_intensity.evaluate $(ARGS)

carbon-select: install ## Esegue la selezione dei modelli/history per il forecast
	$(VENV_PYTHON) -m carbon_intensity.selection $(ARGS)

carbon-snapshots: install ## Genera gli snapshot storici di forecast
	$(VENV_PYTHON) -m carbon_intensity.snapshots $(ARGS)

carbon-walkforward: install ## Esegue la valutazione walk-forward
	$(VENV_PYTHON) -m carbon_intensity.walkforward $(ARGS)

carbon-figures: install ## Genera le figure del modulo carbon_intensity
	$(VENV_PYTHON) -m carbon_intensity.figures $(ARGS)

carbon-boosted: install ## Esegue il modello boosted residual
	$(VENV_PYTHON) -m carbon_intensity.boosted $(ARGS)

carbon-scheduling-impact: install ## Valuta l'impatto del forecast sullo scheduling
	$(VENV_PYTHON) -m carbon_intensity.scheduling_impact $(ARGS)

# -----------------------------------------------------------------------------
# Job prediction
# -----------------------------------------------------------------------------
job-search: install ## Esegue la ricerca/validazione dei modelli job
	$(VENV_PYTHON) scripts/experiment_job_models.py $(ARGS)

job-train: install ## Addestra il modello gradient selezionato
	$(VENV_PYTHON) scripts/train_job_models.py --model gradient $(ARGS)

job-train-ridge: install ## Addestra il baseline ridge in una directory separata
	$(VENV_PYTHON) scripts/train_job_models.py \
		--model ridge \
		--output-dir $(RIDGE_PREDICTIONS_DIR) \
		$(ARGS)

job-scheduling-impact: install ## Valuta l'impatto delle predizioni job sullo scheduling
	$(VENV_PYTHON) -m job_prediction.scheduling_impact $(ARGS)

job-compare: install ## Confronta gli artifact ridge e gradient
	$(VENV_PYTHON) scripts/compare_job_prediction_impact.py $(ARGS)

# -----------------------------------------------------------------------------
# Simulator and scheduler experiments
# -----------------------------------------------------------------------------
simulate: install ## Esegue una simulazione; es. SCHEDULER=easy o carbon
	$(VENV_PYTHON) scripts/run_simulation.py --scheduler $(SCHEDULER) $(ARGS)

simulate-all: install ## Esegue tutti gli scheduler supportati dalla CLI
	@set -euo pipefail; \
	common=(--workload "$(WORKLOAD)" --carbon-cache "$(CARBON_CACHE)"); \
	for scheduler in $(ALL_SCHEDULERS); do \
		extra=(); \
		case "$$scheduler" in \
			power-cap|carbon-power-cap) extra+=(--power-cap-mw "$(POWER_CAP_MW)") ;; \
		esac; \
		case "$$scheduler" in \
			carbon-forecast|carbon-forecast-scaled-delay) \
				extra+=(--forecast-archive "$(FORECAST_ARCHIVE)") ;; \
		esac; \
		printf "\n=== scheduler: %s ===\n" "$$scheduler"; \
		$(VENV_PYTHON) scripts/run_simulation.py \
			--scheduler "$$scheduler" \
			"$${common[@]}" \
			"$${extra[@]}" \
			$(ARGS); \
	done

simulate-predicted: install ## Simula usando il cohort e le predizioni job
	$(VENV_PYTHON) scripts/run_simulation.py \
		--scheduler $(SCHEDULER) \
		--workload $(WORKLOAD) \
		--job-predictions $(PREDICTIONS) \
		$(ARGS)

simulate-forecast: install ## Simula uno scheduler forecast-based con archive e actual cache
	$(VENV_PYTHON) scripts/run_simulation.py \
		--scheduler carbon-forecast \
		--carbon-cache $(CARBON_CACHE) \
		--forecast-archive $(FORECAST_ARCHIVE) \
		$(ARGS)

baselines: install ## Confronta le baseline con i default dello script
	$(VENV_PYTHON) scripts/compare_baselines.py $(ARGS)

baselines-full: install ## Matrice actual/predicted + oracle/forecast (artifact standard)
	$(VENV_PYTHON) scripts/compare_baselines.py \
		--no-replay \
		--workload $(WORKLOAD) \
		--job-predictions gradient=$(PREDICTIONS) \
		--carbon-cache $(CARBON_CACHE) \
		--forecast-archive $(FORECAST_ARCHIVE) \
		--runtime-estimate scheduling \
		$(ARGS)

tradeoff: install ## Sweep carbon/QoS con i default dello script
	$(VENV_PYTHON) scripts/carbon_tradeoff.py $(ARGS)

tradeoff-predicted: install ## Sweep carbon/QoS usando le predizioni job
	$(VENV_PYTHON) scripts/carbon_tradeoff.py \
		--workload $(WORKLOAD) \
		--job-predictions $(PREDICTIONS) \
		$(ARGS)

tradeoff-contention: install ## Sweep includendo i terminal job come contention-only
	$(VENV_PYTHON) scripts/carbon_tradeoff.py \
		--contention-workload $(CONTENTION_WORKLOAD) \
		$(ARGS)

# -----------------------------------------------------------------------------
# Cross-cutting experiments
# -----------------------------------------------------------------------------
ablation: install ## Esegue l'ablation job-model vs carbon-forecast
	$(VENV_PYTHON) -m experiments.ablation $(ARGS)

milp: install ## Esegue il benchmark MILP su istanze ridotte/configurabili
	$(VENV_PYTHON) -m experiments.milp $(ARGS)

PHASE ?= smoke
improved-scheduler: install ## Scheduler delay-aware: PHASE=prepare|smoke|validation|test|milp|report
	$(VENV_PYTHON) -m experiments.improved_scheduler $(PHASE) $(ARGS)

qos-carbon: install ## Consolida QoS/carbon: cache, quattro scenari, controlli errori, MILP, CSV e figure
	$(VENV_PYTHON) -m experiments.qos_carbon run $(ARGS)

qos-carbon-report: install ## Rigenera report e figure QoS/carbon dai soli CSV salvati
	$(VENV_PYTHON) -m experiments.qos_carbon report $(ARGS)

robustness: install ## Completa sensitivity, ablation, fairness e output per la tesi
	$(VENV_PYTHON) -m experiments.robustness run $(ARGS)

robustness-report: install ## Rigenera figure e tabelle di robustezza dai soli CSV
	$(VENV_PYTHON) -m experiments.robustness report $(ARGS)

# -----------------------------------------------------------------------------
# Cleanup (intentionally preserves datasets and experiment results)
# -----------------------------------------------------------------------------
clean-pycache: ## Rimuove __pycache__, file .pyc e .pyo
	@find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	@find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

clean: clean-pycache ## Rimuove cache/build Python; conserva data/ e risultati
	@rm -rf build dist .pytest_cache .mypy_cache .ruff_cache
	@find . -maxdepth 2 -type d -name '*.egg-info' -prune -exec rm -rf {} +

clean-venv: ## Rimuove l'ambiente virtuale
	rm -rf $(VENV)
