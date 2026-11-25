UV := uv

HF_CACHE := /goinfre/$(USER)/hf_cache
UV_CACHE_DIR := /goinfre/$(USER)/uv_cache

files := --functions_definition data/input/functions_definition.json \
         --input data/input/function_calling_tests.json \
         --output data/output/function_calls.json

.PHONY: install run debug clean lint lint-strict

install:
	@mkdir -p $(UV_CACHE_DIR)
	@UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) sync

run:
	@mkdir -p $(HF_CACHE)
	@HF_HOME=$(HF_CACHE) \
	HUGGINGFACE_HUB_CACHE=$(HF_CACHE) \
	TRANSFORMERS_CACHE=$(HF_CACHE) \
	$(UV) run python -m src $(files)

debug:
	@mkdir -p $(HF_CACHE)
	@HF_HOME=$(HF_CACHE) \
	HUGGINGFACE_HUB_CACHE=$(HF_CACHE) \
	TRANSFORMERS_CACHE=$(HF_CACHE) \
	$(UV) run python -m pdb src/__main__.py $(files)

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -rf .mypy_cache
	@rm -rf .pytest_cache
	@rm -rf *.egg-info
	@rm -rf src/*.egg-info

lint:
	@echo "Running flake8..."
	@$(UV) run flake8 src

	@echo "Running mypy..."
	@$(UV) run mypy src \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs
