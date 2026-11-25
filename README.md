*This project has been created as part of the 42 curriculum by vhedo-ga.*

# call me maybe — Introduction to function calling in LLMs

## Description

This project is a small, local (offline) prototype of **LLM function calling**.

Given:
- A list of available “tools” (functions) with their names, descriptions, and parameter types (see `data/input/functions_definition.json`), and
- A list of natural-language prompts (see `data/input/function_calling_tests.json`),

the program generates, for each prompt, a **JSON object** describing the function call to execute:

```json
{"name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}
```

The core idea is to guide the model to output a structured JSON object by using a **constrained decoding** approach.

## Instructions

### Requirements

- Python >= 3.10
- `uv` (used for dependency management in the provided Makefile)

Note: the first run will download the Hugging Face model weights/tokenizer (cached under `/goinfre/<user>/hf_cache` by `make run`).

### Install

```bash
make install
```

### Run

```bash
make run
```

This runs the CLI entrypoint (`python -m src`) with the default files:

- Functions definition: `data/input/functions_definition.json`
- Prompts: `data/input/function_calling_tests.json`
- Output: `data/output/function_calls.json`

### Lint / Type-check

```bash
make lint
```

## Example usage

### 1) Run with the default dataset

```bash
make run
```

Under the hood, this is equivalent to running:

```bash
uv run python -m src \
	--functions_definition data/input/functions_definition.json \
	--input data/input/function_calling_tests.json \
	--output data/output/function_calls.json
```

Then inspect the generated results:

```bash
cat data/output/function_calls.json
```

Expected format (example of one item in the output array):

```json
[
  {
    "prompt": "Add 2 and 3",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2,
      "b": 3
    }
  }
]
```

### 2) Run with custom input files

```bash
uv run python -m src \
	--functions_definition path/to/functions_definition.json \
	--input path/to/prompts.json \
	--output path/to/function_calls.json
```

Tip: if you want to reproduce the same Hugging Face cache behavior as `make run`, you can also set:

```bash
HF_HOME=/goinfre/$USER/hf_cache \
HUGGINGFACE_HUB_CACHE=/goinfre/$USER/hf_cache \
TRANSFORMERS_CACHE=/goinfre/$USER/hf_cache \
	uv run python -m src \
		--functions_definition path/to/functions_definition.json \
		--input path/to/prompts.json \
		--output path/to/function_calls.json
```

## Algorithm explanation (constrained decoding)

### High-level approach

The generation loop in `src/solver.py` uses **greedy decoding** (selecting the token with the highest logit via `argmax`) and applies a **token-level constraint** at the beginning of generation.

For each prompt:

1. Build a “system-like” instruction that asks the model to reply **only** with a JSON object.
2. Serialize the available tool schemas into a `<tools> ... </tools>` block.
3. Encode the full prompt with the model tokenizer.
4. Autoregressively generate tokens:
	 - Get next-token logits from the model.
	 - Optionally mask logits to enforce a specific token.
	 - Pick the next token with `argmax`.
	 - Update the decoding “state” based on the decoded text so far.

### Constraint mechanism

The constraint is implemented by masking logits:

- When the decoder is in a *restricted* state, only a single token is allowed.
- All other token logits are set to $-\infty$.
- `argmax` then necessarily selects the permitted token.

In the current implementation, the first generated token is forced to be `{` (opening JSON brace). After that, decoding switches to a “free” mode where the model can generate any token.

### Stop condition

While in free generation mode, the code checks the decoded text and stops when:

- The number of `{` matches the number of `}` and is greater than 0 (heuristic for “JSON object is closed”), OR
- A maximum length threshold is reached (safety guard).

Finally, the generated text is post-processed by extracting the substring between the first `{` and the last `}` and parsing it with `json.loads`.

## Design decisions

- **Local model inference**: implemented via the in-repo package `llm_sdk` using Hugging Face `transformers` for portability and “no external API” operation.
- **Small model choice**: `llm_sdk` defaults to `Qwen/Qwen3-0.6B` to keep memory usage and latency reasonable for a school environment.
- **Greedy decoding**: chosen for determinism and simplicity (repeatable outputs, easy debugging).
- **Minimal constraints**: the decoder forces `{` as the first token to reduce non-JSON preambles; the rest relies on prompting + a stop heuristic.
- **Input validation**: function definitions and prompt inputs are validated with `pydantic` (`extra="forbid"`) to reject unexpected fields early.

## Performance analysis

### Accuracy

- Accuracy depends mostly on how well the model follows the instruction to emit a JSON object.
- Forcing the initial `{` reduces failure modes where the model starts with natural language instead of JSON.
- The final JSON parsing step is strict; any malformed JSON will trigger the fallback output.

### Speed

- Each generated token requires one forward pass for next-token logits.
- Greedy decoding is faster than sampling strategies and avoids additional computation.
- The first run can be slower due to model downloads; subsequent runs reuse the cache.

### Reliability

Known failure modes (and current mitigations):

- **Invalid JSON**: caught during `json.loads`; the program emits an `error` entry for that prompt.
- **Overlong generations**: a length guard stops generation to avoid infinite loops.
- **Schema mismatch**: input files are validated with Pydantic to prevent unexpected tool/prompt formats.

## Challenges faced

- **Keeping the output strictly structured**: small models often drift into explanations. Forcing an opening brace and using a strict “JSON-only” instruction helped.
- **Stopping at the right time**: the “balanced braces” heuristic is simple but effective for short JSON objects; a max-length cutoff prevents runaway generation.
- **Token-level constraints with arbitrary tokenizers**: mapping string tokens to ids requires reading the tokenizer vocabulary from the model files.

## Testing strategy

- **Dataset-based validation**: the provided prompts in `data/input/function_calling_tests.json` are used as a regression set.
- **Output sanity checks**:
	- The program must always produce a JSON array in `data/output/function_calls.json`.
	- Each item must contain `prompt`, `name`, and `parameters`.
- **Static checks**: `make lint` runs `flake8` and `mypy` over `src/`.

## Resources

### References

- Hugging Face Transformers — model & tokenizer basics: https://huggingface.co/docs/transformers/
- Hugging Face Hub — file download & caching: https://huggingface.co/docs/huggingface_hub/
- Qwen model card (default model used here): https://huggingface.co/Qwen/Qwen3-0.6B
- Pydantic — data validation: https://docs.pydantic.dev/
- General background on constrained decoding (survey/overview style):
	- “Guided (constrained) decoding” concepts are commonly discussed in structured generation literature and libraries (e.g., JSON/CFG-guided generation).

### How AI was used

- AI assistance (GitHub Copilot Chat, GPT-5.2) was used to draft and refine this README (structure, wording, and completeness versus the 42 README requirements).
- No claims are made here about AI generating or modifying the project source code.