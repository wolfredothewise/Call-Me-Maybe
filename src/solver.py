import json
from typing import Dict, List, Any
from .parser import Prompts, FuncDef
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from numpy import argmax


class ConstrainedDecoding:
    def __init__(
            self, list_of_prompts: List[Prompts],
            list_of_function_def: List[FuncDef], output: str
              ) -> None:

        self.list_of_prompts: List[Prompts] = list_of_prompts
        self.list_of_function_def: List[FuncDef] = list_of_function_def
        self.output: str = output

        self.model: Small_LLM_Model = Small_LLM_Model()
        self.vocab_path: str
        self.vocabulary: Dict[str, int]

        self._get_vocab_from_path()
        final_output = self.generate_output()
        self._write_output(final_output)

    def generate_output(self) -> List[Dict[str, Any]]:
        final_outputs = []

        for p in self.list_of_prompts:
            finished = False
            state: Dict[str, str] = {"opener": "{"}

            full_prompt = self.generate_prompt(p)
            input_ids_tensor = self.model.encode(full_prompt)
            input_ids_list = input_ids_tensor.cpu().view(-1).tolist()

            generated_tokens: List[int] = []

            while not finished:
                if "end" in state:
                    finished = True
                    break

                logits = self.model.get_logits_from_input_ids(input_ids_list)
                final_result = self._constrained_logic(state, logits)

                input_ids_list.append(final_result)
                generated_tokens.append(final_result)

                state = self._next_state(state, generated_tokens)

            texto_generado = self.model.decode(generated_tokens)
            try:
                final_outputs.append(self.format_output(texto_generado, p))
            except Exception as e:
                print(f"Exception captured: {e}")
                final_outputs.append({
                    "prompt": p.prompt,
                    "name": "error",
                    "parameters": {}
                })
        return final_outputs

    def _constrained_logic(self, state: Dict[str, str],
                           logits: List[float]) -> int:
        if "free" not in state:
            valid_token_str = list(state.values())[0]
            token_id = self.vocabulary.get(valid_token_str)

            if token_id is not None:
                new_logits = [
                    -float('inf') if i != token_id else logit
                    for i, logit in enumerate(logits)
                ]
                return int(argmax(new_logits))
        return int(argmax(logits))

    def _next_state(self, state: Dict[str, str],
                    generated_tokens: List[int]) -> Dict[str, str]:
        if "free" in state:
            dec_text = self.model.decode(generated_tokens)
            if dec_text.count("{") == dec_text.count("}") and \
               dec_text.count("{") > 0:
                return ({"end": "}"})
            elif len(generated_tokens) > 150:
                return ({"end": "}"})
            return {"free": ""}
        else:
            return (self._get_state(state))

    def _get_state(self, state: Dict[str, str]) -> Dict[str, str]:
        if "start" in state:
            return {"opener": "{"}
        else:
            return {"free": ""}

    def generate_prompt(self, p: Prompts) -> str:
        system_instructions = (
            "You are an AI. Respond ONLY with a JSON object"
            "representing the correct function to call.\n"
            "Format required: {\"name\": \"function_name\","
            "\"parameters\": {\"arg\": value}}\n"
            "<tools>\n"
        )
        tools_str = json.dumps(
            [
                f.model_dump()
                for f in self.list_of_function_def
            ], indent=2
        )
        system_instructions += tools_str + "\n</tools>\n\n"

        user_message = f"User: {p.prompt}\nAssistant: "
        full_prompt = system_instructions + user_message
        return full_prompt

    def _get_vocab_from_path(self) -> None:
        self.vocab_path = self.model.get_path_to_vocab_file()
        with open(self.vocab_path, "r", encoding="utf-8") as voc_file:
            self.vocabulary = json.load(voc_file)

    def format_output(self, text_generated: str, p: Prompts) -> Dict[str, Any]:
        start_idx = text_generated.find('{')
        end_idx = text_generated.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            json_str = text_generated[start_idx:end_idx]
            obj = json.loads(json_str)
            return {
                "prompt": p.prompt,
                "name": obj.get("name", ""),
                "parameters": obj.get("parameters", {})
            }
        else:
            raise Exception()

    def _write_output(self, final_outputs: List[Dict[str, Any]]) -> None:
        with open(self.output, "w", encoding="utf-8") as f:
            json.dump(final_outputs, f, indent=2)
