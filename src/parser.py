from typing import Dict, List, Literal
from pydantic import BaseModel, ConfigDict
import json
from argparse import Namespace


class Parser:
    def __init__(self, args: Namespace) -> None:
        self.list_of_prompts: List[Prompts]
        self.list_of_function_def: List[FuncDef]

        self.args_functions_definition = args.functions_definition
        self.args_input = args.input
        self.args_output = args.output

        self._load_content()

    def _load_content(self) -> None:
        with open(self.args_functions_definition, "r", encoding="utf-8") as f:
            def_content = json.load(f)
            self.list_of_function_def = parse_function_def(def_content)
        with open(self.args_input, "r", encoding="utf-8") as f:
            calling_content = json.load(f)
            self.list_of_prompts = parse_calling(calling_content)


class ParamDef(BaseModel):
    type: Literal["string", "number", "boolean"]


class ReturnDef(BaseModel):
    type: Literal["string", "number", "boolean"]


class FuncDef(BaseModel):
    # no campos extras que no esten definidos en el modelo
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: Dict[str, ParamDef]
    # pq tiene que ser un dict
    returns: ReturnDef


class Prompts(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str


def parse_function_def(content: list[dict]) -> list[FuncDef]:
    return [FuncDef(**c) for c in content]


def parse_calling(content: list[dict]) -> list[Prompts]:
    return [Prompts(**c) for c in content]
