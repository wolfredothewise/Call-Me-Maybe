import argparse
import json
import os

from .solver import ConstrainedDecoding
from .parser import Parser
from pydantic import ValidationError


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--functions_definition", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    os.makedirs("data/output", exist_ok=True)

    args = parser.parse_args()

    try:
        data = Parser(args)
    except (FileNotFoundError, PermissionError) as e:
        print("File Error:", e)
        return

    except (json.JSONDecodeError) as e:
        print("Json Error:", e)
        return
    except (ValidationError) as e:
        print("Validation Error:", e)
        return

    ConstrainedDecoding(
        data.list_of_prompts,
        data.list_of_function_def,
        data.args_output
        )


if __name__ == "__main__":
    main()
