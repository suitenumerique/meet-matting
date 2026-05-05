from dataclasses import dataclass
from typing import Any, Literal

ParamType = Literal["int", "float", "bool", "str", "choice", "number"]


@dataclass
class ParameterSpec:
    """Self-describing parameter declaration. The UI uses this to build widgets."""

    name: str
    type: ParamType
    default: Any
    label: str
    help: str = ""
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    choices: list | None = None
