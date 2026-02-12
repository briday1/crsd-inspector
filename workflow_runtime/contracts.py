"""Contracts used by data providers and generic renderers."""

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ParamSpec:
    """Declarative parameter definition for init/workflow UIs."""

    key: str
    label: str
    type: str = "text"
    default: Any = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: list[dict[str, Any]] = field(default_factory=list)
    help: str = ""


@dataclass
class InitializerSpec:
    """Definition of an initialization API for a data domain."""

    id: str
    name: str
    description: str
    params: list[ParamSpec]
    initialize: Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class WorkflowSpec:
    """Definition of a workflow with declared params and execution function."""

    id: str
    name: str
    description: str
    params: list[ParamSpec]
    run: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


@dataclass
class AppSpec:
    """Full provider definition consumed by generic renderers."""

    app_name: str
    initializers: list[InitializerSpec]
    workflows: list[WorkflowSpec]

