"""Discovery helpers for generic renderers."""

from importlib import import_module

from workflow_runtime.contracts import AppSpec


def load_app_spec(target_package: str) -> AppSpec:
    """
    Load provider app definition from `<target_package>.app_definition`.

    The target module must expose `get_app_spec() -> AppSpec`.
    """
    module = import_module(f"{target_package}.app_definition")
    if not hasattr(module, "get_app_spec"):
        raise AttributeError(
            f"{target_package}.app_definition is missing get_app_spec()"
        )
    spec = module.get_app_spec()
    if not isinstance(spec, AppSpec):
        raise TypeError("get_app_spec() must return workflow_runtime.contracts.AppSpec")
    return spec

