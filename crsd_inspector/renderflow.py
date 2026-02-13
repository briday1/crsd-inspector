"""CRSD Inspector provider contract for renderflow."""

from __future__ import annotations

from renderflow.autodefine import auto_build_app_spec

APP_NAME = "CRSD Inspector"
WORKFLOWS_PACKAGE = "crsd_inspector.workflows"


def get_app_spec():
    return auto_build_app_spec("crsd_inspector")
