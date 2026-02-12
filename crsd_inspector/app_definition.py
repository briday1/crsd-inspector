"""CRSD provider definition: initializers + workflows."""

from __future__ import annotations

import glob
import importlib.util
import os
from pathlib import Path
from typing import Any

import numpy as np

from renderflow.contracts import AppSpec, InitializerSpec, ParamSpec, WorkflowSpec

CRSD_XML_NS = "http://api.nsgreg.nga.mil/schema/crsd/1.0"
CRSD_FILE_PATTERNS = ["*.crsd", "*.CRSD", "*.nitf", "*.NITF", "*.ntf", "*.NTF"]


def crsd_tag(local_name: str) -> str:
    """Build a namespaced CRSD XML tag."""
    return f"{{{CRSD_XML_NS}}}{local_name}"


def _param_specs_from_dict(params_dict: dict[str, dict[str, Any]]) -> list[ParamSpec]:
    specs: list[ParamSpec] = []
    for key, cfg in params_dict.items():
        specs.append(
            ParamSpec(
                key=key,
                label=cfg.get("label", key),
                type=cfg.get("type", "text"),
                default=cfg.get("default"),
                min=cfg.get("min"),
                max=cfg.get("max"),
                step=cfg.get("step"),
                options=cfg.get("options", []),
                help=cfg.get("help", ""),
            )
        )
    return specs


def _extract_channel_ids(root) -> list[str]:
    channels = root.findall(f".//{crsd_tag('Channel')}")
    return [ch.find(crsd_tag("ChId")).text for ch in channels if ch.find(crsd_tag("ChId")) is not None]


def _load_tx_waveform(reader, tx_file_path: str | None, tx_channel_id: str | None = None):
    tx_wfm = None
    if tx_file_path and os.path.isfile(tx_file_path):
        import sarkit.crsd as crsd

        with open(tx_file_path, "rb") as tx_f:
            tx_reader = crsd.Reader(tx_f)
            tx_root = tx_reader.metadata.xmltree.getroot()
            tx_channel_ids = _extract_channel_ids(tx_root)
            tx_wfm_array = tx_reader.read_support_array("TX_WFM")
            tx_wfm_array = np.asarray(tx_wfm_array)
            if tx_wfm_array.ndim == 1:
                tx_wfm = np.asarray(tx_wfm_array, dtype=np.complex64)
            else:
                row_idx = 0
                if tx_channel_id and tx_channel_ids and tx_channel_id in tx_channel_ids:
                    row_idx = tx_channel_ids.index(tx_channel_id)
                row_idx = min(row_idx, tx_wfm_array.shape[0] - 1)
                tx_wfm = np.asarray(tx_wfm_array[row_idx, :], dtype=np.complex64)
    else:
        try:
            tx_wfm_array = reader.read_support_array("TX_WFM")
            tx_wfm = np.asarray(tx_wfm_array[0, :], dtype=np.complex64)
        except Exception:
            pass
    return tx_wfm


def _scan_directory_for_crsd(directory_path: str) -> list[str]:
    files: list[str] = []
    if not directory_path or not os.path.isdir(directory_path):
        return files
    for pattern in CRSD_FILE_PATTERNS:
        files.extend(glob.glob(os.path.join(directory_path, pattern)))
    return sorted(files)


def _dropdown_options(values: list[str], include_none: bool = False) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    if include_none:
        options.append({"label": "(None)", "value": "(None)"})
    for value in values:
        options.append({"label": value, "value": value})
    return options


def _discover_channel_ids_from_file(file_path: str) -> list[str]:
    if not file_path or not os.path.isfile(file_path):
        return []
    try:
        import sarkit.crsd as crsd
        with open(file_path, "rb") as f:
            reader = crsd.Reader(f)
            root = reader.metadata.xmltree.getroot()
            return _extract_channel_ids(root)
    except Exception:
        return []


def _resolve_input_crsd_file(params: dict[str, Any]) -> tuple[str, list[str]]:
    """
    Resolve target CRSD file.
    """
    directory = str(params.get("crsd_directory", "")).strip() or os.getcwd()
    if not os.path.isdir(directory):
        raise ValueError(f"Directory not found: {directory}")

    files = _scan_directory_for_crsd(directory)
    if not files:
        raise ValueError(f"No CRSD files found in directory: {directory}")

    requested = str(params.get("crsd_file", "")).strip()
    if not requested:
        return files[0], files

    if os.path.isabs(requested) and os.path.isfile(requested):
        return requested, files

    by_name = [f for f in files if os.path.basename(f) == requested]
    if by_name:
        return by_name[0], files

    by_substring = [f for f in files if requested in os.path.basename(f)]
    if by_substring:
        return by_substring[0], files

    raise ValueError(
        f"Requested file '{requested}' not found in {directory}. "
        f"Available: {', '.join(os.path.basename(f) for f in files)}"
    )


def _resolve_optional_tx_file(params: dict[str, Any]) -> str | None:
    directory = str(params.get("tx_crsd_directory", "")).strip()
    selected = str(params.get("tx_crsd_file", "")).strip()

    if not directory or not selected or selected == "(None)":
        return None
    if not os.path.isdir(directory):
        return None

    files = _scan_directory_for_crsd(directory)
    if not files:
        return None

    by_name = [f for f in files if os.path.basename(f) == selected]
    if by_name:
        return by_name[0]
    return None


def initialize_crsd(params: dict[str, Any]) -> dict[str, Any]:
    """Initializer for CRSD files."""
    import sarkit.crsd as crsd

    file_path, discovered_files = _resolve_input_crsd_file(params)
    if not os.path.isfile(file_path):
        raise ValueError(f"CRSD file not found: {file_path}")

    selected_channel = str(params.get("channel_id", "")).strip()
    tx_crsd_file = _resolve_optional_tx_file(params)
    tx_channel_id = str(params.get("tx_channel_id", "")).strip()

    channel_data = {}
    channel_ids: list[str] = []
    sample_rate_hz = None
    prf_hz = None

    with open(file_path, "rb") as f:
        reader = crsd.Reader(f)
        root = reader.metadata.xmltree.getroot()

        channel_ids = _extract_channel_ids(root)

        if channel_ids:
            for ch_id in channel_ids:
                channel_data[ch_id] = reader.read_signal(ch_id)
        elif hasattr(reader, "read_signal_block"):
            channel_data["CHAN1"] = reader.read_signal_block(0, 0, 256, 256)
            channel_ids = ["CHAN1"]

        radar_params = root.find(f".//{crsd_tag('RadarParameters')}")
        if radar_params is not None:
            sample_rate = radar_params.find(f".//{crsd_tag('SampleRate')}")
            if sample_rate is not None and sample_rate.text:
                sample_rate_hz = float(sample_rate.text)

            prf = radar_params.find(f".//{crsd_tag('PRF')}")
            if prf is not None and prf.text:
                prf_hz = float(prf.text)

        tx_wfm = _load_tx_waveform(reader, tx_crsd_file, tx_channel_id)

        file_header_kvps = {}
        try:
            if hasattr(reader, "file_header"):
                file_header_kvps = reader.file_header.kvps
        except Exception:
            pass

    if not channel_ids:
        raise ValueError(f"No channels could be loaded from {file_path}")

    if not selected_channel:
        selected_channel = channel_ids[0]
    if selected_channel not in channel_data:
        raise ValueError(
            f"Selected channel '{selected_channel}' not found. Available: {', '.join(channel_ids)}"
        )

    metadata = {
        "filename": os.path.basename(file_path),
        "file_path": file_path,
        "available_crsd_files": discovered_files,
        "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
        "shape": channel_data[selected_channel].shape,
        "num_channels": len(channel_ids),
        "channel_ids": channel_ids,
        "selected_channel": selected_channel,
        "channel_data": channel_data,
        "tx_crsd_file": tx_crsd_file or "",
        "tx_channel_id": tx_channel_id,
        "tx_source": (
            f"External TX file: {os.path.basename(tx_crsd_file)}"
            + (f" (channel: {tx_channel_id})" if tx_channel_id else " (channel: first available)")
            if tx_crsd_file
            else "Embedded TX support array from RX file"
        ),
        "file_header_kvps": file_header_kvps,
    }
    if sample_rate_hz is not None:
        metadata["sample_rate_hz"] = sample_rate_hz
    if prf_hz is not None:
        metadata["prf_hz"] = prf_hz

    return {
        "signal_data": channel_data[selected_channel],
        "metadata": metadata,
        "tx_wfm": tx_wfm,
    }


def _discover_workflow_specs() -> list[WorkflowSpec]:
    workflows_dir = Path(__file__).parent / "workflows"
    if not workflows_dir.is_dir():
        return []

    specs: list[WorkflowSpec] = []
    for filepath in sorted(glob.glob(str(workflows_dir / "*.py"))):
        filename = os.path.basename(filepath)
        if filename.startswith("_") or filename in {"workflow.py", "__init__.py"}:
            continue

        module_name = filename[:-3]
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:
            continue

        if not hasattr(module, "run_workflow"):
            continue

        if hasattr(module, "workflow"):
            wf_name = module.workflow.name
            wf_desc = module.workflow.description
            wf_params = getattr(module.workflow, "params", {})
        else:
            wf_name = getattr(module, "WORKFLOW_NAME", module_name)
            wf_desc = getattr(module, "WORKFLOW_DESCRIPTION", "")
            wf_params = getattr(module, "PARAMS", {})

        param_specs = _param_specs_from_dict(wf_params)

        def make_run(module_ref):
            def _run(context: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
                metadata = dict(context.get("metadata", {}))
                metadata.update(params)
                tx_wfm = context.get("tx_wfm")
                if tx_wfm is not None:
                    metadata["tx_wfm"] = tx_wfm
                return module_ref.run_workflow(
                    signal_data=context["signal_data"],
                    metadata=metadata,
                )

            return _run

        specs.append(
            WorkflowSpec(
                id=module_name,
                name=wf_name,
                description=wf_desc,
                params=param_specs,
                run=make_run(module),
            )
        )
    return specs


def get_app_spec() -> AppSpec:
    """Return CRSD provider definition for generic renderers."""
    default_rx_dir = "examples"
    default_tx_dir = "examples"
    rx_files = [os.path.basename(f) for f in _scan_directory_for_crsd(default_rx_dir)]
    tx_files = [os.path.basename(f) for f in _scan_directory_for_crsd(default_tx_dir)]
    default_rx_file_path = os.path.join(default_rx_dir, rx_files[0]) if rx_files else ""
    default_tx_file_path = os.path.join(default_tx_dir, tx_files[0]) if tx_files else ""
    rx_channels = _discover_channel_ids_from_file(default_rx_file_path)
    tx_channels = _discover_channel_ids_from_file(default_tx_file_path)

    initializer = InitializerSpec(
        id="crsd_file_initializer",
        name="CRSD File Initializer",
        description="Load a CRSD file and selected channel into workflow context.",
        params=[
            ParamSpec(
                key="crsd_directory",
                label="CRSD Directory",
                type="text",
                default=default_rx_dir,
                help="Directory to scan for CRSD/NITF files.",
            ),
            ParamSpec(
                key="crsd_file",
                label="CRSD File Name (optional)",
                type="dropdown",
                default=rx_files[0] if rx_files else "",
                options=_dropdown_options(rx_files),
                help="Basename or substring match within scanned directory. Defaults to first file found.",
            ),
            ParamSpec(
                key="channel_id",
                label="Channel ID (optional)",
                type="dropdown",
                default=rx_channels[0] if rx_channels else "",
                options=_dropdown_options(rx_channels),
                help="Leave empty to use the first available channel.",
            ),
            ParamSpec(
                key="tx_crsd_directory",
                label="TX CRSD Directory (optional)",
                type="text",
                default=default_tx_dir,
                help="Directory to scan for optional external TX CRSD files.",
            ),
            ParamSpec(
                key="tx_crsd_file",
                label="TX CRSD File (optional)",
                type="dropdown",
                default="(None)",
                options=_dropdown_options(tx_files, include_none=True),
                help="Optional TX file from the selected TX directory.",
            ),
            ParamSpec(
                key="tx_channel_id",
                label="TX Channel ID (optional)",
                type="dropdown",
                default=tx_channels[0] if tx_channels else "",
                options=_dropdown_options(tx_channels),
                help="Optional TX channel. Defaults to first available in TX file.",
            ),
        ],
        initialize=initialize_crsd,
    )

    return AppSpec(
        app_name="CRSD Inspector Provider",
        initializers=[initializer],
        workflows=_discover_workflow_specs(),
    )
