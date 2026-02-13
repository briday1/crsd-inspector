"""Shared CRSD input loading for workflow-only execution."""

from __future__ import annotations

import glob
import os
from typing import Any

import numpy as np

CRSD_XML_NS = "http://api.nsgreg.nga.mil/schema/crsd/1.0"
CRSD_FILE_PATTERNS = ["*.crsd", "*.CRSD", "*.nitf", "*.NITF", "*.ntf", "*.NTF"]


def crsd_tag(local_name: str) -> str:
    return f"{{{CRSD_XML_NS}}}{local_name}"


def _scan_directory_for_crsd(directory_path: str) -> list[str]:
    files: list[str] = []
    if not directory_path or not os.path.isdir(directory_path):
        return files
    for pattern in CRSD_FILE_PATTERNS:
        files.extend(glob.glob(os.path.join(directory_path, pattern)))
    return sorted(files)


def _extract_channel_ids(root) -> list[str]:
    channels = root.findall(f".//{crsd_tag('Channel')}")
    return [ch.find(crsd_tag("ChId")).text for ch in channels if ch.find(crsd_tag("ChId")) is not None]


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


def default_crsd_param_specs(include_tx: bool = False) -> dict[str, dict[str, Any]]:
    default_rx_dir = "examples"
    default_tx_dir = "examples"
    rx_files = [os.path.basename(f) for f in _scan_directory_for_crsd(default_rx_dir)]
    default_rx_file_path = os.path.join(default_rx_dir, rx_files[0]) if rx_files else ""
    rx_channels = _discover_channel_ids_from_file(default_rx_file_path)

    params: dict[str, dict[str, Any]] = {
        "crsd_directory": {
            "type": "text",
            "label": "CRSD Directory",
            "default": default_rx_dir,
            "help": "Directory to scan for CRSD/NITF files.",
        },
        "crsd_file": {
            "type": "dropdown",
            "label": "CRSD File Name (optional)",
            "default": rx_files[0] if rx_files else "",
            "options": [{"label": value, "value": value} for value in rx_files],
            "help": "Basename or substring match within scanned directory. Defaults to first file found.",
        },
        "channel_id": {
            "type": "dropdown",
            "label": "Channel ID (optional)",
            "default": rx_channels[0] if rx_channels else "",
            "options": [{"label": value, "value": value} for value in rx_channels],
            "help": "Leave empty to use the first available channel.",
        },
    }

    if include_tx:
        tx_files = [os.path.basename(f) for f in _scan_directory_for_crsd(default_tx_dir)]
        default_tx_file_path = os.path.join(default_tx_dir, tx_files[0]) if tx_files else ""
        tx_channels = _discover_channel_ids_from_file(default_tx_file_path)
        params.update(
            {
                "tx_crsd_directory": {
                    "type": "text",
                    "label": "TX CRSD Directory (optional)",
                    "default": default_tx_dir,
                    "help": "Directory to scan for optional external TX CRSD files.",
                },
                "tx_crsd_file": {
                    "type": "dropdown",
                    "label": "TX CRSD File (optional)",
                    "default": "(None)",
                    "options": [{"label": "(None)", "value": "(None)"}] + [
                        {"label": value, "value": value} for value in tx_files
                    ],
                    "help": "Optional TX file from the selected TX directory.",
                },
                "tx_channel_id": {
                    "type": "dropdown",
                    "label": "TX Channel ID (optional)",
                    "default": tx_channels[0] if tx_channels else "",
                    "options": [{"label": value, "value": value} for value in tx_channels],
                    "help": "Optional TX channel. Defaults to first available in TX file.",
                },
            }
        )

    return params


def load_workflow_inputs(params: dict[str, Any], include_tx: bool = False) -> tuple[Any, dict[str, Any]]:
    import sarkit.crsd as crsd

    file_path, discovered_files = _resolve_input_crsd_file(params)
    if not os.path.isfile(file_path):
        raise ValueError(f"CRSD file not found: {file_path}")

    selected_channel = str(params.get("channel_id", "")).strip()
    tx_crsd_file = _resolve_optional_tx_file(params) if include_tx else None
    tx_channel_id = str(params.get("tx_channel_id", "")).strip() if include_tx else ""

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

        tx_wfm = _load_tx_waveform(reader, tx_crsd_file, tx_channel_id) if include_tx else None

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
        "file_header_kvps": file_header_kvps,
    }
    if sample_rate_hz is not None:
        metadata["sample_rate_hz"] = sample_rate_hz
    if prf_hz is not None:
        metadata["prf_hz"] = prf_hz

    if include_tx:
        metadata.update(
            {
                "tx_crsd_file": tx_crsd_file or "",
                "tx_channel_id": tx_channel_id,
                "tx_source": (
                    f"External TX file: {os.path.basename(tx_crsd_file)}"
                    + (f" (channel: {tx_channel_id})" if tx_channel_id else " (channel: first available)")
                    if tx_crsd_file
                    else "Embedded TX support array from RX file"
                ),
            }
        )
        if tx_wfm is not None:
            metadata["tx_wfm"] = tx_wfm

    return channel_data[selected_channel], metadata
