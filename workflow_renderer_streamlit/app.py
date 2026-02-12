"""Generic Streamlit renderer for AppSpec providers."""

from __future__ import annotations

import argparse
import glob
import html
import os
import sys
import time
from typing import Any

import pandas as pd
import streamlit as st

from workflow_runtime.discovery import load_app_spec

CRSD_FILE_PATTERNS = ["*.crsd", "*.CRSD", "*.nitf", "*.NITF", "*.ntf", "*.NTF"]


def _render_param_inputs(prefix: str, params) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for spec in params:
        key = f"{prefix}_{spec.key}"
        if spec.type == "dropdown":
            options = [opt.get("value") for opt in spec.options]
            labels_map = {opt.get("value"): opt.get("label", opt.get("value")) for opt in spec.options}
            default_idx = options.index(spec.default) if spec.default in options else 0
            val = st.sidebar.selectbox(
                spec.label,
                options=options,
                index=default_idx if options else 0,
                format_func=lambda x: labels_map.get(x, x),
                key=key,
                help=spec.help or None,
            )
        elif spec.type == "number":
            val = st.sidebar.number_input(
                spec.label,
                value=float(spec.default) if spec.default is not None else 0.0,
                min_value=float(spec.min) if spec.min is not None else None,
                max_value=float(spec.max) if spec.max is not None else None,
                step=float(spec.step) if spec.step is not None else None,
                key=key,
                help=spec.help or None,
            )
        elif spec.type == "checkbox":
            val = st.sidebar.checkbox(
                spec.label,
                value=bool(spec.default) if spec.default is not None else False,
                key=key,
                help=spec.help or None,
            )
        else:
            val = st.sidebar.text_input(
                spec.label,
                value=str(spec.default) if spec.default is not None else "",
                key=key,
                help=spec.help or None,
            )
        values[spec.key] = val
    return values


def _scan_crsd_files(directory_path: str) -> list[str]:
    files: list[str] = []
    if not directory_path or not os.path.isdir(directory_path):
        return files
    for pattern in CRSD_FILE_PATTERNS:
        files.extend(glob.glob(os.path.join(directory_path, pattern)))
    return sorted(files)


def _discover_crsd_channels(file_path: str) -> list[str]:
    """Read channel IDs from a CRSD file for initializer dropdowns."""
    if not file_path or not os.path.isfile(file_path):
        return []
    try:
        import sarkit.crsd as crsd
    except Exception:
        return []

    channel_ids: list[str] = []
    try:
        with open(file_path, "rb") as f:
            reader = crsd.Reader(f)
            root = reader.metadata.xmltree.getroot()
            namespace = "http://api.nsgreg.nga.mil/schema/crsd/1.0"
            channels = root.findall(f".//{{{namespace}}}Channel")
            for channel in channels:
                ch_id = channel.find(f"{{{namespace}}}ChId")
                if ch_id is not None and ch_id.text:
                    channel_ids.append(ch_id.text)
    except Exception:
        return []
    return channel_ids


def _render_initializer_inputs(prefix: str, params) -> dict[str, Any]:
    """
    Render initializer parameters.

    Special-case CRSD-style params:
    - RX: `crsd_directory` -> `crsd_file` dropdown -> `channel_id` dropdown
    - TX (optional): `tx_crsd_directory` -> `tx_crsd_file` dropdown -> `tx_channel_id` dropdown
    """
    specs_by_key = {p.key: p for p in params}
    has_crsd_scan = "crsd_directory" in specs_by_key and "crsd_file" in specs_by_key

    if not has_crsd_scan:
        return _render_param_inputs(prefix, params)

    values: dict[str, Any] = {}

    # Render directory first
    dir_spec = specs_by_key["crsd_directory"]
    dir_key = f"{prefix}_{dir_spec.key}"
    directory = st.sidebar.text_input(
        dir_spec.label,
        value=str(dir_spec.default) if dir_spec.default is not None else "",
        key=dir_key,
        help=dir_spec.help or None,
    )
    values["crsd_directory"] = directory

    # Discover RX files and render dropdown
    discovered = _scan_crsd_files(directory)
    file_spec = specs_by_key["crsd_file"]
    file_key = f"{prefix}_{file_spec.key}"
    if discovered:
        file_names = [os.path.basename(f) for f in discovered]
        if file_key not in st.session_state:
            st.session_state[file_key] = file_names[0]
        selected_name = st.sidebar.selectbox(
            file_spec.label,
            options=file_names,
            index=file_names.index(st.session_state[file_key]) if st.session_state[file_key] in file_names else 0,
            key=file_key,
            help=file_spec.help or None,
        )
        values["crsd_file"] = selected_name
    else:
        st.sidebar.warning("No CRSD files discovered in directory.")
        values["crsd_file"] = st.sidebar.text_input(
            file_spec.label,
            value=str(file_spec.default) if file_spec.default is not None else "",
            key=file_key,
            help=file_spec.help or None,
        )

    selected_path = ""
    if discovered and values.get("crsd_file"):
        selected_name = values["crsd_file"]
        for fpath in discovered:
            if os.path.basename(fpath) == selected_name:
                selected_path = fpath
                break

    # Optional TX directory + file picker
    tx_selected_path = ""
    if "tx_crsd_directory" in specs_by_key and "tx_crsd_file" in specs_by_key:
        tx_dir_spec = specs_by_key["tx_crsd_directory"]
        tx_dir_key = f"{prefix}_{tx_dir_spec.key}"
        tx_directory = st.sidebar.text_input(
            tx_dir_spec.label,
            value=str(tx_dir_spec.default) if tx_dir_spec.default is not None else "",
            key=tx_dir_key,
            help=tx_dir_spec.help or None,
        )
        values["tx_crsd_directory"] = tx_directory

        tx_files = _scan_crsd_files(tx_directory)
        tx_file_spec = specs_by_key["tx_crsd_file"]
        tx_file_key = f"{prefix}_{tx_file_spec.key}"
        tx_options = ["(None)"] + [os.path.basename(f) for f in tx_files]
        if tx_file_key not in st.session_state or st.session_state[tx_file_key] not in tx_options:
            st.session_state[tx_file_key] = "(None)"
        tx_selected_name = st.sidebar.selectbox(
            tx_file_spec.label,
            options=tx_options,
            index=tx_options.index(st.session_state[tx_file_key]),
            key=tx_file_key,
            help=tx_file_spec.help or None,
        )
        values["tx_crsd_file"] = tx_selected_name

        if tx_selected_name != "(None)":
            for fpath in tx_files:
                if os.path.basename(fpath) == tx_selected_name:
                    tx_selected_path = fpath
                    break

    # Render remaining params in declared order
    for spec in params:
        if spec.key in {"crsd_directory", "crsd_file", "tx_crsd_directory", "tx_crsd_file"}:
            continue
        key = f"{prefix}_{spec.key}"
        if spec.key == "channel_id":
            channels = _discover_crsd_channels(selected_path)
            if channels:
                if key not in st.session_state or st.session_state[key] not in channels:
                    st.session_state[key] = channels[0]
                val = st.sidebar.selectbox(
                    spec.label,
                    options=channels,
                    index=channels.index(st.session_state[key]),
                    key=key,
                    help=spec.help or None,
                )
            else:
                val = st.sidebar.text_input(
                    spec.label,
                    value=str(spec.default) if spec.default is not None else "",
                    key=key,
                    help=spec.help or None,
                )
            values[spec.key] = val
            continue

        if spec.key == "tx_channel_id":
            tx_channels = _discover_crsd_channels(tx_selected_path)
            if tx_channels:
                if key not in st.session_state or st.session_state[key] not in tx_channels:
                    st.session_state[key] = tx_channels[0]
                val = st.sidebar.selectbox(
                    spec.label,
                    options=tx_channels,
                    index=tx_channels.index(st.session_state[key]),
                    key=key,
                    help=spec.help or None,
                )
            else:
                val = st.sidebar.selectbox(
                    spec.label,
                    options=[""],
                    index=0,
                    key=key,
                    help=spec.help or None,
                )
            values[spec.key] = val
            continue

        if spec.type == "dropdown":
            options = [opt.get("value") for opt in spec.options]
            labels_map = {opt.get("value"): opt.get("label", opt.get("value")) for opt in spec.options}
            default_idx = options.index(spec.default) if spec.default in options else 0
            val = st.sidebar.selectbox(
                spec.label,
                options=options,
                index=default_idx if options else 0,
                format_func=lambda x: labels_map.get(x, x),
                key=key,
                help=spec.help or None,
            )
        elif spec.type == "number":
            val = st.sidebar.number_input(
                spec.label,
                value=float(spec.default) if spec.default is not None else 0.0,
                min_value=float(spec.min) if spec.min is not None else None,
                max_value=float(spec.max) if spec.max is not None else None,
                step=float(spec.step) if spec.step is not None else None,
                key=key,
                help=spec.help or None,
            )
        elif spec.type == "checkbox":
            val = st.sidebar.checkbox(
                spec.label,
                value=bool(spec.default) if spec.default is not None else False,
                key=key,
                help=spec.help or None,
            )
        else:
            val = st.sidebar.text_input(
                spec.label,
                value=str(spec.default) if spec.default is not None else "",
                key=key,
                help=spec.help or None,
            )
        values[spec.key] = val

    return values


def _render_results(results: dict[str, Any]):
    if not results:
        st.info("No results returned.")
        return

    if "results" in results:
        for item in results["results"]:
            if item["type"] == "text":
                for text in item.get("content", []):
                    st.markdown(text)
            elif item["type"] == "table":
                st.markdown(f"**{item.get('title', 'Table')}**")
                data = item.get("data", {})
                if isinstance(data, dict):
                    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
                else:
                    st.dataframe(data, use_container_width=True)
            elif item["type"] == "plot":
                st.plotly_chart(item.get("figure"), use_container_width=True)
        return

    for text in results.get("text", []):
        st.markdown(text)
    for table in results.get("tables", []):
        st.markdown(f"**{table.get('title', 'Table')}**")
        data = table.get("data", {})
        if isinstance(data, dict):
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
        else:
            st.dataframe(data, use_container_width=True)
    for fig in results.get("plots", []):
        st.plotly_chart(fig, use_container_width=True)


def _make_progress_callback(status_panel):
    with status_panel:
        progress_window = st.empty()
    completed_entries = []
    progress_state = {"active_entry": None}
    step_start_times = {}

    def render_progress_window():
        active_entry = progress_state["active_entry"]
        lines = completed_entries + ([active_entry] if active_entry else [])
        html_lines = []
        for line in lines:
            if line.startswith("__RUNNING__::"):
                running_text = html.escape(line.replace("__RUNNING__::", "", 1))
                html_lines.append(
                    "<li><span class='wr-live-spinner'></span>"
                    f"{running_text}</li>"
                )
            else:
                html_lines.append(f"<li>{html.escape(line)}</li>")
        progress_window.markdown(
            (
                "<style>"
                ".wr-live-scroll{max-height:12rem;overflow-y:auto;padding-right:0.3rem;}"
                ".wr-live-list{margin:0.25rem 0 0.25rem 1.1rem;padding:0;}"
                ".wr-live-list li{margin:0.2rem 0;list-style:disc;}"
                ".wr-live-spinner{display:inline-block;width:0.85rem;height:0.85rem;"
                "margin-right:0.45rem;border:2px solid currentColor;border-top-color:transparent;"
                "border-radius:50%;vertical-align:-0.1rem;animation:wrspin 0.8s linear infinite;}"
                "@keyframes wrspin{to{transform:rotate(360deg);}}"
                "</style>"
                f"<div class='wr-live-scroll'><ul class='wr-live-list'>{''.join(html_lines)}</ul></div>"
            ),
            unsafe_allow_html=True,
        )

    def progress_callback(step, status, detail=""):
        line = f"{step}"
        if detail:
            line += f" - {detail}"
        if status == "running":
            step_start_times[step] = time.perf_counter()
            progress_state["active_entry"] = f"__RUNNING__::{line}"
            status_panel.update(label=f"Executing: {step}...", state="running", expanded=True)
        elif status == "done":
            started = step_start_times.get(step)
            if started is not None:
                elapsed = time.perf_counter() - started
                completed_entries.append(f"✔️ {line} ({elapsed:.3f}s)")
            else:
                completed_entries.append(f"✔️ {line}")
            progress_state["active_entry"] = None
        elif status == "failed":
            completed_entries.append(f"❌ {line}")
            progress_state["active_entry"] = None
        render_progress_window()

    return progress_callback


def run_renderer(target_package: str):
    st.set_page_config(page_title="Workflow Renderer", layout="wide")
    st.title("Workflow Renderer")
    st.caption(f"Provider package: `{target_package}`")

    try:
        app_spec = load_app_spec(target_package)
    except Exception as exc:
        st.error(f"Failed to load app definition from {target_package}: {exc}")
        st.stop()

    st.sidebar.header(app_spec.app_name)

    if "initialized_context" not in st.session_state:
        st.session_state.initialized_context = None
    if "init_signature" not in st.session_state:
        st.session_state.init_signature = None
    if "last_results" not in st.session_state:
        st.session_state.last_results = None

    if not app_spec.initializers:
        st.error("No initializers defined by provider.")
        st.stop()

    initializer = app_spec.initializers[0]
    st.sidebar.subheader("Initialization")
    init_values = _render_initializer_inputs("init", initializer.params)
    init_signature = repr(sorted(init_values.items()))

    if st.session_state.initialized_context is None or st.session_state.init_signature != init_signature:
        try:
            st.session_state.initialized_context = initializer.initialize(init_values)
            st.session_state.init_signature = init_signature
            st.session_state.last_results = None
        except Exception as exc:
            st.sidebar.error(f"Initialization failed: {exc}")
            st.session_state.initialized_context = None

    context = st.session_state.initialized_context
    if context is None:
        st.info("Fix initialization inputs in the sidebar to load context.")
        return

    if not app_spec.workflows:
        st.warning("No workflows discovered.")
        return

    st.sidebar.markdown("---")
    workflow_ids = [w.id for w in app_spec.workflows]
    workflow_map = {w.id: w for w in app_spec.workflows}
    selected_id = st.sidebar.selectbox("Workflow", options=workflow_ids, format_func=lambda x: workflow_map[x].name)
    wf = workflow_map[selected_id]
    if wf.description:
        st.sidebar.caption(wf.description)

    st.sidebar.subheader("Workflow Parameters")
    wf_values = _render_param_inputs(f"wf_{wf.id}", wf.params)
    run_clicked = st.sidebar.button("Execute Workflow", use_container_width=True)

    if run_clicked:
        status_panel = st.status("Executing workflow...", state="running", expanded=True)
        callback = _make_progress_callback(status_panel)
        wf_values["_progress_callback"] = callback
        try:
            st.session_state.last_results = wf.run(context, wf_values)
            status_panel.update(label="Workflow complete", state="complete")
        except Exception as exc:
            status_panel.update(label=f"Workflow failed: {exc}", state="error")
            st.session_state.last_results = {
                "results": [{"type": "text", "content": [f"❌ Workflow failed: {exc}"]}]
            }

    if st.session_state.last_results is not None:
        _render_results(st.session_state.last_results)
    else:
        st.info("Choose a workflow and click Execute Workflow.")


def main():
    """CLI entry point for launching renderer with a target package."""
    parser = argparse.ArgumentParser(description="Generic Streamlit workflow renderer")
    parser.add_argument("--target-package", default="crsd_inspector")
    args = parser.parse_args()
    launch_streamlit_renderer(args.target_package)


def launch_streamlit_renderer(target_package: str):
    """Launch the generic Streamlit renderer for a specific provider package."""
    os.environ["WORKFLOW_TARGET_PACKAGE"] = target_package

    import streamlit.web.cli as stcli

    sys.argv = ["streamlit", "run", __file__]
    sys.exit(stcli.main())


if __name__ == "__main__":
    run_renderer(os.environ.get("WORKFLOW_TARGET_PACKAGE", "crsd_inspector"))
