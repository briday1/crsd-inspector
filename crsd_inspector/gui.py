"""CRSD Inspector GUI bridge API."""


def main():
    """
    Launch the GUI using the generic Streamlit renderer.

    CRSD Inspector owns this API while delegating rendering to
    `workflow_renderer_streamlit`.
    """
    try:
        from workflow_renderer_streamlit.app import launch_streamlit_renderer
    except Exception as exc:
        raise RuntimeError(
            "workflow_renderer_streamlit is required for GUI launch."
        ) from exc

    launch_streamlit_renderer("crsd_inspector")

