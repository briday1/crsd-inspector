"""CRSD Inspector GUI bridge API."""


def main():
    """
    Launch the GUI using the generic Streamlit renderer.

    CRSD Inspector owns this API while delegating rendering to
    `renderflow`.
    """
    try:
        from renderflow.streamlit_renderer import launch_streamlit_renderer
    except Exception as exc:
        raise RuntimeError(
            "renderflow is required for GUI launch."
        ) from exc

    launch_streamlit_renderer("crsd-inspector")
