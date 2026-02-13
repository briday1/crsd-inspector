"""
Workflow execution utilities - timing and progress tracking
"""
import traceback

from renderflow.progress import emit_progress, wrap_with_timing


# `emit_progress` and `wrap_with_timing` are sourced from renderflow.progress.
# Keep this module focused on CRSD-inspector-specific utilities.


def safe_plot_wrapper(fn, label, output_keys):
    """
    Wrap a plotting function to catch exceptions and return None on error.
    This prevents one failing plot from stopping the entire workflow.
    
    Parameters
    ----------
    fn : callable
        The plotting function to wrap
    label : str
        Label for error messages
    output_keys : list of str
        List of output keys that should be set to None on error
    """
    def wrapper(inputs):
        try:
            result = fn(inputs)
            return result
        except Exception as e:
            error_msg = f"ERROR in {label}: {str(e)}"
            # Store error in a special key so we can retrieve it later
            result = {k: None for k in output_keys}
            result['_plot_error'] = error_msg
            result['_plot_error_traceback'] = traceback.format_exc()
            return result
    return wrapper


def make_window(n, window_type):
    """Create window function"""
    import numpy as np
    
    if (window_type is None) or (window_type == 'none'):
        return np.ones(n)
    if window_type == 'hamming':
        return np.hamming(n)
    if window_type == 'hanning':
        return np.hanning(n)
    if window_type == 'blackman':
        return np.blackman(n)
    return np.ones(n)


def downsample_heatmap(data_2d, max_width=2000, max_height=1000):
    """Downsample 2D array for efficient rendering"""
    rows, cols = data_2d.shape
    skip_x = max(1, cols // max_width)
    skip_y = max(1, rows // max_height)
    downsampled = data_2d[::skip_y, ::skip_x]
    return downsampled, skip_x, skip_y
