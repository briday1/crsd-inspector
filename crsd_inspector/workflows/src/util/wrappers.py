"""
Workflow execution utilities - timing and progress tracking
"""
import time
import traceback


def emit_progress(metadata, step, status, detail=""):
    """Emit optional progress events to the app UI."""
    if metadata is None:
        return
    callback = metadata.get('_progress_callback')
    if callable(callback):
        callback(step=step, status=status, detail=detail)


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


def wrap_with_timing(fn, label, description, metadata):
    """Wrap a node function with timing and progress tracking.
    
    This wrapper adds '_timing' to the function's output dict and
    emits progress callbacks. It preserves all original outputs from the function.
    
    If the function fails, it returns a dict with error information and dummy outputs
    to prevent the entire graph from stopping.
    """
    def wrapper(inputs):
        # Emit progress start
        emit_progress(metadata, label, "running", description)
        
        # Time the execution
        start_time = time.perf_counter()
        try:
            output = fn(inputs)
            elapsed_s = time.perf_counter() - start_time
            
            # Add timing to output (ensure output is a dict)
            if not isinstance(output, dict):
                output = {'result': output}
            output['_timing'] = elapsed_s
            
            # Emit progress done
            emit_progress(metadata, label, "done", description)
            
            return output
            
        except Exception as e:
            elapsed_s = time.perf_counter() - start_time
            emit_progress(metadata, label, "failed", description)
            
            error_msg = f"ERROR in {label}: {str(e)}"
            
            # Return error information instead of crashing
            return {
                '_timing': elapsed_s,
                '_node_error': error_msg,
                '_node_error_traceback': traceback.format_exc(),
                '_node_failed': True
            }
    
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
