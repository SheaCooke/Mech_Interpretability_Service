import numpy as np

def get_extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower()


def numpy_safe(obj):
    """Recursively convert numpy types to native Python types for JSON serialisation."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: numpy_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [numpy_safe(i) for i in obj]
    return obj
