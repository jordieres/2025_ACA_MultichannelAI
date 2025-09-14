"""
Logging utilities for the multimodal_fin package.

Provides a standardized logger with timestamps and log levels.
"""

import logging
from typing import List, Tuple, Optional

# Default logger for this module (used if none is provided explicitly)
_default_logger = None

def get_logger(name: str) -> logging.Logger:
    """Create and configure a logger for a given module or component.

    Ensures a consistent logging format and level across the package.

    Args:
        name (str): Name of the logger (typically use __name__).

    Returns:
        logging.Logger: A configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    logger.propagate = False

    return logger


def log_ensemble_prediction(
    model_outputs: List[Tuple[str, str, float]],
    final_label: str,
    final_confidence: float,
    logger: Optional[logging.Logger] = None,
):
    """
    Logs the ensemble classification results in a clean, readable format.

    Args:
        model_outputs (List[Tuple[str, str, float]]): Tuples (model_name, predicted_label, confidence).
        final_label (str): Final combined prediction label.
        final_confidence (float): Final confidence (0 to 100).
        logger (Optional[logging.Logger]): Logger to use. If None, uses default package logger.
    """
    if logger is None:
        global _default_logger
        if _default_logger is None:
            _default_logger = get_logger("multimodal_fin")
        logger = _default_logger

    lines = [
        "========== Ensemble Classification ==========",
        *(f"Model {name:<14} → {label} ({conf:.2f}%)" for name, label, conf in model_outputs),
        "",
        f"✅ Final prediction: {final_label} | Combined confidence: {final_confidence:.2f}%",
        "============================================\n"
    ]

    logger.info("\n" + "\n".join(lines))
