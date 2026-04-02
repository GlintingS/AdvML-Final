import logging
from pathlib import Path

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_mae(models, mae_values):
    """
    Plot a bar chart of Mean Absolute Error (MAE) for different models.

    Args:
        models (list): List of model names.
        mae_values (list): List of MAE values corresponding to the models.

                Example: LR_mae=0.5, RF_mae=0.3
    """

    project_root = Path(__file__).resolve().parents[2]
    output_path = project_root / "mae_comparison.png"

    logger.info("Generating model comparison chart")
    try:
        plt.bar(models, mae_values)
        plt.ylabel("Mean Absolute Error")
        plt.title("Model Comparison")
        plt.savefig(output_path)
        logger.info("Chart saved to %s", output_path)
        plt.show()
    except Exception as exc:
        logger.error("Failed to generate chart: %s", exc)
        raise
