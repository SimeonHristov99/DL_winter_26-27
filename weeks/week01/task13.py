import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

SEED = 42
NUM = 10000
BINS = 30

def main() -> None:
    rng = np.random.default_rng(seed=SEED)

    uniform_samples = rng.uniform(size=NUM)
    print_output("Uniform", uniform_samples)

    normal_samples = rng.standard_normal(NUM)
    print_output("Normal", normal_samples)

    exponential_samples = rng.exponential(size=NUM)
    print_output("Exponential", exponential_samples)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    plot_distribution_histogram(ax1, "Uniform", uniform_samples)
    plot_distribution_histogram(ax2, "Normal", normal_samples)
    plot_distribution_histogram(ax3, "Exponential", exponential_samples)

    fig.tight_layout()
    plt.show()


def print_output(distribution_name: str, samples: np.ndarray) -> None:
    print(
        f"{distribution_name}: mean={np.mean(samples):.4f}, " 
        f"std={np.std(samples):.4f}, " 
        f"min={np.min(samples):.4f}, " 
        f"max={np.max(samples):.4f}"
        )
    
def plot_distribution_histogram(ax: Axes, distribution_name: str, samples: np.ndarray) -> None:
    ax.hist(samples, bins=BINS)
    ax.set_title(distribution_name)
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    ax.grid(True)
