import matplotlib.pyplot as plt
import numpy as np

SEED = 42
NUM = 1000

def main () -> None:
    radians = np.linspace(0, 2*np.pi, num=NUM)
    sin_values = np.sin(radians)
    cos_values = np.cos(radians)

    rng = np.random.default_rng(seed=SEED)
    normal_samples = rng.standard_normal(NUM)

    print(f"sin(x) range:{np.min(sin_values):.4f}, {np.max(sin_values):.4f}")
    print(f"cos(x) range:{np.min(cos_values):.4f}, {np.max(cos_values):.4f}")
    print(f"Histogram sample count: {len(normal_samples)}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(radians, sin_values, label='sin(x)')
    ax1.plot(radians, cos_values, label='cos(x)')
    ax1.set_title("Line plot")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    ax1.grid(True)

    ax2.hist(normal_samples, bins=30)
    ax2.set_title("Histogram")
    ax2.set_xlabel("value")
    ax2.set_ylabel("count")
    ax2.grid(True)

    fig.tight_layout()
    plt.show()  