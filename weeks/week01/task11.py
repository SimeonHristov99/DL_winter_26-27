import matplotlib.pyplot as plt
import numpy as np

SEED = 42
NUM_POINTS = 50
X_MIN = 0
X_MAX = 10
SLOPE = 2.0
INTERCEPT = 1.0

def main() -> None:
    rng = np.random.default_rng(seed=SEED)
    
    x = np.linspace(X_MIN, X_MAX, num = NUM_POINTS, endpoint = True)
    noise = rng.standard_normal(NUM_POINTS)
    y = SLOPE * x + INTERCEPT    + noise

    print(f"Generated {len(x)} points for y = {SLOPE}*x + {INTERCEPT} + noise")
    print(f"x range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"y range: [{y.min():.2f}, {y.max():.2f}]")

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y)

    plt.title("Synthetic Linear Data Generation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.tight_layout()      
    plt.show()