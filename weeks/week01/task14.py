import numpy as np

from dl_lib.preprocessing.functional import min_max_normalize, standardize


def main() -> None:
    array = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    normalized_array = min_max_normalize(array)
    standardized_array = standardize(array)

    print(
        f"Original: {array}\n",
        f"Min-max: {normalized_array}\n",
        f"Standardized: {standardized_array}\n",
        f"Min-max range: [{np.min(normalized_array):.4f}, {np.max(normalized_array):.4f}] \n",
        f"Standardized mean/std: {np.mean(standardized_array):.4f} / {np.std(standardized_array):.4f}"
        )