import numpy as np


def main() -> None:
    matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print(f"Matrix:\n{matrix}\n")

    print(f"Second row: {matrix[1]}")
    print(f"Third column: {matrix[:, 2]}")
    print(f"Submatrix:\n{matrix[0:2, 1:3]}")

    mask = matrix > 6
    filtered_matrix = matrix[mask]

    print(f"Boolean mask for more than 6::\n{mask}\n")
    print(f"Values satisfying the mask:\n{filtered_matrix}")
