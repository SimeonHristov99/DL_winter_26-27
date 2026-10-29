import numpy as np


def main() -> None:
    matrix = np.array([[1,2,3], [4,5,6]])
    vector = np.array([10, 20, 30])
    column_vector = np.array([[100], [200]])

    print(f"Matrix + scalar (10):\n{matrix + 10}\n")
    print(f"Matrix {matrix.shape} + row vector {vector.shape}:\n{matrix + vector}\n")
    print(f"Matrix {matrix.shape} + column vector {column_vector.shape}:\n{matrix + column_vector}\n")

    try:
        vector_1d = np.array([10, 20])
        err = matrix + vector_1d
    except ValueError as e:
        print(f"Matrix (2, 3) + incompatible (2,) raises:\n {e}")