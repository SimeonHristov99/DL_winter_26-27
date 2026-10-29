import numpy as np


def main() -> None:
    matrix_a = np.array([[1,2], [3,4]])
    matrix_b = np.array([[5,6], [7,8]])

    print(f"Matrix A:\n{matrix_a}")
    print(f"Matrix B:\n{matrix_b}")
    print(f"Elementwise:\n{matrix_a * matrix_b}")
    print(f"Matrix multiplication:\n{matrix_a @ matrix_b}")
    print(f"The transpose of A:\n{matrix_a.transpose()}")


    
