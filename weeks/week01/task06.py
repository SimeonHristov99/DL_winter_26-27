import numpy as np


def main() -> None:
    vector = np.array([1, 2, 3, 4, 5])
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    volume = np.array([[[ 0,  1,  2,  3], [ 4,  5,  6,  7], [ 8,  9, 10, 11]], 
                       [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]])
    
    print(
        f"""
            1D array (vector): shape=({vector.shape},), dtype={vector.dtype}, ndim={vector.ndim}, size={vector.size}
            {vector}
            2D array (matrix): shape={matrix.shape}, dtype={matrix.dtype}, ndim={matrix.ndim}, size={matrix.size}
            {matrix}
            3D array (volume): shape={volume.shape}, dtype={volume.dtype}, ndim={volume.ndim}, size={volume.size}
            {volume}
        """
        )