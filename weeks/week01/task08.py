import time

import numpy as np


def main() -> None:

    array_rand = np.random.standard_normal(1000000)

    loop_total = 0
    start_time_loop = time.perf_counter()

    for num in array_rand:
        loop_total += num

    end_time_loop = time.perf_counter()
    execution_time_loop = end_time_loop - start_time_loop

    start_time_vectorized_sum = time.perf_counter()
    vectorized_total = np.sum(array_rand)
    end_time_vectorized_sum = time.perf_counter()

    execution_time_vectorized_sum = end_time_vectorized_sum - start_time_vectorized_sum

    print(f"Execution time for for loop sum: {execution_time_loop:.6f} seconds")
    print(f"Execution time for vectorized sum: {execution_time_vectorized_sum:.6f} seconds")
    print(f"Vectorized speedup: {execution_time_loop / execution_time_vectorized_sum:.6f}\n")

    print(f"Vectorized mean: {np.mean(array_rand)}")
    print(f"Vectorized std: {np.std(array_rand)}")
    print(f"Vectorized min/max: {np.min(array_rand)} / {np.max(array_rand)}")
