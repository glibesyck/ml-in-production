import numpy as np
import time
import multiprocessing
import threading
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

NUM_TASKS = 100
NUM_WORKERS = multiprocessing.cpu_count()
MATRIX_SIZE = (2000, 2000)

# matrix multiplication as inference
def matrix_multiplication(_=None):
    A = np.random.rand(*MATRIX_SIZE)
    B = np.random.rand(*MATRIX_SIZE)
    return np.dot(A, B)

# additional util function
def worker_process(return_list):
    return_list.append(matrix_multiplication())

# single-threaded execution
def benchmark_single():
    start_time = time.time()
    _ = [matrix_multiplication() for _ in range(NUM_TASKS)]
    return time.time() - start_time

# using Thread
def benchmark_threading_threads():
    start_time = time.time()
    threads = []
    results = []
    
    def worker():
        result = matrix_multiplication()
        results.append(result)

    threads = [threading.Thread(target=worker) for _ in range(NUM_TASKS)]
    for thread in threads:
        thread.start()
    
    for thread in threads:
        thread.join()

    return time.time() - start_time

# using ThreadPoolExecutor
def benchmark_threading_pool():
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(matrix_multiplication) for _ in range(NUM_TASKS)]
        _ = [future.result() for future in futures]
    return time.time() - start_time

# using Process
def benchmark_multiprocessing_process():
    start_time = time.time()
    processes = []
    
    for _ in range(NUM_TASKS):
        process = multiprocessing.Process(target=matrix_multiplication)
        processes.append(process)
        process.start()
    
    for process in processes:
        process.join()

    return time.time() - start_time

# using Pool
def benchmark_multiprocessing_pool():
    start_time = time.time()
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        _ = pool.map(matrix_multiplication, range(NUM_TASKS))
    return time.time() - start_time

# using ProcessPoolExecutor
def benchmark_multiprocessing_executor():
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(matrix_multiplication) for _ in range(NUM_TASKS)]
        _ = [future.result() for future in futures]
    return time.time() - start_time

# AsyncIO
async def async_matrix_multiplication():
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, matrix_multiplication)

async def benchmark_asyncio():
    start_time = time.time()
    tasks = [async_matrix_multiplication() for _ in range(NUM_TASKS)]
    _ = await asyncio.gather(*tasks)
    return time.time() - start_time

if __name__ == "__main__":
    print(f"Single-threaded execution time: {benchmark_single():.4f} seconds")
    
    print(f"Threading (Thread) execution time: {benchmark_threading_threads():.4f} seconds")
    print(f"Threading (ThreadPoolExecutor) execution time: {benchmark_threading_pool():.4f} seconds")
    
    print(f"Multiprocessing (Process) execution time: {benchmark_multiprocessing_process():.4f} seconds")
    print(f"Multiprocessing (Pool) execution time: {benchmark_multiprocessing_pool():.4f} seconds")
    print(f"Multiprocessing (ProcessPoolExecutor) execution time: {benchmark_multiprocessing_executor():.4f} seconds")
    
    print(f"AsyncIO execution time: {asyncio.run(benchmark_asyncio()):.4f} seconds")
