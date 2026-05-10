
#   time_func.py

import time
from functools import wraps

def time_func(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        # end_time = time.time()
        print(f" {func.__name__} executed in {time.time() - start_time:.4f} seconds")
        return result

    return wrapper


if __name__ == "__main__":
    @time_func
    def long_time(n):
        for i in range(n):
            for j in range(100000):
                i*j


    long_time(5)
