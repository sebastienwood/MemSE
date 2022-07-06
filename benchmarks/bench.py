import sys
import time
from tqdm import tqdm

def timefunc(func, *args, **kwargs):
    """Time a function.
    args:
        iterations=3
    Usage example:
        timeit(myfunc, 1, b=2)
    """
    try:
        iterations = kwargs.pop("iterations")
    except KeyError:
        iterations = 3
    elapsed = sys.maxsize
    for _ in tqdm(range(iterations)):
        start = time.perf_counter()
        try:
            _ = func(*args, **kwargs)
        except AssertionError as e:
            pass
        
        elapsed = min(time.perf_counter() - start, elapsed)
    print(("Best of {} {}(): {:.9f}".format(iterations, func.__name__, elapsed)))
    return elapsed


if __name__ == '__main__':
    from tests.test_nn import seq, test_conv2duf
    timefunc(test_conv2duf, seq, True)
    timefunc(test_conv2duf, seq, False)