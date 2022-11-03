import gc
import torch
import time

class Timer():
    #__slots__ = ('start_time', 'end_time')
    def __init__(self) -> None:
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.synchronize()
        self.start_time = time.time()
        
    def __exit__(self):
        torch.cuda.synchronize()
        self.end_time = time.time()
        
    def __call__(self) -> float:
        if self.start_time is None or self.end_time is None:
            raise RuntimeError('The timer has not been used yet')
        return self.end_time - self.start_time