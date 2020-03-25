import traceback
from time import time


def ignore_exception(f):
    def apply_func(*args, **kwargs):
        try:
            result = f(*args, **kwargs)
            return result
        except Exception:
            print(f'Catched exception in {f}:')
            traceback.print_exc()
            return None
    
    return apply_func


def time_it(f):
    def apply_func(*args, **kwargs):
        t_start = time()
        result = f(*args, **kwargs)
        t_end = time()
        dur = round(t_end - t_start, ndigits=2)
        return result, dur
    
    return apply_func
