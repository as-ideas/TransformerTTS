import traceback


def ignore_exception(f):
    def apply_func(*args, **kwargs):
        try:
            f(*args,**kwargs)
        except Exception:
            print(f'Catched exception in {f}:')
            traceback.print_exc()
    return apply_func
