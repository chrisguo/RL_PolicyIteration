import time

"""
    Module for any logging methods required accross the solution

"""


def func_timer(func):
    """Times how long a function takes."""

    def f(*args, **kwargs):
        start = time.time()
        results = func(*args, **kwargs)
        print "\nElapsed: %.2fs" % (time.time() - start)
        return results

    return f
