from time import perf_counter
from typing import Callable, Tuple, TypeVar


T = TypeVar("T")


def timed_call(callback: Callable[[], T]) -> Tuple[T, float]:
    start = perf_counter()
    result = callback()
    elapsed_ms = (perf_counter() - start) * 1000.0
    return result, elapsed_ms
