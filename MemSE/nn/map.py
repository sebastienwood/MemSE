from typing import Callable, Optional
from MemSE.misc import UniqueKeyDict


MEMSE_MAP = UniqueKeyDict()


def register_memse_mapping(dropin_for:Optional[set] = None) -> Callable:
    def inner_decorator(func: Callable):
        d = dropin_for if dropin_for else func.dropin_for
        for t in d:
            MEMSE_MAP[t] = func
        return func
    return inner_decorator
