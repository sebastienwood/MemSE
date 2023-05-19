from abc import ABC


class Optimizer(ABC):
    def ascent_step(self, *args, **kwargs):
        pass

    def descent_step(self, *args, **kwargs):
        pass
