from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from MemSE.quant import CrossBar
import enum
import torch
import torch.nn as nn
import opt_einsum as oe


class FORWARD_MODE(enum.Enum):
    BASE = enum.auto()
    MEMSE = enum.auto()
    MONTECARLO = enum.auto()


@dataclass
class MemSEReturn:
    out: torch.Tensor
    gamma: torch.Tensor
    gamma_shape: Optional[List[int]]
    power: torch.Tensor


@dataclass
class MontecarloReturn:
    out: torch.Tensor
    power: torch.Tensor


class MemSELayer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        """Base class for MemSE layer that shows all needed elements."""
        super().__init__()
        self.initialize_from_module(*args, **kwargs)
        if self.memristored:
            existing_k = [
                k
                for k in self.memristored.keys()
                if hasattr(self, k) and getattr(self, k) is not None
            ]
            tensors = {k: getattr(self, k) for k in existing_k}
            self._crossbar: CrossBar = CrossBar(self, tensors, self.__class__.__name__)
            self.register_buffer("Wmax", torch.tensor([0.0] * self.out_features))
            self.register_parameter(
                "Gmax", nn.Parameter(torch.tensor([0.0] * self.out_features))
            )
            assert all([k in self.memristored_einsum for k in existing_k])
            assert "out" in self.memristored_einsum

    def initialize_from_module(
        self, module: Optional[nn.Module] = None, *args, **kwargs
    ):
        pass

    @classmethod
    @property
    def dropin_for(cls):
        pass

    @property
    def tia_resistance(self) -> float:
        return self._crossbar.manager.tia_resistance

    @property
    def std_noise(self) -> float:
        return self._crossbar.manager.std_noise

    def forward(self, x, *args, **kwargs):
        match x:
            case MemSEReturn():
                return self.memse(x, *args, **kwargs)
            case MontecarloReturn():
                return self.memse_montecarlo(x, *args, **kwargs)
            case _:
                return self.functional_base(x, *args, **kwargs)

    def functional_base(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise ValueError("This property should be set in chilren class")

    def memse(
        self, previous_layer:MemSEReturn, *args, **kwargs
    ) -> MemSEReturn:
        raise ValueError("This property should be set in chilren class")

    def memse_montecarlo(
        self, previous_layer:MontecarloReturn, *args, **kwargs
    ) -> MontecarloReturn:
        if not self.memristored:
            return MontecarloReturn(out=self.functional_base(previous_layer.out, *args, **kwargs), power=previous_layer.power)
        ct = self.Gmax / self.Wmax
        noisy = self.get_noisy(
            ct
        )  # TODO should maybe pass around a dict to disambiguate

        # PRECOMPUTE CONVS
        zp_mu = self.functional_base(previous_layer.out, *noisy[0], *args, **kwargs)
        zm_mu = self.functional_base(previous_layer.out, *noisy[1], *args, **kwargs)

        # ENERGY
        if previous_layer.power is not None:
            sum_x_gd: torch.Tensor = previous_layer.out**2

            e_p_mem: torch.Tensor = torch.sum(
                self.functional_base(sum_x_gd, *noisy[2]).view(sum_x_gd.shape[0], -1), dim=(1)
            )
            e_p_tiap = torch.sum(
                (((zp_mu * self.tia_resistance) ** 2) / self.tia_resistance).view(sum_x_gd.shape[0], -1),
                dim=(1)
            )
            e_p_tiam = torch.sum(
                (((zm_mu * self.tia_resistance) ** 2) / self.tia_resistance).view(sum_x_gd.shape[0], -1),
                dim=(1)
            )

            previous_layer.power.add_(e_p_mem + e_p_tiap + e_p_tiam)

        return MontecarloReturn(
            out=self.tia_resistance
            * (
                torch.einsum(self.memristored_einsum["out"], zp_mu, 1 / ct)
                - torch.einsum(self.memristored_einsum["out"], zm_mu, 1 / ct)
            ),
            power=previous_layer.power
        )

    @property
    def out_features(self) -> int:
        return 0

    @property
    def memristored(self) -> dict:
        return {}

    @property
    def memristored_einsum(self) -> dict:
        return {}

    @property
    def memristored_real_shape(self) -> dict:
        """Overwrite the access method for the memristored keys if usage is different in Pytorch than the Memristor representation stored in the crossbar.

        Returns:
            dict: _description_
        """
        return {}

    def get_noisy(self, ct):
        noisy = []
        assert self._crossbar.scaled is False
        for k, i in self.memristored.items():
            old_shape = i.shape
            if k in self.memristored_real_shape:
                i = self.memristored_real_shape[k]
            if i is None:
                noisy.append((None,) * 3)
                continue
            w_noise = torch.normal(
                mean=0.0, std=self.std_noise, size=i.shape, device=i.device
            )
            w_noise_n = torch.normal(
                mean=0.0, std=self.std_noise, size=i.shape, device=i.device
            )
            sign_w = torch.sign(i)
            abs_w: torch.Tensor = oe.contract(
                self.memristored_einsum[k], torch.abs(i), ct
            ).to(i)
            Gpos = torch.clip(torch.where(sign_w > 0, abs_w, 0.0) + w_noise, min=0)
            Gneg = torch.clip(torch.where(sign_w < 0, abs_w, 0.0) + w_noise_n, min=0)
            self._crossbar.rescale([Gpos.view(old_shape), Gneg.view(old_shape), abs_w.view(old_shape)])
            noisy.append((Gpos.view(i.shape), Gneg.view(i.shape), abs_w.view(i.shape)))
        return list(zip(*noisy))
