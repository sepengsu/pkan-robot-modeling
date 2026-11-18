"""Interpretable residual branches; all outputs are generalized torque in Nm."""

from __future__ import annotations

import math

import torch
from torch import nn


class SplineBasis(nn.Module):
    def __init__(self, intervals: int = 5, degree: int = 3):
        super().__init__()
        if intervals < 2 or degree != 3:
            raise ValueError("expected at least two intervals and cubic splines")
        self.intervals = intervals
        self.degree = degree
        spacing = 2.0 / intervals
        self.register_buffer("knots", torch.linspace(-1 - degree * spacing, 1 + degree * spacing,
                                                    intervals + 2 * degree + 1))

    @property
    def n_basis(self) -> int:
        return len(self.knots) - self.degree - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Identical bounded-grid convention to the legacy SplineKANLayer.
        x = torch.tanh(x).unsqueeze(-1)
        knots = self.knots.to(x.dtype)
        basis = ((x >= knots[:-1]) & (x < knots[1:])).to(x.dtype)
        for order in range(1, self.degree + 1):
            count = len(knots) - order - 1
            left = (x - knots[:count]) / (knots[order:order + count] - knots[:count])
            right = (knots[order + 1:order + count + 1] - x) / (knots[order + 1:order + count + 1] - knots[1:count + 1])
            basis = left * basis[..., :count] + right * basis[..., 1:count + 1]
        return basis


class JointwiseSpline(nn.Module):
    """Six independent 1D cubic splines, not a multilayer KAN."""

    def __init__(self, intervals: int = 5):
        super().__init__()
        self.basis = SplineBasis(intervals)
        self.coefficients = nn.Parameter(torch.zeros(6, self.basis.n_basis))
        self.bias = nn.Parameter(torch.zeros(6))

    def forward(self, dq: torch.Tensor) -> torch.Tensor:
        if dq.ndim != 2 or dq.shape[1] != 6:
            raise ValueError("jointwise branch expects [batch, 6] velocities")
        return torch.einsum("bin,in->bi", self.basis(dq), self.coefficients) + self.bias

    def smoothness(self) -> torch.Tensor:
        return torch.diff(self.coefficients, n=2, dim=-1).square().mean()


class KANLayer(nn.Module):
    def __init__(self, inputs: int, outputs: int, intervals: int = 5):
        super().__init__()
        self.inputs = inputs
        self.basis = SplineBasis(intervals)
        self.coefficients = nn.Parameter(torch.empty(outputs, inputs, self.basis.n_basis))
        self.bias = nn.Parameter(torch.zeros(outputs))
        nn.init.normal_(self.coefficients, std=0.03 / math.sqrt(inputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.inputs:
            raise ValueError(f"KAN layer expects [batch, {self.inputs}]")
        return torch.einsum("bin,oin->bo", self.basis(x), self.coefficients) + self.bias

    def smoothness(self) -> torch.Tensor:
        return torch.diff(self.coefficients, n=2, dim=-1).square().mean()


class CoupledKAN(nn.Module):
    """Two edge-spline layers: [12, 8, 6], or [18, 8, 6] for ablation."""

    def __init__(self, inputs: int = 12, intervals: int = 5):
        super().__init__()
        if inputs not in (12, 18):
            raise ValueError("coupled branch expects q,dq with optional ddq")
        self.layer1 = KANLayer(inputs, 8, intervals)
        self.layer2 = KANLayer(8, 6, intervals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.layer1(x))

    def smoothness(self) -> torch.Tensor:
        return self.layer1.smoothness() + self.layer2.smoothness()


class ResidualMLP(nn.Module):
    """[12, 25, 25, 6] closely matches the 5-grid KAN parameter count."""

    def __init__(self, inputs: int = 12, width: int = 25):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(inputs, width), nn.SiLU(),
                                     nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def smoothness(self) -> torch.Tensor:
        return next(self.parameters()).new_zeros(())


class PhysicsStructured(nn.Module):
    """Add a frozen, pretrained jointwise branch and a learned coupled branch."""

    def __init__(self, jointwise: JointwiseSpline, coupled: CoupledKAN):
        super().__init__()
        self.jointwise = jointwise
        self.coupled = coupled
        for parameter in self.jointwise.parameters():
            parameter.requires_grad_(False)

    def forward(self, dq: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        joint = self.jointwise(dq)
        coupled = self.coupled(x)
        return joint + coupled, joint, coupled


def parameter_count(model: nn.Module, *, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if not trainable_only or p.requires_grad)
