"""Prediction-driven fast state used at deployment.

The mechanism families are registered offline.  Online updates consume only
the current state, commanded action, task index, and observed next state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .model import ACTION_DIM, STATE_DIM, SpikeWorld

FAST_VALUES = 6_096
FAST_BYTES = FAST_VALUES * 4
ACTION_CAP = 0.05
ROUTES = ("source", "shear", "attenuation", "noise")


@dataclass(frozen=True)
class AdaptationConfig:
    router_observations: int = 4
    residual_learning_rate: float = 0.02
    residual_radius: float = 0.20
    action_cap: float = ACTION_CAP

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> AdaptationConfig:
        """Parse the deployment controls shared by offline and live runners."""
        return cls(
            router_observations=int(values["router_observations"]),
            residual_learning_rate=float(values["residual_learning_rate"]),
            residual_radius=float(values["residual_radius"]),
            action_cap=float(values["action_cap"]),
        )


def candidate_spec(device: torch.device | str = "cpu") -> tuple[list[str], torch.Tensor]:
    shear = torch.linspace(0.05, 0.75, 15, device=device)
    attenuation = torch.linspace(0.20, 0.95, 16, device=device)
    families = ["source"] + ["shear"] * len(shear) + ["attenuation"] * len(attenuation)
    return families, torch.cat((torch.zeros(1, device=device), shear, attenuation))


@torch.no_grad()
def candidate_signature(
    model: SpikeWorld,
    state: torch.Tensor,
    action: torch.Tensor,
    task: torch.Tensor,
    endpoint: torch.Tensor,
    families: list[str],
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = len(values)
    candidate_actions = action.expand(count, -1).clone()
    shear_stop = 1 + families.count("shear")
    candidate_actions[1:shear_stop, 1] += (
        values[1:shear_stop] * candidate_actions[1:shear_stop, 0]
    )
    candidate_actions[shear_stop:, :2] *= values[shear_stop:, None]
    candidate_actions.clamp_(-1.0, 1.0)
    predictions, _ = model.predict_standardized(
        state.expand(count, -1), candidate_actions, task.expand(count)
    )
    target = model.target_standardized(state, endpoint)
    errors = predictions - target.expand_as(predictions)
    losses = errors.square().mean(1)
    identity_error = errors[0]
    base = torch.cat(
        (
            losses - losses[0],
            identity_error,
            identity_error.abs(),
            action[0],
            action[0].abs(),
        )
    )
    delta = endpoint[0] - state[0]
    cross = torch.outer(action[0, :2], delta[:6]).flatten()
    return torch.cat((base, state[0], delta, delta.abs(), cross)), losses


class MechanismRouter(nn.Module):
    """Offline-calibrated four-way router; it is frozen during deployment."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 64), nn.ReLU(), nn.Linear(64, len(ROUTES))
        )
        self.register_buffer("feature_mean", torch.zeros(feature_dim))
        self.register_buffer("feature_scale", torch.ones(feature_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_scale
        return self.network(normalized)

    def freeze(self) -> MechanismRouter:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self


def fit_router(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    seed: int,
    updates: int = 700,
    learning_rate: float = 3e-3,
) -> MechanismRouter:
    torch.manual_seed(seed)
    router = MechanismRouter(features.shape[1]).to(features.device)
    router.feature_mean.copy_(features.mean(0))
    router.feature_scale.copy_(features.std(0).clamp_min(1e-5))
    optimizer = torch.optim.AdamW(router.parameters(), lr=learning_rate, weight_decay=1e-4)
    router.train()
    for _ in range(updates):
        loss = F.cross_entropy(router(features), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return router.freeze()


class PersistentFastState(nn.Module):
    """A 24,384-byte persistent state with mechanism-specific residual slots."""

    def __init__(
        self,
        basis_mean: torch.Tensor,
        basis: torch.Tensor,
        families: list[str] | None = None,
        values: torch.Tensor | None = None,
        config: AdaptationConfig | None = None,
    ) -> None:
        super().__init__()
        config = AdaptationConfig() if config is None else config
        if basis_mean.shape != (96,) or basis.shape[1] != 96 or basis.shape[0] < 30:
            raise ValueError("expected a width-96 mean and at least 30 PCA rows")
        if families is None or values is None:
            families, values = candidate_spec(basis.device)
        self.config = config
        self.families = list(families)
        self.register_buffer("basis_mean", basis_mean.detach().clone())
        self.register_buffer("basis", basis[:30].detach().clone())
        self.register_buffer("values", values.detach().clone())
        self.shear_matrix = nn.Parameter(torch.zeros(96, 30))
        self.attenuation_matrix = nn.Parameter(torch.zeros(96, 30))
        self.scores = nn.Parameter(torch.zeros(len(values)), requires_grad=False)
        self.count = nn.Parameter(torch.zeros(()), requires_grad=False)
        padding = FAST_VALUES - 2 * 96 * 30 - len(values) - 1
        if padding < 0:
            raise ValueError("candidate bank exceeds the registered state budget")
        self.padding = nn.Parameter(torch.zeros(padding), requires_grad=False)
        self.route = "source"
        self.write_count = 0
        self._episode_signatures: list[torch.Tensor] = []
        self._pending_losses: list[torch.Tensor] = []

    def start_episode(self) -> None:
        self.route = "source"
        self._episode_signatures.clear()
        self._pending_losses.clear()

    def candidate_index(self) -> int:
        if self.route not in {"shear", "attenuation"}:
            return 0
        indices = [i for i, family in enumerate(self.families) if family == self.route]
        return indices[int(torch.argmin(self.scores[indices]))]

    def transform(self, action: torch.Tensor) -> torch.Tensor:
        mapped = action.clone()
        value = self.values[self.candidate_index()]
        if self.route == "shear":
            mapped[:, 1] = action[:, 1] + value * action[:, 0]
        elif self.route == "attenuation":
            mapped[:, :2] = value * action[:, :2]
        return mapped.clamp(-1.0, 1.0)

    def inverse(self, desired: torch.Tensor) -> torch.Tensor:
        raw = desired.clone()
        value = self.values[self.candidate_index()]
        if self.route == "shear":
            raw[:, 1] = desired[:, 1] - value * desired[:, 0]
        elif self.route == "attenuation":
            raw[:, :2] = desired[:, :2] / value.clamp_min(0.20)
        candidate = desired + (raw - desired).clamp(
            -self.config.action_cap, self.config.action_cap
        )
        invalid = torch.any(candidate.abs() > 1.0, dim=1, keepdim=True)
        return torch.where(invalid, desired, candidate)

    def correction(self, hidden: torch.Tensor) -> torch.Tensor:
        features = (hidden - self.basis_mean) @ self.basis.T
        if self.route == "shear":
            return features @ self.shear_matrix.T + 0.0 * self.padding.sum()
        if self.route == "attenuation":
            return features @ self.attenuation_matrix.T + 0.0 * self.padding.sum()
        return torch.zeros_like(hidden) + 0.0 * self.padding.sum()

    def predict(
        self,
        model: SpikeWorld,
        state: torch.Tensor,
        action: torch.Tensor,
        task: torch.Tensor,
    ) -> torch.Tensor:
        hidden = model.control_hidden(state, self.transform(action), task)
        return model.next_state_head(hidden + self.correction(hidden))

    def observe(
        self,
        model: SpikeWorld,
        router: MechanismRouter,
        state: torch.Tensor,
        action: torch.Tensor,
        task: torch.Tensor,
        endpoint: torch.Tensor,
        evidence_action: torch.Tensor | None = None,
    ) -> dict[str, object]:
        evidence_action = action if evidence_action is None else evidence_action
        signature, losses = candidate_signature(
            model,
            state,
            evidence_action,
            task,
            endpoint,
            self.families,
            self.values,
        )
        self._episode_signatures.append(signature.detach())
        self._pending_losses.append(losses.detach())
        just_routed = False
        with torch.no_grad():
            if len(self._episode_signatures) == self.config.router_observations:
                evidence = torch.stack(self._episode_signatures).mean(0, keepdim=True)
                self.route = ROUTES[int(router(evidence).argmax(1))]
                just_routed = True
            if self.route in {"shear", "attenuation"}:
                if just_routed:
                    self.scores.add_(torch.stack(self._pending_losses).sum(0))
                    self.count.add_(len(self._pending_losses))
                elif len(self._episode_signatures) > self.config.router_observations:
                    self.scores.add_(losses)
                    self.count.add_(1.0)

        wrote = False
        loss_value = None
        if self.route in {"shear", "attenuation"}:
            target = model.target_standardized(state, endpoint).detach()
            prediction = self.predict(model, state, evidence_action, task)
            loss = F.mse_loss(prediction, target)
            matrix = self.shear_matrix if self.route == "shear" else self.attenuation_matrix
            gradient = torch.autograd.grad(loss, matrix)[0]
            norm = gradient.norm().clamp_min(1e-12)
            with torch.no_grad():
                matrix.add_(
                    gradient, alpha=-self.config.residual_learning_rate / float(norm)
                )
                current = matrix.norm().clamp_min(1e-12)
                if float(current) > self.config.residual_radius:
                    matrix.mul_(self.config.residual_radius / float(current))
            wrote = True
            self.write_count += 1
            loss_value = float(loss.detach())
        return {
            "route": self.route,
            "wrote": wrote,
            "candidate_value": float(self.values[self.candidate_index()]),
            "prediction_loss": loss_value,
        }

    def accounting(self) -> dict[str, int]:
        values = sum(parameter.numel() for parameter in self.parameters())
        return {"mutable_values": values, "mutable_bytes": values * 4}


def validate_online_batch(batch: dict[str, torch.Tensor]) -> None:
    required = {"state", "action", "task", "endpoint"}
    forbidden = {"label", "teacher", "reward", "success", "true_gain", "true_shear"}
    missing = required - batch.keys()
    leaked = forbidden & batch.keys()
    if missing or leaked:
        raise ValueError(
            f"invalid online batch: missing={sorted(missing)}, leaked={sorted(leaked)}"
        )
    if batch["state"].shape[-1] != STATE_DIM or batch["action"].shape[-1] != ACTION_DIM:
        raise ValueError("state/action dimensions do not match the registered interface")
