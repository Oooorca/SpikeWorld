"""Closed-Loop Control experiment under hidden attenuation."""

from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import torch
from torch.nn import functional as F

from ..adaptation import (
    ROUTES,
    AdaptationConfig,
    MechanismRouter,
    PersistentFastState,
    candidate_signature,
    candidate_spec,
)
from ..model import TASKS, SpikeWorld, tensor_hash
from .common import bootstrap, device_from, load_full_checkpoint, read_config, write_json


def safe_inverse(desired: torch.Tensor, gain: float, cap: float = 0.05) -> torch.Tensor:
    raw = desired.clone()
    raw[:, :2] = desired[:, :2] / max(gain, 0.20)
    candidate = desired + (raw - desired).clamp(-cap, cap)
    invalid = torch.any(candidate.abs() > 1.0, dim=1, keepdim=True)
    return torch.where(invalid, desired, candidate)


@torch.no_grad()
def local_gain_observation(
    model: SpikeWorld,
    state: torch.Tensor,
    action: torch.Tensor,
    task: torch.Tensor,
    endpoint: torch.Tensor,
    finite_difference: float = 0.05,
) -> tuple[float, float, float]:
    low, high = action.clone(), action.clone()
    low[:, :2] *= 1.0 - finite_difference
    high[:, :2] *= 1.0 + finite_difference
    low_prediction, _ = model.predict_standardized(state, low, task)
    high_prediction, _ = model.predict_standardized(state, high, task)
    base, _ = model.predict_standardized(state, action, task)
    derivative = (high_prediction - low_prediction) / (2.0 * finite_difference)
    residual = model.target_standardized(state, endpoint) - base
    numerator = float(torch.sum(derivative * residual))
    denominator = float(torch.sum(derivative.square())) + 1e-8
    return float(np.clip(1.0 + numerator / denominator, 0.20, 1.20)), numerator, denominator


class RLSController:
    def __init__(self, model: SpikeWorld):
        self.model = model
        self.numerator = 0.0
        self.denominator = 1e-3

    def start_episode(self) -> None:
        pass

    def gain(self) -> float:
        return float(np.clip(1.0 + self.numerator / self.denominator, 0.20, 1.20))

    def command(self, desired, _state, _task):
        return safe_inverse(desired, self.gain())

    def observe(self, state, command, task, endpoint):
        _, numerator, denominator = local_gain_observation(
            self.model, state, command, task, endpoint
        )
        self.numerator += numerator
        self.denominator += denominator
        return {"route": "attenuation", "wrote": True, "gain": self.gain()}


class ReplayController:
    def __init__(self, model: SpikeWorld, capacity: int = 141):
        self.model = model
        self.capacity = capacity
        self.features: list[torch.Tensor] = []
        self.gains: list[float] = []

    def start_episode(self) -> None:
        pass

    @staticmethod
    def feature(state: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(task, num_classes=len(TASKS)).float()
        return torch.cat((state, one_hot), dim=1)[0].detach().cpu()

    def gain(self, state, task) -> float:
        if not self.features:
            return 1.0
        query = self.feature(state, task)
        index = int(torch.argmin((torch.stack(self.features) - query).square().sum(1)))
        return self.gains[index]

    def command(self, desired, state, task):
        return safe_inverse(desired, self.gain(state, task))

    def observe(self, state, command, task, endpoint):
        gain, _, _ = local_gain_observation(self.model, state, command, task, endpoint)
        if len(self.features) >= self.capacity:
            self.features.pop(0)
            self.gains.pop(0)
        self.features.append(self.feature(state, task))
        self.gains.append(gain)
        return {"route": "attenuation", "wrote": True, "gain": gain}


class FastController:
    def __init__(self, model, router, memory):
        self.model, self.router, self.memory = model, router, memory

    def start_episode(self) -> None:
        self.memory.start_episode()

    def command(self, desired, _state, _task):
        return self.memory.inverse(desired)

    def observe(self, state, command, task, endpoint):
        return self.memory.observe(self.model, self.router, state, command, task, endpoint)


class EvidenceOnlyFastState(PersistentFastState):
    def observe(self, model, router, state, action, task, endpoint, evidence_action=None):
        evidence_action = action if evidence_action is None else evidence_action
        signature, losses = candidate_signature(
            model, state, evidence_action, task, endpoint, self.families, self.values
        )
        self._episode_signatures.append(signature.detach())
        self._pending_losses.append(losses.detach())
        with torch.no_grad():
            just_routed = False
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
        return {
            "route": self.route,
            "wrote": False,
            "candidate_value": float(self.values[self.candidate_index()]),
        }


class FullTuningController:
    def __init__(
        self,
        source: SpikeWorld,
        router: MechanismRouter,
        memory: EvidenceOnlyFastState,
        learning_rate: float = 1e-5,
    ):
        self.model = copy.deepcopy(source)
        self.router = router
        self.memory = memory
        for name, parameter in self.model.named_parameters():
            stable = "semantic" in name or "stable_semantic" in name
            parameter.requires_grad_(not stable)
        self.trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.trainable, lr=learning_rate)

    def start_episode(self) -> None:
        self.memory.start_episode()

    def command(self, desired, _state, _task):
        return self.memory.inverse(desired)

    def observe(self, state, command, task, endpoint):
        evidence = self.memory.observe(
            self.model, self.router, state, command, task, endpoint
        )
        target = self.model.target_standardized(state, endpoint).detach()
        prediction, _ = self.model.predict_standardized(state, command, task)
        loss = F.mse_loss(prediction, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.trainable, 1.0)
        self.optimizer.step()
        return {
            "route": evidence["route"],
            "wrote": True,
            "gain": evidence["candidate_value"],
        }


def controllers(model, bundle, device, config: AdaptationConfig):
    families, values = candidate_spec(device)
    router = MechanismRouter(int(bundle["router_feature_dim"])).to(device)
    router.load_state_dict(bundle["router_state"], strict=True)
    router.freeze()
    memory = PersistentFastState(
        bundle["basis_mean"].to(device),
        bundle["basis"].to(device),
        families,
        values,
        config,
    ).to(device)
    evidence_memory = EvidenceOnlyFastState(
        bundle["basis_mean"].to(device),
        bundle["basis"].to(device),
        families,
        values,
        config,
    ).to(device)
    return {
        "safe_fast": FastController(model, router, memory),
        "full_tuning": FullTuningController(model, router, evidence_memory),
        "rls": RLSController(model),
        "replay": ReplayController(model),
    }


def execute_macro(env, command, remaining, gain, horizon):
    executed = command.copy()
    executed[:2] *= gain
    executed = np.clip(executed, -1.0, 1.0).astype(np.float32)
    endpoint, reward, success, steps = None, 0.0, 0.0, 0
    terminated = truncated = False
    for _ in range(min(horizon, remaining)):
        endpoint, local_reward, terminated, truncated, info = env.step(executed)
        reward += float(local_reward)
        success = max(success, float(info["success"]))
        steps += 1
        if terminated or truncated:
            break
    return (
        np.asarray(endpoint, dtype=np.float32),
        reward,
        success,
        steps,
        terminated,
        truncated,
    )


def run_episode(task, seed, arm, controller, cfg, device):
    try:
        import gymnasium as gym
        import metaworld  # noqa: F401
        from metaworld.policies import ENV_POLICY_MAP
    except ImportError as error:
        raise RuntimeError("install the 'control' extra for closed-loop evaluation") from error
    env = gym.make("Meta-World/MT1", env_name=task, seed=seed, disable_env_checker=True)
    policy = ENV_POLICY_MAP[task]()
    rng = np.random.default_rng(seed + int(cfg["action_noise_offset"]))
    observation, _ = env.reset(seed=seed)
    observation = np.asarray(observation, dtype=np.float32)
    task_tensor = torch.tensor([TASKS.index(task)], dtype=torch.long, device=device)
    if controller is not None:
        controller.start_episode()
    reward_total, success, elapsed, decisions, writes = 0.0, 0.0, 0, 0, 0
    update_times = []
    while elapsed < int(cfg["max_steps"]):
        state = torch.from_numpy(observation[None]).to(device)
        desired_np = np.clip(
            policy.get_action(observation.copy())
            + rng.normal(0.0, float(cfg["action_noise_std"]), 4),
            -1.0,
            1.0,
        ).astype(np.float32)
        desired = torch.from_numpy(desired_np[None]).to(device)
        if arm == "frozen":
            command = desired
        elif arm == "oracle":
            command = desired.clone()
            command[:, :2] = desired[:, :2] / float(cfg["hidden_attenuation"])
            command.clamp_(-1.0, 1.0)
        else:
            command = controller.command(desired, state, task_tensor)
        endpoint, reward, local_success, steps, terminated, truncated = execute_macro(
            env,
            command.detach().cpu().numpy()[0],
            int(cfg["max_steps"]) - elapsed,
            float(cfg["hidden_attenuation"]),
            int(cfg["macro_horizon"]),
        )
        if controller is not None:
            started = time.perf_counter()
            report = controller.observe(
                state, command, task_tensor, torch.from_numpy(endpoint[None]).to(device)
            )
            update_times.append(time.perf_counter() - started)
            writes += int(report["wrote"])
        observation = endpoint
        reward_total += reward
        success = max(success, local_success)
        elapsed += steps
        decisions += 1
        if terminated or truncated:
            break
    env.close()
    return {
        "task": task,
        "seed": seed,
        "arm": arm,
        "cumulative_reward": reward_total,
        "success": success,
        "decisions": decisions,
        "writes": writes,
        "mean_update_seconds": float(np.mean(update_times)) if update_times else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/control.toml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = read_config(args.config)
    paths, run = cfg["paths"], cfg["run"]
    device = device_from(run.get("device", "cpu"))
    adaptation_config = AdaptationConfig.from_mapping(run)
    tasks = run["tasks"][:1] if args.smoke else run["tasks"]
    replicas = 1 if args.smoke else int(run["replicas"])
    checkpoints = (
        paths["model_checkpoints"][:1] if args.smoke else paths["model_checkpoints"]
    )
    bundles = paths["deployment_bundles"][:1] if args.smoke else paths["deployment_bundles"]
    rows, integrity = [], {}
    for model_index, (checkpoint, bundle_path) in enumerate(
        zip(checkpoints, bundles, strict=True)
    ):
        model = load_full_checkpoint(checkpoint, device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        before = tensor_hash(model)
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        for task_index, task in enumerate(tasks):
            local_controllers = controllers(model, bundle, device, adaptation_config)
            for replica in range(replicas):
                seed = int(run["base_seed"]) + task_index * 1_000 + replica
                for arm in run["arms"]:
                    rows.append(
                        {
                            "model_index": model_index,
                            "replica": replica,
                            **run_episode(
                                task, seed, arm, local_controllers.get(arm), run, device
                            ),
                        }
                    )
        integrity[str(model_index)] = {
            "frozen_model_hash_unchanged": tensor_hash(model) == before
        }
    arms = {}
    frozen = {
        (r["model_index"], r["task"], r["replica"]): r for r in rows if r["arm"] == "frozen"
    }
    for arm in run["arms"]:
        local = [row for row in rows if row["arm"] == arm]
        arms[arm] = {
            "reward": float(np.mean([row["cumulative_reward"] for row in local])),
            "success": float(np.mean([row["success"] for row in local])),
        }
        if arm != "frozen":
            delta = [
                row["cumulative_reward"]
                - frozen[(row["model_index"], row["task"], row["replica"])][
                    "cumulative_reward"
                ]
                for row in local
            ]
            arms[arm]["reward_delta"] = bootstrap(delta, 20_000, 43110 + len(arms))
    write_json(
        paths["output"],
        {"experiment": "live_control", "arms": arms, "integrity": integrity, "rows": rows},
    )


if __name__ == "__main__":
    main()
