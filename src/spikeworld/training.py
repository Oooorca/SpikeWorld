"""Losses and short training routines shared by the paper experiments."""

from __future__ import annotations

import random
from collections.abc import Iterator

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import action_batch, persistence_denominators, sensory_loaders
from .model import SpikeWorld

STREAMS = ("audio_shd", "audio_ssc", "text", "image", "video")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def multi_positive_contrastive_loss(
    image: torch.Tensor, text: torch.Tensor, labels: torch.Tensor, temperature: float = 0.12
) -> torch.Tensor:
    image = F.normalize(image, dim=-1)
    text = F.normalize(text, dim=-1)
    scores = image @ text.T / temperature
    positives = labels[:, None].eq(labels[None, :])
    minimum = torch.finfo(scores.dtype).min

    def direction(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        numerator = torch.logsumexp(logits.masked_fill(~mask, minimum), dim=1)
        return -(numerator - torch.logsumexp(logits, dim=1)).mean()

    return 0.5 * (direction(scores, positives) + direction(scores.T, positives.T))


class _Cycle:
    def __init__(self, loader):
        self.loader = loader
        self.iterator: Iterator = iter(loader)

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


def _move(batch, device: torch.device):
    return tuple(value.to(device) for value in batch)


def multimodal_loss(
    model: SpikeWorld,
    cycles: dict[str, _Cycle],
    denominator: dict[str, float],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    predictive = model.anchored.predictive
    backbone = predictive.backbone
    shd, shd_y = _move(cycles["audio_shd"].next(), device)
    ssc, ssc_y = _move(cycles["audio_ssc"].next(), device)
    image, text, retinal_y = _move(cycles["retinal"].next(), device)
    video, video_y, _teacher = _move(cycles["video"].next(), device)

    def predict(values: torch.Tensor, stream: str):
        output, hidden, _ = backbone.predict_next(values, stream)
        loss = F.mse_loss(output, values[:, 1:]) / max(denominator[stream], 1e-8)
        return loss, hidden

    shd_loss, shd_h = predict(shd, "audio_shd")
    ssc_loss, ssc_h = predict(ssc, "audio_ssc")
    image_loss, image_h = predict(image, "image")
    text_loss, text_h = predict(text, "text")
    video_loss, video_h = predict(video, "video")
    next_loss = 0.25 * (0.5 * (shd_loss + ssc_loss) + image_loss + text_loss + video_loss)
    semantic_loss = torch.stack(
        (
            F.cross_entropy(
                model.anchored.coupled_semantic_logits(shd_h, "audio_shd"), shd_y
            ),
            F.cross_entropy(
                model.anchored.coupled_semantic_logits(ssc_h, "audio_ssc"), ssc_y
            ),
            F.cross_entropy(
                model.anchored.coupled_semantic_logits(image_h, "image"), retinal_y
            ),
            F.cross_entropy(
                model.anchored.coupled_semantic_logits(text_h, "text"), retinal_y
            ),
            F.cross_entropy(
                model.anchored.coupled_semantic_logits(video_h, "video"), video_y
            ),
        )
    ).mean()
    binding = multi_positive_contrastive_loss(
        predictive.binding_representation(image_h),
        predictive.binding_representation(text_h),
        retinal_y,
    )
    return next_loss + semantic_loss + 0.25 * binding, {
        "next": next_loss,
        "semantic": semantic_loss,
        "binding": binding,
    }


def action_loss(
    model: SpikeWorld,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    state, action, task, endpoint = action_batch(data, indices, device)
    prediction, _ = model.predict_standardized(state, action, task)
    return F.mse_loss(prediction, model.target_standardized(state, endpoint))


def configure_joint_trainable(model: SpikeWorld) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = (
        model.anchored.predictive,
        model.anchored.coupled_semantic_norm,
        model.anchored.coupled_semantic_heads,
        model.state_stem,
        model.action_stem,
        model.task_embedding,
        model.token_type,
        model.next_state_head,
    )
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.control_modality.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def train_joint(
    model: SpikeWorld,
    sensory: dict[str, object],
    manifest: dict[str, object],
    action: dict[str, np.ndarray],
    *,
    seed: int,
    updates: int = 200,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    arm: str = "joint",
    device: torch.device,
) -> list[dict[str, float]]:
    if arm not in {"joint", "action_only"}:
        raise ValueError(arm)
    set_seed(seed)
    trainable = configure_joint_trainable(model)
    loaders = sensory_loaders(sensory, batch_size, seed + 434)
    cycles = {
        name: _Cycle(loaders[f"train_{name}"])
        for name in ("audio_shd", "audio_ssc", "retinal", "video")
    }
    denominator = persistence_denominators(manifest, "train")
    rng = np.random.default_rng(seed + 435)
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-4)
    curve = []
    for update in range(1, updates + 1):
        indices = rng.integers(0, len(action["state"]), batch_size)
        model.train()
        action_term = action_loss(model, action, indices, device)
        if arm == "joint":
            _, parts = multimodal_loss(model, cycles, denominator, device)
            loss = (
                4.0 * action_term
                + parts["next"]
                + parts["semantic"]
                + 0.25 * parts["binding"]
            )
        else:
            parts = {
                name: action_term.new_zeros(()) for name in ("next", "semantic", "binding")
            }
            loss = 4.0 * action_term
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = float(nn.utils.clip_grad_norm_(trainable, 1.0))
        optimizer.step()
        if update == 1 or update == updates or update % max(1, updates // 10) == 0:
            curve.append(
                {
                    "update": update,
                    "loss": float(loss.detach()),
                    "action": float(action_term.detach()),
                    **{name: float(value.detach()) for name, value in parts.items()},
                    "gradient_before_clip": gradient,
                }
            )
    return curve


@torch.no_grad()
def pca_basis(
    model: SpikeWorld,
    action_data: dict[str, np.ndarray],
    device: torch.device,
    rank: int = 63,
    batch_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = []
    for start in range(0, len(action_data["state"]), batch_size):
        index = np.arange(start, min(start + batch_size, len(action_data["state"])))
        state, action, task, _ = action_batch(action_data, index, device)
        hidden.append(model.control_hidden(state, action, task).cpu())
    rows = torch.cat(hidden)
    mean = rows.mean(0)
    _, _, vectors = np.linalg.svd(
        (rows - mean).numpy().astype(np.float64), full_matrices=False
    )
    return mean.float(), torch.from_numpy(vectors[:rank].astype(np.float32))


@torch.no_grad()
def evaluate_model(
    model: SpikeWorld,
    sensory: dict[str, object],
    manifest: dict[str, object],
    action: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int = 128,
) -> dict[str, float]:
    model.eval()
    denominator = persistence_denominators(manifest, "val")
    next_losses, accuracies = [], []
    representations: dict[str, list[torch.Tensor]] = {"image": [], "text": []}
    retinal_labels: list[torch.Tensor] = []
    for stream in STREAMS:
        source = "retinal" if stream in {"image", "text"} else stream
        data = sensory[source]["val"]
        values = data[0 if stream != "text" else 1]
        labels = data[1] if source != "retinal" else data[2]
        squared, elements, correct, examples = 0.0, 0, 0, 0
        for start in range(0, len(values), batch_size):
            local = values[start : start + batch_size].to(device)
            local_labels = labels[start : start + batch_size].to(device)
            prediction, hidden, _ = model.backbone.predict_next(local, stream)
            squared += float(F.mse_loss(prediction, local[:, 1:], reduction="sum"))
            elements += prediction.numel()
            logits = model.anchored.coupled_semantic_logits(hidden, stream)
            correct += int((logits.argmax(1) == local_labels).sum())
            examples += len(local)
            if stream in representations:
                representations[stream].append(
                    model.anchored.predictive.binding_representation(hidden).cpu()
                )
                if stream == "image":
                    retinal_labels.append(local_labels.cpu())
        next_losses.append((squared / elements) / max(denominator[stream], 1e-8))
        accuracies.append(correct / examples)
    image = F.normalize(torch.cat(representations["image"]), dim=-1)
    text = F.normalize(torch.cat(representations["text"]), dim=-1)
    labels = torch.cat(retinal_labels)
    similarity = image @ text.T
    binding = 0.5 * (
        (labels[similarity.argmax(1)] == labels).float().mean()
        + (labels[similarity.argmax(0)] == labels).float().mean()
    )
    action_squared, action_elements = 0.0, 0
    for start in range(0, len(action["state"]), batch_size):
        indices = np.arange(start, min(start + batch_size, len(action["state"])))
        state, command, task, endpoint = action_batch(action, indices, device)
        prediction, _ = model.predict_standardized(state, command, task)
        target = model.target_standardized(state, endpoint)
        action_squared += float(F.mse_loss(prediction, target, reduction="sum"))
        action_elements += target.numel()
    return {
        "action_mse": action_squared / action_elements,
        "mean_normalized_next_slice": float(np.mean(next_losses)),
        "mean_semantic_accuracy": float(np.mean(accuracies)),
        "binding_mean_class_r1": float(binding),
    }


def train_control_path(
    model: SpikeWorld,
    action: dict[str, np.ndarray],
    *,
    seed: int,
    updates: int = 800,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: torch.device,
) -> None:
    set_seed(seed)
    modules = (
        model.state_stem,
        model.action_stem,
        model.task_embedding,
        model.token_type,
        model.next_state_head,
    )
    parameters = [model.control_modality]
    parameters.extend(p for module in modules for p in module.parameters())
    for parameter in model.anchored.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    for _ in range(updates):
        indices = rng.integers(0, len(action["state"]), batch_size)
        loss = action_loss(model, action, indices, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
