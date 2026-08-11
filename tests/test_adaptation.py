import pytest
import torch

from spikeworld.adaptation import (
    FAST_BYTES,
    AdaptationConfig,
    MechanismRouter,
    PersistentFastState,
    candidate_signature,
    validate_online_batch,
)
from spikeworld.model import ModelConfig, SpikeWorld, tensor_hash


def test_exact_fast_state_budget_and_action_cap():
    memory = PersistentFastState(torch.zeros(96), torch.eye(96)[:63])
    assert memory.accounting() == {"mutable_values": 6096, "mutable_bytes": FAST_BYTES}
    memory.route = "attenuation"
    memory.scores.fill_(1.0)
    attenuation = [i for i, family in enumerate(memory.families) if family == "attenuation"]
    memory.scores[attenuation[0]] = 0.0
    desired = torch.tensor([[0.5, -0.5, 0.0, 0.0]])
    command = memory.inverse(desired)
    assert torch.all((command - desired).abs() <= 0.050001)
    assert torch.all(command.abs() <= 1.0)


def test_online_interface_rejects_privileged_fields():
    valid = {
        "state": torch.zeros(1, 39),
        "action": torch.zeros(1, 4),
        "task": torch.zeros(1, dtype=torch.long),
        "endpoint": torch.zeros(1, 39),
    }
    validate_online_batch(valid)
    with pytest.raises(ValueError):
        validate_online_batch({**valid, "reward": torch.ones(1)})


def test_mapping_controls_the_fast_state():
    values = {
        "router_observations": 7,
        "residual_learning_rate": 0.004,
        "residual_radius": 0.11,
        "action_cap": 0.03,
    }
    config = AdaptationConfig.from_mapping(values)
    memory = PersistentFastState(torch.zeros(96), torch.eye(96)[:30], config=config)
    assert memory.config == config


def test_same_input_semantic_logits_are_bitwise_invariant_to_fast_state():
    torch.manual_seed(19)
    model = SpikeWorld(ModelConfig()).eval()
    memory = PersistentFastState(torch.zeros(96), torch.eye(96)[:30])
    sensory_input = torch.randn(2, 16, 40)
    with torch.no_grad():
        before = model.anchored.semantic_logits(sensory_input, "image").clone()
    model_hash = tensor_hash(model)

    memory.route = "attenuation"
    with torch.no_grad():
        memory.scores.copy_(
            torch.arange(memory.scores.numel(), dtype=memory.scores.dtype)
        )
        memory.shear_matrix.normal_()
        memory.attenuation_matrix.normal_()
        memory.count.fill_(9.0)

    with torch.no_grad():
        after = model.anchored.semantic_logits(sensory_input, "image")
    assert torch.equal(before, after)
    assert tensor_hash(model) == model_hash
    assert tensor_hash({"semantic_logits": before}) == tensor_hash(
        {"semantic_logits": after}
    )


def test_four_observations_first_affect_decision_five():
    torch.manual_seed(23)
    model = SpikeWorld(ModelConfig()).eval()
    memory = PersistentFastState(torch.zeros(96), torch.eye(96)[:30])
    state = torch.zeros(1, 39)
    desired = torch.tensor([[0.4, -0.3, 0.0, 0.0]])
    task = torch.zeros(1, dtype=torch.long)
    endpoint = state + 0.03
    signature, _ = candidate_signature(
        model,
        state,
        desired,
        task,
        endpoint,
        memory.families,
        memory.values,
    )
    router = MechanismRouter(signature.numel()).eval()
    with torch.no_grad():
        for parameter in router.parameters():
            parameter.zero_()
        router.network[-1].bias[2] = 10.0  # registered attenuation route
    router.freeze()

    commands = []
    for _ in range(4):
        commands.append(memory.inverse(desired).clone())
        memory.observe(model, router, state, desired, task, endpoint)
    commands.append(memory.inverse(desired).clone())

    for command in commands[:4]:
        assert torch.equal(command, desired)
    assert not torch.equal(commands[4], desired)
    assert torch.all((commands[4] - desired).abs() <= 0.050001)
