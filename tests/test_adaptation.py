import pytest
import torch

from spikeworld.adaptation import FAST_BYTES, PersistentFastState, validate_online_batch


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
