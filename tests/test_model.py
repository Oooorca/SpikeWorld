import torch

from spikeworld.model import ModelConfig, SpikeWorld, local_qk_pairs


def test_registered_model_size_and_sparse_qk_count():
    model = SpikeWorld(ModelConfig())
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_451_388
    assert local_qk_pairs(16, 4) == 58


def test_multimodal_and_control_shapes():
    model = SpikeWorld(ModelConfig()).eval()
    with torch.no_grad():
        prediction, hidden, stats = model.backbone.predict_next(
            torch.zeros(2, 16, 40), "image", collect_stats=True
        )
        control, control_hidden = model.predict_standardized(
            torch.zeros(2, 39), torch.zeros(2, 4), torch.zeros(2, dtype=torch.long)
        )
    assert prediction.shape == (2, 15, 40)
    assert hidden.shape == (2, 16, 96)
    assert stats["attention_density"] == 58 / 256
    assert control.shape == (2, 39)
    assert control_hidden.shape == (2, 96)
