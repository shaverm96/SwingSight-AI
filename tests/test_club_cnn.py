from __future__ import annotations

import pytest

import swingsight.club_cnn as club_cnn


@pytest.mark.skipif(club_cnn.torch is None, reason="PyTorch is optional")
def test_resnet50_checkpoint_uses_the_matching_loader(monkeypatch):
    class FakeModel:
        def __init__(self) -> None:
            self.loaded_state = None
            self.device = None
            self.evaluated = False

        def load_state_dict(self, state_dict, strict=True):
            assert strict is True
            self.loaded_state = state_dict

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.evaluated = True
            return self

    model = FakeModel()
    device = object()
    checkpoint = {
        "format": club_cnn.CHECKPOINT_FORMAT,
        "task": "club_type_5way",
        "architecture": "resnet50_v1",
        "classifier_dropout": 0.30,
        "class_names": ["driver", "wood", "hybrid", "iron", "wedge"],
        "input_size": 288,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "state_dict": {"fc.1.weight": object()},
    }

    def build_resnet(num_classes: int, *, classifier_dropout: float):
        assert num_classes == 5
        assert classifier_dropout == pytest.approx(0.30)
        return model

    monkeypatch.setattr(club_cnn, "_torch_load", lambda path: checkpoint)
    monkeypatch.setattr(club_cnn, "_torch_device", lambda: device)
    monkeypatch.setattr(club_cnn, "build_resnet50", build_resnet)

    loaded = club_cnn._load_checkpoint.__wrapped__("resnet50.pt", "club_type_5way")

    assert loaded.model is model
    assert loaded.class_names == ("driver", "wood", "hybrid", "iron", "wedge")
    assert loaded.input_size == 288
    assert model.loaded_state == checkpoint["state_dict"]
    assert model.device is device
    assert model.evaluated is True
