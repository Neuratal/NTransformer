"""Serialization: roundtrip bitwise equality, tying preservation, failure modes."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from neuratal import NeuratalConfig, NeuratalModel, load_pretrained, save_pretrained

V = 262_144


def make_model(**overrides) -> NeuratalModel:
    torch.manual_seed(0)
    return NeuratalModel(NeuratalConfig(**overrides))


class TestRoundTrip:
    def test_state_dict_bitwise_equal(self, tmp_path):
        model = make_model()
        save_pretrained(model, tmp_path)
        loaded = load_pretrained(tmp_path)
        original = model.state_dict()
        restored = loaded.state_dict()
        assert set(original) == set(restored)
        for key, tensor in original.items():
            assert torch.equal(tensor, restored[key]), key

    def test_logits_identical_after_roundtrip(self, tmp_path):
        model = make_model().eval()
        ids = torch.randint(0, V, (1, 8))
        with torch.no_grad():
            ref = model(ids).logits
            save_pretrained(model, tmp_path)
            loaded = load_pretrained(tmp_path).eval()
            out = loaded(ids).logits
        assert torch.equal(ref, out)

    def test_weight_tying_effective_after_load(self, tmp_path):
        model = make_model()
        save_pretrained(model, tmp_path)
        loaded = load_pretrained(tmp_path)
        # exactly one embedding matrix in the loaded model
        keys = list(loaded.state_dict().keys())
        assert keys.count("embed_tokens.token_embedding.weight") == 1
        assert not any("output_projection" in k and "token_embedding" in k for k in keys)
        # parameter identity: the head still shares the embedding object at call time
        assert loaded.embed_tokens.embedding_weight is loaded.embed_tokens.token_embedding.weight

    def test_tied_matrix_serialized_exactly_once(self, tmp_path):
        model = make_model()
        save_pretrained(model, tmp_path)
        with open(tmp_path / "model.safetensors", "rb") as handle:
            header_len = int.from_bytes(handle.read(8), "little")
            header = json.loads(handle.read(header_len))
        embedding_keys = [k for k in header if "token_embedding" in k]
        assert embedding_keys == ["embed_tokens.token_embedding.weight"]
        assert "output_projection.out_proj.weight" in header

    def test_checkpoint_config_is_resolved_and_self_describing(self, tmp_path):
        model = make_model()
        save_pretrained(model, tmp_path)
        payload = json.loads((tmp_path / "config.json").read_text())
        assert payload["format_version"] == 1
        assert payload["architectures"] == ["NeuratalModel"]
        cfg = payload["config"]
        assert cfg["vocab_size"] == V
        assert cfg["embedding_size"] == 32
        assert cfg["cache_capacity"] == cfg["max_position_embeddings"] == 2048
        assert cfg["tie_word_embeddings"] is True

    def test_config_roundtrip_preserves_values(self, tmp_path):
        model = make_model()
        save_pretrained(model, tmp_path)
        loaded = load_pretrained(tmp_path)
        assert loaded.config.to_dict() == model.config.to_dict()


class TestFailureModes:
    def test_missing_weights_file(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        with pytest.raises(ValueError, match="model.safetensors"):
            load_pretrained(tmp_path)

    def test_missing_config_file(self, tmp_path):
        (tmp_path / "model.safetensors").write_bytes(b"")
        with pytest.raises(ValueError, match="config.json"):
            load_pretrained(tmp_path)

    def test_non_directory_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            load_pretrained(tmp_path / "nope")

    def test_truncated_safetensors_rejected(self, tmp_path):
        model = make_model()
        save_pretrained(model, tmp_path)
        path = tmp_path / "model.safetensors"
        data = path.read_bytes()
        path.write_bytes(data[: len(data) // 2])
        with pytest.raises(ValueError, match="safetensors"):
            load_pretrained(tmp_path)

    def test_missing_key_rejected(self, tmp_path):
        model = make_model()
        save_pretrained(model, tmp_path)
        sd = {k: v for k, v in load_file(str(tmp_path / "model.safetensors")).items()
              if k != "layers.0.mlp.gate_up_proj.weight"}
        save_file(sd, str(tmp_path / "model.safetensors"))
        with pytest.raises(ValueError, match="missing keys"):
            load_pretrained(tmp_path)

    def test_unexpected_key_rejected(self, tmp_path):
        model = make_model()
        save_pretrained(model, tmp_path)
        sd = load_file(str(tmp_path / "model.safetensors"))
        sd["layers.0.mlp.gate_up_proj.bias"] = torch.zeros(1280)
        save_file(sd, str(tmp_path / "model.safetensors"))
        with pytest.raises(ValueError, match="unexpected keys"):
            load_pretrained(tmp_path)

    def test_dtype_mismatch_between_config_and_file_rejected(self, tmp_path):
        model = make_model()
        save_pretrained(model, tmp_path)
        payload = json.loads((tmp_path / "config.json").read_text())
        payload["config"]["dtype"] = "bfloat16"  # file stays fp32
        (tmp_path / "config.json").write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="dtype mismatch"):
            load_pretrained(tmp_path)

    def test_shape_mismatch_rejected(self, tmp_path):
        model = make_model()
        save_pretrained(model, tmp_path)
        sd = load_file(str(tmp_path / "model.safetensors"))
        sd["layers.0.mlp.gate_up_proj.weight"] = torch.zeros(1281, 256)
        save_file(sd, str(tmp_path / "model.safetensors"))
        with pytest.raises(ValueError, match="shape mismatch"):
            load_pretrained(tmp_path)

    def test_unknown_config_key_rejected(self, tmp_path):
        model = make_model()
        save_pretrained(model, tmp_path)
        payload = json.loads((tmp_path / "config.json").read_text())
        payload["config"]["sneaky_flag"] = 1
        (tmp_path / "config.json").write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="unknown config keys"):
            load_pretrained(tmp_path)

    def test_unsupported_format_version_rejected(self, tmp_path):
        model = make_model()
        save_pretrained(model, tmp_path)
        payload = json.loads((tmp_path / "config.json").read_text())
        payload["format_version"] = 99
        (tmp_path / "config.json").write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="format_version"):
            load_pretrained(tmp_path)

    def test_wrong_architecture_rejected(self, tmp_path):
        model = make_model()
        save_pretrained(model, tmp_path)
        payload = json.loads((tmp_path / "config.json").read_text())
        payload["architectures"] = ["SomeOtherModel"]
        (tmp_path / "config.json").write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="architecture"):
            load_pretrained(tmp_path)

    def test_invalid_config_json_rejected(self, tmp_path):
        (tmp_path / "config.json").write_text("{not json")
        (tmp_path / "model.safetensors").write_bytes(b"")
        with pytest.raises(ValueError):
            load_pretrained(tmp_path)


    def test_bf16_model_roundtrip_records_actual_dtype(self, tmp_path):
        model = make_model().to(torch.bfloat16)
        with pytest.warns(UserWarning, match="dtype"):
            save_pretrained(model, tmp_path)
        payload = json.loads((tmp_path / "config.json").read_text())
        assert payload["config"]["dtype"] == "bfloat16"
        loaded = load_pretrained(tmp_path)
        for key, tensor in model.state_dict().items():
            restored = loaded.state_dict()[key]
            assert torch.equal(tensor, restored), key


class TestCudaRoundTrip:
    def test_cuda_save_load_cpu(self, tmp_path):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        model = make_model().to("cuda")
        save_pretrained(model, tmp_path)
        loaded = load_pretrained(tmp_path, map_location="cpu")
        for key, tensor in model.state_dict().items():
            assert torch.equal(tensor.cpu(), loaded.state_dict()[key]), key
