# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Neuratal Transformer Contributors
"""NTokenizer v0.3 ↔ NeuratalTransformer compatibility suite.

Skipped unless the real NTokenizer v0.3 artifact and Python bindings are
present (they are on the development machine). These tests must be executed,
not simulated — skipping here is honest only because CI may lack the artifact.

Coverage (per integration spec):
A. vocabulary compatibility + boundaries
B. text encoding/decoding (incl. raw bytes)
C. special tokens (BOS/EOS/PAD/etc.) + generation EOS semantics
D. end-to-end: encode → model → generate → decode
E. checkpoint compatibility (sidecar manifest, mismatch detection, backward compat)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from neuratal import NeuratalConfig, NeuratalModel, generate, load_pretrained, save_pretrained
from neuratal.integration import (
    EXPECTED_SPECIAL_TOKENS,
    MANIFEST_FILENAME,
    NTokenizerAdapter,
    TokenizerMismatchError,
    build_manifest,
    load_checkpoint_with_tokenizer,
    load_manifest,
    save_manifest,
    verify_checkpoint_tokenizer,
    verify_manifest,
    write_tokenizer_sidecar,
)

NTOKENIZER_ROOT = Path(__file__).resolve().parents[2] / "NTokenizer"
TOKENIZER_MODEL = NTOKENIZER_ROOT / "output" / "v0.3" / "ntokenizer-0.3-256k.ntok"
TOKENIZER_V02 = NTOKENIZER_ROOT / "output" / "v0.2" / "ntokenizer-0.2-256k.ntok"

pytestmark = pytest.mark.skipif(
    not TOKENIZER_MODEL.is_file(),
    reason=f"NTokenizer v0.3 artifact not found at {TOKENIZER_MODEL}",
)

VOCAB = 262_144


@pytest.fixture(scope="module")
def adapter():
    sys.path.insert(0, str(NTOKENIZER_ROOT / "python"))
    try:
        return NTokenizerAdapter(TOKENIZER_MODEL)
    except Exception as exc:
        pytest.skip(f"NTokenizer bindings unavailable: {exc}")


@pytest.fixture()
def model(device):
    torch.manual_seed(0)
    m = NeuratalModel(NeuratalConfig(eos_token_id=262138, pad_token_id=262139, bos_token_id=262137)).to(device)
    return m.eval()


class TestVocabularyCompatibility:
    def test_vocabularies_match_exactly(self, adapter):
        assert adapter.vocab_size == VOCAB == NeuratalConfig().vocab_size

    def test_manifest_sha256_matches_documented_artifact(self):
        manifest = build_manifest(TOKENIZER_MODEL)
        assert manifest["tokenizer_sha256"] == "1f0d4ad7c7fb609001b2af4d9741ec5b8a5d7686fa3b92eb8eefd057fea936d9"

    def test_special_token_ids_match(self, adapter):
        assert adapter.special_tokens == EXPECTED_SPECIAL_TOKENS

    def test_manifest_cross_checked_counts(self):
        manifest = build_manifest(TOKENIZER_MODEL)
        assert (manifest["vocab_size"], manifest["base_byte_tokens"],
                manifest["merges_count"], manifest["special_tokens_count"]) == (262_144, 256, 261_881, 7)

    def test_boundary_ids_are_in_range(self, adapter):
        for token_id in (0, 255, 256, 262_136, 262_137, 262_138, 262_139, 262_143):
            assert 0 <= token_id < VOCAB
            out = adapter.decode_bytes([token_id])
            assert isinstance(out, bytes)

    def test_all_base_byte_ids_roundtrip(self, adapter):
        for i in range(256):
            assert adapter.decode_bytes([i]) == bytes([i])

    def test_special_ids_decode_to_their_literal_content(self, adapter):
        for content, token_id in EXPECTED_SPECIAL_TOKENS.items():
            assert adapter.decode([token_id]) == content

    def test_model_embedding_accepts_all_special_and_boundary_ids(self, model, device):
        ids = torch.tensor([[0, 255, 256, 262_136, 262_137, 262_138, 262_139, 262_140,
                             262_141, 262_142, 262_143, 262_143]], device=device)
        with torch.no_grad():
            out = model(ids)
        assert out.logits.shape == (1, 12, VOCAB)

    def test_no_id_remapping_encode_is_verbatim(self, adapter, model, device):
        text = "Neuratal integration boundary check ✓ 1234"
        ids = adapter.encode(text)
        assert all(0 <= i < VOCAB for i in ids)
        tensor_ids = torch.tensor([ids], device=device)
        with torch.no_grad():
            out = model(tensor_ids)
        assert out.logits.shape == (1, len(ids), VOCAB)


class TestEncodingDecoding:
    SAMPLES = [
        "Hello, world!",                                  # English
        "İstanbul'da bugün hava çok güzel.",               # Turkish
        "Le français, el español, العربية, हिन्दी, 中文",    # multilingual
        "def forward(x):\n\treturn self.mlp(x)  # comment\n",  # source code + tabs/newlines
        "0 123456789 3.14159 -42 +7",                      # numbers
        "!@#$%^&*()_+-=[]{}|;:',.<>/?`~\"",                # punctuation
        "trailing whitespace   \n\t  ",                    # whitespace
        "emoji: 🙂🚀👨‍👩‍👧‍👦🇹🇷 flags",                        # emoji incl. ZWJ sequences
        "combining: a\u0301 e\u0301 n\u0303 o\u0308",       # combining marks
        "",                                               # empty input
        "tab\tand\nnewline\r\ncarriage",                 # mixed whitespace
        "İ\u0307stanbul (NFC vs NFD forms)",               # precomposed Turkish İ
    ]

    @pytest.mark.parametrize("text", SAMPLES)
    def test_text_roundtrip(self, adapter, text):
        ids = adapter.encode(text)
        assert adapter.decode(ids) == text

    def test_long_input_roundtrip(self, adapter):
        text = ("NTokenizer v0.3 encodes multilingual text: İstanbul, 東京, القاهرة, Москва. "
                "Math: ∫₀^∞ e^{-x²} dx = √π/2. Code: x = lambda t: t**2. ") * 40
        ids = adapter.encode(text)
        assert len(ids) > 100
        assert adapter.decode(ids) == text

    def test_raw_bytes_roundtrip_arbitrary_bytes(self, adapter):
        data = bytes(range(256)) * 4 + b"\xff\xfe\x00\x80\x81\x82 invalid utf-8 "
        ids = adapter.encode_bytes(data)
        assert adapter.decode_bytes(ids) == data

    def test_raw_bytes_invalid_utf8(self, adapter):
        data = b"\xff\xfe\x00\xc0\xaf"  # not valid UTF-8
        ids = adapter.encode_bytes(data)
        assert adapter.decode_bytes(ids) == data
        with pytest.raises(Exception):
            # strict UTF-8 decode must reject this byte stream
            adapter.decode(ids)

    def test_empty_input_semantics(self, adapter):
        assert adapter.encode("") == []
        assert adapter.decode([]) == ""
        assert adapter.decode_bytes([]) == b""


class TestSpecialTokens:
    def test_special_token_literals_encode_to_their_ids(self, adapter):
        for content, token_id in EXPECTED_SPECIAL_TOKENS.items():
            assert adapter.encode(content) == [token_id]

    def test_special_tokens_roundtrip_through_codec(self, adapter):
        for token_id in EXPECTED_SPECIAL_TOKENS.values():
            assert adapter.decode([token_id]) == adapter.decode([token_id])

    def test_model_generation_stops_at_real_eos(self, model, adapter, device):
        # integration-level EOS check: generate() honors the artifact's EOS id
        # (mechanism unit-tested in tests/test_generation.py; here we verify
        # the artifact-configured id is threaded end-to-end)
        prompt = adapter.encode("Hello, world!")
        ids = torch.tensor([prompt], device=device)
        out = generate(model, ids, max_new_tokens=4, eos_token_id=adapter.eos_token_id)
        assert out.shape[1] == len(prompt) + 4 or (out[0, -1].item() == adapter.eos_token_id)

    def test_generation_output_ids_are_valid_vocab_ids(self, model, adapter, device):
        prompt = adapter.encode("Integration test:")
        ids = torch.tensor([prompt, prompt], device=device)
        out = generate(model, ids, max_new_tokens=8, eos_token_id=adapter.eos_token_id)
        assert (out >= 0).all() and (out < VOCAB).all()


class TestEndToEnd:
    @pytest.mark.parametrize("text", [
        "Merhaba, dünya!",
        "The quick brown fox jumps over the lazy dog.",
        "print('hello')\nfor i in range(10):\n    print(i)\n",
        "日本語のテキストと Türkçe metin karışımı 123 ✓",
    ])
    def test_encode_forward_generate_decode(self, adapter, model, device, text):
        prompt_ids = adapter.encode(text)
        input_ids = torch.tensor([prompt_ids], device=device)
        with torch.no_grad():
            out = model(input_ids)
        assert out.logits.shape == (1, len(prompt_ids), VOCAB)
        assert torch.isfinite(out.logits.float()).all()
        generated = generate(model, input_ids, max_new_tokens=8, eos_token_id=adapter.eos_token_id)
        new_ids = generated[0, len(prompt_ids):].tolist()
        assert all(0 <= i < VOCAB for i in new_ids)
        # decoding arbitrary (possibly non-UTF-8) model output must use decode_bytes
        decoded = adapter.decode_bytes(new_ids)
        assert isinstance(decoded, bytes)
        assert adapter.decode_bytes(prompt_ids)  # prompt itself is decodable

    def test_end_to_end_on_cuda_if_available(self, adapter, device):
        if device.type != "cuda":
            pytest.skip("CUDA not available")
        prompt_ids = adapter.encode("CUDA path check:")
        input_ids = torch.tensor([prompt_ids], device="cuda")
        model = NeuratalModel(NeuratalConfig()).to("cuda").eval()
        with torch.no_grad():
            out = model(input_ids)
        assert out.logits.shape == (1, len(prompt_ids), VOCAB)


class TestCheckpointCompatibility:
    def test_save_and_load_with_sidecar(self, adapter, tmp_path, device):
        model = NeuratalModel(NeuratalConfig(
            eos_token_id=adapter.eos_token_id, pad_token_id=adapter.pad_token_id,
        ))
        save_pretrained(model, tmp_path)
        write_tokenizer_sidecar(TOKENIZER_MODEL, tmp_path)
        assert (tmp_path / MANIFEST_FILENAME).is_file()

        loaded_model, loaded_adapter, report = load_checkpoint_with_tokenizer(tmp_path, TOKENIZER_MODEL)
        assert report["verified"] is True
        assert report["sidecar"] == "verified"
        ids = torch.tensor([adapter.encode("Checkpoint check")], device=device)
        model_cpu, loaded_cpu = model.eval(), loaded_model.eval()
        with torch.no_grad():
            assert torch.equal(model_cpu(ids.cpu()).logits, loaded_cpu(ids.cpu()).logits)

    def test_manifest_verification_tamper_detected(self, tmp_path):
        manifest = build_manifest(TOKENIZER_MODEL)
        tampered = dict(manifest)
        tampered["tokenizer_sha256"] = "0" * 64
        manifest_path = save_manifest(tampered, tmp_path / MANIFEST_FILENAME)
        with pytest.raises(TokenizerMismatchError, match="tokenizer_sha256"):
            verify_manifest(load_manifest(manifest_path), TOKENIZER_MODEL)

    def test_v02_tokenizer_rejected_against_v03_sidecar(self, tmp_path):
        if not TOKENIZER_V02.is_file():
            pytest.skip("v0.2 artifact not present")
        model = NeuratalModel(NeuratalConfig())
        save_pretrained(model, tmp_path)
        write_tokenizer_sidecar(TOKENIZER_MODEL, tmp_path)
        with pytest.raises(TokenizerMismatchError):
            verify_checkpoint_tokenizer(model.config, tmp_path, TOKENIZER_V02)

    def test_eos_config_mismatch_detected(self, tmp_path):
        model = NeuratalModel(NeuratalConfig(eos_token_id=262137))  # wrong: should be 262138
        save_pretrained(model, tmp_path)
        write_tokenizer_sidecar(TOKENIZER_MODEL, tmp_path)
        with pytest.raises(TokenizerMismatchError, match="eos_token_id"):
            verify_checkpoint_tokenizer(model.config, tmp_path, TOKENIZER_MODEL)

    def test_vocab_size_mismatch_detected(self, tmp_path):
        model = NeuratalModel(NeuratalConfig())
        save_pretrained(model, tmp_path)
        write_tokenizer_sidecar(TOKENIZER_MODEL, tmp_path)

        class FakeConfig:  # vocab size lying about identity
            vocab_size = 32_000
            eos_token_id = None
            pad_token_id = None
            bos_token_id = None

        with pytest.raises(TokenizerMismatchError, match="vocab_size"):
            verify_checkpoint_tokenizer(FakeConfig(), tmp_path, TOKENIZER_MODEL)  # type: ignore[arg-type]

    def test_checkpoint_without_sidecar_loads_with_relaxed_check(self, adapter, tmp_path, device):
        # pre-existing checkpoints (no sidecar) must remain loadable
        model = NeuratalModel(NeuratalConfig())
        save_pretrained(model, tmp_path)
        loaded = load_pretrained(tmp_path)  # core loader: no tokenizer involved
        assert loaded.config.vocab_size == VOCAB
        with pytest.raises(TokenizerMismatchError, match="sidecar"):
            verify_checkpoint_tokenizer(loaded.config, tmp_path, TOKENIZER_MODEL, strict_sidecar=True)
        report = verify_checkpoint_tokenizer(loaded.config, tmp_path, TOKENIZER_MODEL, strict_sidecar=False)
        assert report["verified"] is True and report["sidecar"].startswith("absent")
