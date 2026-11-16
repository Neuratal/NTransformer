"""Optional NTokenizer ↔ NeuratalTransformer integration (never imported by the core)."""

from neuratal.integration.adapter import (
    DEFAULT_TOKENIZER_PATH,
    EXPECTED_SPECIAL_TOKENS,
    NTokenizerAdapter,
    NTokenizerUnavailableError,
    load_checkpoint_with_tokenizer,
    verify_checkpoint_tokenizer,
    write_tokenizer_sidecar,
)
from neuratal.integration.manifest import (
    MANIFEST_FILENAME,
    SCHEMA,
    ManifestError,
    NTokenizerIntegrationError,
    TokenizerMismatchError,
    build_manifest,
    load_manifest,
    parse_ntok_file,
    save_manifest,
    verify_manifest,
)

__all__ = [
    "SCHEMA",
    "MANIFEST_FILENAME",
    "DEFAULT_TOKENIZER_PATH",
    "EXPECTED_SPECIAL_TOKENS",
    "NTokenizerAdapter",
    "NTokenizerUnavailableError",
    "NTokenizerIntegrationError",
    "ManifestError",
    "TokenizerMismatchError",
    "build_manifest",
    "load_manifest",
    "parse_ntok_file",
    "save_manifest",
    "verify_manifest",
    "verify_checkpoint_tokenizer",
    "load_checkpoint_with_tokenizer",
    "write_tokenizer_sidecar",
]
