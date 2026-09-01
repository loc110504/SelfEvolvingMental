from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from psyvec.config import (  # noqa: E402
    BackendConfig,
    ConfigurationError,
    load_backend_config,
    load_backend_manifest,
)


class BackendConfigTests(unittest.TestCase):
    def test_manifest_loader_reads_the_versioned_default(self) -> None:
        manifest = load_backend_manifest()

        self.assertEqual(manifest["backend"], "legacy-autogen")
        self.assertNotIn("api_key", manifest)

    def test_manifest_loader_default_is_independent_of_working_directory(self) -> None:
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary_directory:
            try:
                os.chdir(temporary_directory)
                manifest = load_backend_manifest()
            finally:
                os.chdir(original_directory)

        self.assertEqual(manifest["model"], "qwen2.5-72b")

    def test_manifest_loader_rejects_unknown_fields(self) -> None:
        manifest_path = self._write_manifest('{"unexpected": true}')

        with self.assertRaisesRegex(ConfigurationError, "unknown manifest"):
            load_backend_manifest(manifest_path)

    def test_manifest_loader_rejects_secret_fields(self) -> None:
        manifest_path = self._write_manifest('{"api_key": "not-a-real-secret"}')

        with self.assertRaisesRegex(ConfigurationError, "secret fields"):
            load_backend_manifest(manifest_path)

    def test_manifest_loader_reports_missing_file(self) -> None:
        missing_path = PROJECT_ROOT / "tests" / "fixtures" / "missing-backend.json"

        with self.assertRaisesRegex(ConfigurationError, "does not exist"):
            load_backend_manifest(missing_path)

    def test_manifest_loader_reports_malformed_json(self) -> None:
        manifest_path = self._write_manifest("{")

        with self.assertRaisesRegex(ConfigurationError, "malformed JSON"):
            load_backend_manifest(manifest_path)

    def test_versioned_base_config_matches_code_defaults(self) -> None:
        base_path = PROJECT_ROOT / "configs" / "base" / "backend.json"
        values = json.loads(base_path.read_text(encoding="utf-8"))
        self.assertEqual(BackendConfig(**values), BackendConfig())

    def test_layers_apply_in_documented_order(self) -> None:
        config = load_backend_config(
            environment={
                "PSYVEC_MODEL": "environment-model",
                "PSYVEC_MAX_TOKENS": "100",
            },
            manifest={"model": "manifest-model", "max_tokens": 200},
            overrides={"max_tokens": 300},
        )
        self.assertEqual(config.model, "manifest-model")
        self.assertEqual(config.max_tokens, 300)

    def test_secrets_are_redacted_and_do_not_change_provenance_hash(self) -> None:
        first = BackendConfig(api_key="secret-one")
        second = BackendConfig(api_key="secret-two")

        self.assertEqual(first.redacted_dict()["api_key"], "<redacted>")
        self.assertNotIn("secret-one", repr(first.redacted_dict()))
        self.assertNotIn("secret-one", repr(first))
        self.assertEqual(first.provenance_hash, second.provenance_hash)

    def test_feature_flag_can_restore_legacy_call_path(self) -> None:
        config = load_backend_config(
            environment={"PSYVEC_USE_BACKEND_ABSTRACTION": "false"}
        )
        self.assertFalse(config.use_backend_abstraction)

    def test_unknown_manifest_field_fails_closed(self) -> None:
        with self.assertRaises(ConfigurationError):
            load_backend_config(environment={}, manifest={"api_token": "secret"})

    def test_invalid_values_fail_closed(self) -> None:
        with self.assertRaises(ConfigurationError):
            load_backend_config(environment={"PSYVEC_TIMEOUT_SECONDS": "not-a-number"})

    def _write_manifest(self, contents: str) -> Path:
        manifest_path = PROJECT_ROOT / "tests" / "fixtures" / "backend-manifest.json"
        self.addCleanup(manifest_path.unlink, missing_ok=True)
        manifest_path.write_text(contents, encoding="utf-8")
        return manifest_path


if __name__ == "__main__":
    unittest.main()
