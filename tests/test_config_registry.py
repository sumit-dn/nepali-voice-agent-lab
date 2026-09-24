import json

import pytest

from voice_lab import config
from voice_lab.config import KINDS, ConfigError, ModelSpec, load_models
from voice_lab.evaluation.manifest import ManifestError, load_manifest
from voice_lab.providers.registry import ADAPTERS, CloudNotAllowed, build, select

REAL_CONFIGS = config.CONFIG_DIR
ALLOWED_COMMERCIAL = {"yes", "no", "conditional", "REVIEW REQUIRED"}


def test_real_registries_are_valid():
    ids = []
    for kind in KINDS:
        for spec in load_models(kind, REAL_CONFIGS).values():
            ids.append(spec.id)
            assert spec.provider in ADAPTERS or spec.provider == "none", spec.id
            assert spec.license.get("commercial_use") in ALLOWED_COMMERCIAL, spec.id
            assert spec.license.get("url"), f"{spec.id}: licence reference missing"
            assert spec.verified, f"{spec.id}: verification date missing"
            if spec.external:
                assert not spec.enabled, f"{spec.id}: cloud models must be off by default"
    assert len(ids) == len(set(ids)), "model ids must be unique across kinds (CLI looks them up globally)"


def test_unknown_and_duplicate_keys(tmp_path):
    (tmp_path / "asr.yaml").write_text("models:\n  - {id: a, provider: hf-asr, typo_field: 1}\n")
    with pytest.raises(ConfigError, match="unknown keys"):
        load_models("asr", tmp_path)
    (tmp_path / "asr.yaml").write_text("models:\n  - {id: a, provider: x}\n  - {id: a, provider: x}\n")
    with pytest.raises(ConfigError, match="duplicate"):
        load_models("asr", tmp_path)


def test_fingerprint_changes_with_params():
    a = ModelSpec(id="m", provider="p", params={"x": 1})
    b = ModelSpec(id="m", provider="p", params={"x": 2})
    assert a.fingerprint() != b.fingerprint()


def test_cloud_guard_and_selection(lab):
    with pytest.raises(CloudNotAllowed, match="Nothing was sent"):
        build("asr", "fake-cloud-asr")
    assert build("asr", "fake-cloud-asr", allow_cloud=True).spec.id == "fake-cloud-asr"
    assert [s.id for s in select("asr", None)] == ["fake-asr"]  # disabled/cloud excluded by default
    with pytest.raises(Exception, match="No adapter"):
        build("asr", "tracked-only")


def test_manifest_validation(lab):
    bad = lab / "bad.jsonl"
    bad.write_text(
        "\n".join(
            [
                json.dumps({"id": "x", "audio": "data/missing.wav", "text": "t"}),
                json.dumps({"id": "x", "audio": "data/a.wav", "text": "t"}),
                json.dumps({"id": "y", "text": "no audio"}),
                "{not json",
            ]
        )
    )
    with pytest.raises(ManifestError) as err:
        load_manifest(bad, "asr")
    msg = str(err.value)
    assert "not found" in msg and "duplicate" in msg and "missing ['audio']" in msg and "invalid JSON" in msg
    assert len(load_manifest(lab / "data/manifests/asr.jsonl", "asr")) == 2


def test_dotenv_does_not_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nVOICE_LAB_T1=from_file\nVOICE_LAB_T2='quoted'\nVOICE_LAB_T3=\n")
    monkeypatch.setenv("VOICE_LAB_T1", "from_env")
    monkeypatch.delenv("VOICE_LAB_T2", raising=False)
    monkeypatch.delenv("VOICE_LAB_T3", raising=False)
    config.load_dotenv(env)
    import os

    assert os.environ["VOICE_LAB_T1"] == "from_env" and os.environ["VOICE_LAB_T2"] == "quoted"
    # an empty template line must not be exported, or it would mask a value added to .env later
    assert "VOICE_LAB_T3" not in os.environ


def test_gated_repo_explains_how_to_get_access(monkeypatch):
    import httpx
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError

    from voice_lab.download import check_access
    from voice_lab.providers.base import ProviderError

    denied = httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co/org/model"))

    def deny(self, repo_id, **kwargs):
        raise GatedRepoError("gated", response=denied)

    monkeypatch.setattr(HfApi, "auth_check", deny)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(ProviderError, match="Agree and access repository") as err:
        check_access("org/model")
    assert "HF_TOKEN is not set" in str(err.value)
