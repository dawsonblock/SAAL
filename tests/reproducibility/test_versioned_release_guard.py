import importlib.util
from pathlib import Path

import pytest


def _builder(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    tools = root / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("legacy_release_builder", tools / "build_reproducible_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_release_builder_rejects_v097_package(tmp_path, monkeypatch):
    builder = _builder(monkeypatch)
    (tmp_path / "pyproject.toml").write_text('[project]\nversion="0.9.7.dev0"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="legacy v1 release builder only supports 0.9.4.3"):
        builder.require_legacy_release(tmp_path)


def test_legacy_release_builder_accepts_only_its_bound_version(tmp_path, monkeypatch):
    builder = _builder(monkeypatch)
    (tmp_path / "pyproject.toml").write_text('[project]\nversion="0.9.4.3"\n', encoding="utf-8")
    builder.require_legacy_release(tmp_path)
