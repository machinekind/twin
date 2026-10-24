"""The package talks to models only through the CLIs: no twin, no model SDK, no HTTP, no sockets."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import machinekind_twin_llm

PACKAGE = Path(machinekind_twin_llm.__file__).resolve().parent

#: Top-level modules the package must never import (the API package is allowed).
FORBIDDEN = ("machinekind_twin", "anthropic", "openai", "httpx", "requests", "aiohttp", "socket")
#: Dotted modules it must never import.
FORBIDDEN_DOTTED = ("urllib.request", "http.client")
#: Hosts it must never name.
HOSTS = ("api.anthropic.com", "api.openai.com")


def _sources() -> list[Path]:
    paths = sorted(PACKAGE.rglob("*.py"))
    assert paths
    return paths


def _forbidden(module: str) -> bool:
    return module.split(".")[0] in FORBIDDEN or any(
        module == m or module.startswith(m + ".") for m in FORBIDDEN_DOTTED)


def test_no_module_imports_the_twin_a_model_sdk_or_the_network():
    for path in _sources():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""] if node.level == 0 else []
                modules += [f"{node.module}.{alias.name}" for alias in node.names if node.module and not node.level]
            else:
                continue
            for module in modules:
                assert not _forbidden(module), f"{path.name}: {module}"


def test_no_module_names_a_model_api_host():
    for path in _sources():
        text = path.read_text(encoding="utf-8")
        for host in HOSTS:
            assert host not in text, f"{path.name}: {host}"


def test_importing_every_module_loads_none_of_them():
    names = sorted(f"machinekind_twin_llm.{p.stem}" for p in _sources() if p.stem != "__init__")
    script = (
        "import importlib, json, sys\n"
        f"for name in {names!r}: importlib.import_module(name)\n"
        "print(json.dumps(sorted(sys.modules)))\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)

    loaded = json.loads(out.stdout)
    assert "machinekind_twin_llm.cli" in loaded
    assert [m for m in loaded if _forbidden(m)] == []
