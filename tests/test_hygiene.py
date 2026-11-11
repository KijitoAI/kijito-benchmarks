"""Repository hygiene: the public harness must not name the service's internals, must not depend on them,
and must not carry secrets or machine-specific paths. Runs offline.

The forbidden identifiers are listed as sha256 prefixes (so this file does not itself contain them):
internal function / module names of the service, and internal project or person names. Fixed-string
checks are built from pieces for the same reason."""
import gzip
import hashlib
import importlib
import ipaddress
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES = ["kijito_bench", "kijito_bench.client", "kijito_bench.llm", "kijito_bench.runner",
           "kijito_bench.baselines", "kijito_bench.longmemeval", "kijito_bench.stale",
           "kijito_bench.stale_agent", "kijito_bench.groupmembench"]

_FORBIDDEN_TOKEN_SHA16 = {
    "06b9a6eacd7a77b9",
    "07726d002663c303",
    "0b2319dc1ccf7803",
    "0c087a00700e6e97",
    "179f42aeeca9a72b",
    "22b5d7bb0a7b3fb3",
    "233e1cded61ace81",
    "23958818c1ee1e44",
    "27c556c9487fb831",
    "2a03c71980f3cd30",
    "2ac45e0526e36a1f",
    "2fc2a6dd013e7ab4",
    "33c8a9ed94b80357",
    "386de061f90d8440",
    "3a5d7960f9b47d9a",
    "3ab6bb05cf7ff055",
    "401cb0a3c9577a6a",
    "444b759c5264422e",
    "44d5ba7b4570b810",
    "4f6b9e1c24d93d9a",
    "57de4cf40144bdf7",
    "587fe2f2b3d0c7cf",
    "5f5a8ed8f139be6d",
    "6d9bd696c1c11886",
    "7a744c786b15a35c",
    "7b4b9a1e2affa870",
    "896b53a34f42233b",
    "9095239e58d405fb",
    "9b2153df65c7cba8",
    "9f58665a48beb853",
    "a3a5c8478f441dca",
    "a4a7f838e3625d37",
    "a98da68c461b4494",
    "aba5093c1be71914",
    "b70a044b64308d53",
    "b95505f201888ad7",
    "cb6e6bd978e77d85",
    "cf8214114789a046",
    "d8872aff63c047bd",
    "dc1b75fbd251a495",
    "e129f5000a4b71cc",
    "e1e08c2a300be7a4",
    "ec49664aacc1fc76",
    "f11959be970bca3b",
    "f1611d869b98fd9b",
    "f3bb58ab52b0b281",
    "f6e3659767e19711",
    "f8a2087f07e314d2",
    "fd4f3d9abeca4f87",
    "fe9832ce7a939a1a",
}
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_PATTERNS = {
    "server env knob prefix": re.compile("KIJITO" + "_"),
    "absolute home path": re.compile("/" + "Users/|/" + "home/[a-z]"),
    "OpenAI/Anthropic-style key": re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_\-]{16,}"),
    "bearer token": re.compile(r"Bearer\s+[A-Za-z0-9_\-.]{16,}"),
    "Kijito-style key": re.compile(r"\bkjt_[A-Za-z0-9_\-]{12,}"),
    "internal service port": re.compile(r":74(?:7[45]|91)\b"),
}
_IPV4_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")


def _files():
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                             capture_output=True, text=True, timeout=30, check=True).stdout.split()
    except Exception:
        out = [os.path.relpath(os.path.join(d, f), ROOT) for d, dirs, fs in os.walk(ROOT)
               for f in fs if not any(p in d for p in (".git", "__pycache__", ".pytest_cache"))]
    return [f for f in out if os.path.isfile(os.path.join(ROOT, f))]


def _text(path):
    # a gzip file is scanned as the text it holds, so a compressed file cannot hide from the scan
    try:
        if path.endswith(".gz"):
            return gzip.open(os.path.join(ROOT, path), "rt", encoding="utf-8").read()
        return open(os.path.join(ROOT, path), encoding="utf-8").read()
    except UnicodeDecodeError:
        return None


def _hits(text, identifiers=True):
    found = []
    for tok in (_TOKEN_RE.findall(text) if identifiers else ()):
        parts = tok.split(".")
        cands = {tok} | set(parts) | {".".join(parts[:i]) for i in range(2, len(parts) + 1)}
        for c in cands:
            if hashlib.sha256(c.lower().encode()).hexdigest()[:16] in _FORBIDDEN_TOKEN_SHA16:
                found.append(f"forbidden identifier {c!r}")
    for name, rx in _PATTERNS.items():
        for m in rx.finditer(text):
            found.append(f"{name}: {m.group(0)[:40]!r}")
    for m in _IPV4_RE.finditer(text):
        try:
            ip = ipaddress.ip_address(m.group(0))
        except ValueError:
            continue
        if not (ip.is_loopback or ip.is_unspecified or ip.is_private):
            found.append(f"public IP address {m.group(0)!r}")
    return found


def test_the_scan_sees_the_whole_repository():
    files = _files()
    assert "README.md" in files and "kijito_bench/client.py" in files and "tests/test_hygiene.py" in files


def test_the_scanner_catches_what_it_is_for():
    assert _hits("x = " + "KIJITO" + "_SOMETHING")
    assert _hits("open('/" + "Users/someone/x')")
    assert _hits("key = 'sk-" + "ant-" + "a" * 30 + "'")
    assert _hits("Authorization: Bearer " + "b" * 30)
    assert _hits("from kijito" + ".core.query import x")
    assert _hits("server at " + ".".join(["8"] * 4))
    assert not _hits("http://127.0.0.1:8080 and https://api.kijito.ai and kijito_recall and kijito_bench")


def test_a_gzip_file_is_scanned_as_its_text(tmp_path, monkeypatch):
    with gzip.open(tmp_path / "x.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write("server at " + ".".join(["8"] * 4) + "\n")
    monkeypatch.setattr(sys.modules[__name__], "ROOT", str(tmp_path))
    assert _hits(_text("x.jsonl.gz"))


def test_no_tracked_file_names_internals_secrets_or_machine_paths():
    bad = {}
    for f in _files():
        text = _text(f)
        if text is None:
            continue
        # Per-question result files, the official-schema copies of them and the STALE distill cache hold
        # model and dataset text, where ordinary words that double as internal names occur (a place name,
        # or an everyday word in a GroupMemBench message); every other check still applies to them.
        model_text = f.startswith("results/") and (
            f.endswith(("items.jsonl", "distill-cache.jsonl.gz")) or "/official-schema/" in f)
        hits = _hits(text, identifiers=not model_text)
        if hits:
            bad[f] = sorted(set(hits))[:10]
    assert bad == {}


@pytest.mark.parametrize("mod", MODULES)
def test_every_module_imports_without_the_service_package(mod):
    """Import in a fresh interpreter that refuses the service's own packages, so a hidden dependency on
    them cannot pass because they happen to be installed."""
    code = (
        "import sys, importlib.abc\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        top = name.split('.')[0]\n"
        "        if top in ('kijito', 'recall" + "_quality', 'benchmarks'):\n"
        "            raise ImportError('blocked: ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        f"import {mod}\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr


def test_modules_import_here_too():
    for m in MODULES:
        importlib.import_module(m)
