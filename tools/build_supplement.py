"""Build an anonymous code/evidence ZIP from an explicit release-file allowlist."""
import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_DIRS = {"src", "tests", "configs", "results", "artifacts", "tools"}
ALLOWED_ROOT = {"LICENSE", "pyproject.toml", "uv.lock", "REPRODUCIBILITY.md"}
TEXT = {".md", ".py", ".json", ".toml", ".lock", ".txt"}
IDENTITY = re.compile(r"Oooorca|Ziqiao\s*Yu|[A-Za-z]:[\\/](?:AI|Users)|/(?:Users|home)/", re.IGNORECASE)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def anonymize(text):
    text = re.sub(r"https://github\.com/Oooorca/SpikeWorld[^\s\"<>)]*", "", text)
    text = re.sub(r"[A-Za-z]:[\\/][^\s\"<>]+", lambda m: "[local]/" + m[0].replace("\\", "/").rsplit("/", 1)[-1], text)
    text = re.sub(r"/(?:Users|home)/[^\s\"<>]+", lambda m: "[local]/" + m[0].rsplit("/", 1)[-1], text)
    text = re.sub(r"Ziqiao\s*Yu", "Anonymous authors", text)
    return text


def transform(value):
    if isinstance(value, dict):
        return {k: transform(v) for k, v in value.items()}
    if isinstance(value, list):
        return [transform(v) for v in value]
    return anonymize(value) if isinstance(value, str) else value


def numeric(value, at=""):
    if isinstance(value, dict):
        return [x for k, v in value.items() for x in numeric(v, at + "/" + k)]
    if isinstance(value, list):
        return [x for i, v in enumerate(value) for x in numeric(v, at + "/" + str(i))]
    return [(at, value)] if isinstance(value, (int, float, bool)) or value is None else []


README = """# SpikeWorld — anonymous supplementary code and evidence

Companion to **SpikeWorld: Adapting Frozen World Models with Compact State**.

## Quick start (Python 3.11)

Unzip and enter `spikeworld_supplement/`. No network or ML package is needed for:

```bash
python tools/verify_inventory.py
python tools/verify_current.py
```

For executable model/interface tests, install the locked environment:

```bash
uv sync --locked --extra dev --extra train
uv run pytest -q
uv run spikeworld-verify
uv run spikeworld-audit --config configs/audit.toml
```

Dependencies must be installed first; they are not vendored in the ZIP. The
representative legacy seed-42401 checkpoint is included and requires no download.
`spikeworld-fetch` only verifies that included file in this anonymous package.
No identity-linked network request is made.

## Contents and boundaries

- `src/`, `tests/`, `configs/`: model, bounded deployment state, reference runners,
  tests, and configurations.
- `results/current/`: current manuscript evidence and an export manifest. The
  main 3-foundation table can be reconstructed from transition records.
- `results/raw/`, `results/paper_results.json`: preserved legacy experiment snapshot;
  `spikeworld-verify` checks these, not the newer table.
- `artifacts/`: representative legacy checkpoint and causal shift streams.
- `REPRODUCIBILITY.md`: per-table mapping, protocol definitions, and unavailable
  full-run dependencies. This is not a complete from-scratch rerun distribution.
- `INVENTORY.json`: checksums of the final anonymous package contents.

The separate later direct-output calibration protocol is retained under
`results/current/additional_protocol/`, not pooled with the main table.
Internal manuscript reviews, author identifiers, Git history, machine paths,
and old submission ZIPs are excluded. Scientific numeric results are unchanged.
The paper PDF is submitted separately; this ZIP contains no duplicate paper PDF.
"""

INVENTORY_CHECK = '''"""Verify the exact unpacked anonymous artifact inventory."""
import hashlib
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "INVENTORY.json").read_text(encoding="utf-8"))
for row in manifest["files"]:
    data = (root / row["file"]).read_bytes()
    assert len(data) == row["bytes"], row["file"]
    assert hashlib.sha256(data).hexdigest() == row["sha256"], row["file"]
print("PASS:", len(manifest["files"]), "files verified")
'''


def build(output):
    output.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="stage-", dir=output)) / "spikeworld_supplement"
    stage.mkdir()
    names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True).splitlines()
    names = sorted(set(names) | {"artifacts/spikeworld_seed42401.pt"})
    changes = []
    for name in names:
        relative = Path(name)
        if name not in ALLOWED_ROOT and relative.parts[0] not in ALLOWED_DIRS:
            continue
        if name == "tools/build_supplement.py" or any(p.startswith("._") or p in {"__pycache__", ".git"} or p.endswith(".egg-info") for p in relative.parts):
            continue
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(name)
        content = source.read_bytes()
        original = content
        if relative.suffix == ".json":
            before = json.loads(content)
            after = transform(before)
            assert numeric(before) == numeric(after), name
            content = (json.dumps(after, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        elif relative.suffix in TEXT or relative.name == "LICENSE":
            content = anonymize(content.decode("utf-8")).encode("utf-8")
        if name == "src/spikeworld/artifacts.py":
            code = content.decode("utf-8")
            code = re.sub(r"CHECKPOINT_URL = \([\s\S]*?\)", 'CHECKPOINT_URL = ""', code, count=1)
            code = code.replace('target.parent.mkdir(parents=True, exist_ok=True)',
                                'raise FileNotFoundError("Use the bundled checkpoint; anonymous package has no remote download.")\n    target.parent.mkdir(parents=True, exist_ok=True)')
            content = code.encode("utf-8")
        if name == "artifacts/README.md":
            content = b"# Bundled artifacts\n\nThe representative legacy seed-42401 checkpoint and causal shift streams are included. See ../REPRODUCIBILITY.md for missing full-run data and protocol boundaries.\n"
        if IDENTITY.search(content.decode("utf-8", errors="ignore")):
            raise ValueError("identity marker found in " + name)
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        if content != original:
            changes.append({"file": name, "original_sha256": sha(original), "packaged_sha256": sha(content)})
    (stage / "README.md").write_text(README, encoding="utf-8")
    (stage / "tools/verify_inventory.py").write_text(INVENTORY_CHECK, encoding="utf-8")
    # Rebind exported-file hashes after anonymous JSON serialization.
    manifest_path = stage / "results/current/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest["files"]:
        data = (manifest_path.parent / row["file"]).read_bytes()
        row["sha256"], row["bytes"] = sha(data), len(data)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for sums in stage.rglob("SHA256SUMS"):
        siblings = sorted(p for p in sums.parent.iterdir() if p.is_file() and p != sums)
        sums.write_text("".join(sha(p.read_bytes()) + "  " + p.name + "\n" for p in siblings), encoding="utf-8")
    inventory = [{"file": p.relative_to(stage).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
                 for p in sorted(stage.rglob("*")) if p.is_file()]
    (stage / "INVENTORY.json").write_text(json.dumps({"files": inventory}, indent=2) + "\n", encoding="utf-8")
    archive = output / "SpikeWorld_ICLR2027_supplement_V5_20260910.zip"
    if archive.exists():
        raise FileExistsError("Refusing to overwrite an existing supplement: " + str(archive))
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                info = zipfile.ZipInfo("spikeworld_supplement/" + path.relative_to(stage).as_posix(), (2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                bundle.writestr(info, path.read_bytes())
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        for info in bundle.infolist():
            assert not IDENTITY.search(info.filename)
            assert not IDENTITY.search(bundle.read(info).decode("utf-8", errors="ignore")), info.filename
    report = {"archive": str(archive), "staging": str(stage), "bytes": archive.stat().st_size,
              "sha256": sha(archive.read_bytes()), "files": len(inventory) + 1,
              "identity_scan": "PASS", "numeric_fields_retained": True,
              "anonymization_changes": changes}
    (output / "PACKAGE_BUILD.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "anonymization_changes"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output.resolve())
