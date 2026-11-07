"""Verify prepared public source bytes; optional checkpoint has its own manifest."""
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
