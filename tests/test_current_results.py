import importlib.util
from pathlib import Path


def test_current_main_stream_reconstruction():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("verify_current", root / "tools/verify_current.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.verify(root)["status"] == "PASS"
