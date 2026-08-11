from pathlib import Path

from spikeworld.results import build, verify


def test_paper_result_snapshot():
    root = Path(__file__).parents[1]
    verify(root / "results" / "paper_results.json", root / "results" / "raw")


def test_raw_outputs_rebuild_all_four_result_groups():
    root = Path(__file__).parents[1]
    rebuilt = build(root / "results" / "raw")
    assert set(rebuilt) == {"joint", "adaptation", "control", "system"}
    assert rebuilt["adaptation"]["episodes"] == 80
