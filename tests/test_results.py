from pathlib import Path

from spikeworld.results import verify


def test_paper_result_snapshot():
    verify(Path(__file__).parents[1] / "results" / "paper_results.json")
