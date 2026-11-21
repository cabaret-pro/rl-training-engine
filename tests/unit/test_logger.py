import json
from pathlib import Path

from rl_engine.observability.logger import CompositeMetricSink, JsonlMetricSink


def test_jsonl_sink_persists_and_appends_complete_records(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    sink = JsonlMetricSink(path)
    sink({"step": 0, "reward": 0.25})
    JsonlMetricSink(path, append=True)({"step": 1, "reward": 0.5})

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records == [
        {"reward": 0.25, "step": 0},
        {"reward": 0.5, "step": 1},
    ]


def test_composite_sink_fans_out_in_order() -> None:
    calls: list[tuple[str, int]] = []
    sink = CompositeMetricSink(
        [
            lambda record: calls.append(("first", int(record["step"]))),
            lambda record: calls.append(("second", int(record["step"]))),
        ]
    )

    sink({"step": 3})

    assert calls == [("first", 3), ("second", 3)]
