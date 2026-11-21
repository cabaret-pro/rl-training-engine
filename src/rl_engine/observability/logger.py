"""Small composable sinks for durable structured experiment metrics."""

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

MetricRecord = Mapping[str, object]
MetricSink = Callable[[dict[str, object]], None]


class JsonlMetricSink:
    """Append one flushed JSON object per line for crash-tolerant run inspection."""

    def __init__(self, path: str | Path, *, append: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not append:
            self.path.write_text("", encoding="utf-8")

    def __call__(self, metrics: dict[str, object]) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, sort_keys=True) + "\n")
            stream.flush()


class CompositeMetricSink:
    def __init__(self, sinks: Sequence[MetricSink]) -> None:
        if not sinks:
            raise ValueError("at least one metric sink is required")
        self._sinks = tuple(sinks)

    def __call__(self, metrics: dict[str, object]) -> None:
        for sink in self._sinks:
            sink(metrics)

