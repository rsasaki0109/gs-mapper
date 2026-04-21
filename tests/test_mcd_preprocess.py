"""Focused unit tests for MCD preprocessing helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from gs_sim2real.preprocess.mcd import MCDCalibrationResolver, MCDPreprocessOptions, parse_topic_arg


def test_parse_topic_arg_splits_comma_separated_topics() -> None:
    assert parse_topic_arg(None) is None
    assert parse_topic_arg(" /cam0/image , /cam1/image ,, ") == ["/cam0/image", "/cam1/image"]
    assert parse_topic_arg("/cam0/image") == "/cam0/image"


def test_options_from_namespace_normalizes_offsets_and_tuples() -> None:
    args = SimpleNamespace(
        image_topic="/cam",
        every_n=3,
        mcd_start_offset_sec=-2.0,
        mcd_gnss_antenna_offset_enu=[1, 2, 3],
        mcd_gnss_antenna_offset_base=None,
    )

    options = MCDPreprocessOptions.from_namespace(args)

    assert options.parsed_image_topics == "/cam"
    assert options.every_n == 3
    assert options.start_offset_sec == 0.0
    assert options.mcd_gnss_antenna_offset_enu == (1.0, 2.0, 3.0)


def test_calibration_resolver_writes_pinhole_from_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "calibration_atv.yaml"
    yaml_path.write_text(
        """
body:
  d455b_color:
    rostopic: /d455b/color/image_raw
    intrinsics: [421.0, 422.0, 320.0, 240.0]
    resolution: [640, 480]
    T:
      - [1, 0, 0, 0]
      - [0, 1, 0, 0]
      - [0, 0, 1, 0]
      - [0, 0, 0, 1]
""",
        encoding="utf-8",
    )

    class FakeLoader:
        @staticmethod
        def _sanitize_topic_name(topic: str) -> str:
            return topic.strip("/").replace("/", "_")

    options = MCDPreprocessOptions(
        image_topic="/d455b/color/image_raw",
        mcd_static_calibration=str(yaml_path),
    )
    resolver = MCDCalibrationResolver(FakeLoader(), tmp_path, options)  # type: ignore[arg-type]

    out = resolver.write_pinhole_from_calibration_yaml("/d455b/color/image_raw")

    assert out == tmp_path / "calibration" / "d455b_color_image_raw.json"
    assert out is not None
    text = out.read_text(encoding="utf-8")
    assert '"width": 640' in text
    assert '"height": 480' in text
    assert '"fx": 421.0' in text
    assert '"frame_id": "d455b_color"' in text
