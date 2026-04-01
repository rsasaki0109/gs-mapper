"""Shared documentation writers for experiment-first development."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def _render_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def build_repo_experiments_markdown(sections: Sequence[dict[str, Any]]) -> str:
    """Render the repository-wide experiment ledger."""
    latest_update = max(
        (section.get("updatedAt") for section in sections if section.get("updatedAt")),
        default=datetime.now(timezone.utc).isoformat(),
    )
    lines = [
        "# Experiments",
        "",
        f"Updated: {latest_update}",
        "",
        "This repository treats design as an evolving comparison space.",
        "Stable code stays in `core`; competing implementations stay in `experiments` until comparison makes part of them worth keeping.",
        "",
    ]
    for section in sections:
        lines.extend(
            [
                f"## {section['title']}",
                "",
                section["problemStatement"],
                "",
                "### Current Comparison",
                "",
            ]
        )
        lines.extend(_render_markdown_table(section["comparisonHeaders"], section["comparisonRows"]))
        lines.append("")
        for fixture in section.get("fixtureSections", []):
            lines.extend(
                [
                    f"### {fixture['title']}",
                    "",
                    fixture["intent"],
                    "",
                ]
            )
            lines.extend(_render_markdown_table(fixture["headers"], fixture["rows"]))
            lines.append("")
        if section.get("highlights"):
            lines.extend(["### Highlights", ""])
            for highlight in section["highlights"]:
                lines.append(f"- {highlight}")
            lines.append("")
    return "\n".join(lines) + "\n"


def build_repo_decisions_markdown(sections: Sequence[dict[str, Any]]) -> str:
    """Render the repository-wide decision ledger."""
    latest_update = max(
        (section.get("updatedAt") for section in sections if section.get("updatedAt")),
        default=datetime.now(timezone.utc).isoformat(),
    )
    lines = [
        "# Decisions",
        "",
        f"Updated: {latest_update}",
        "",
    ]
    for section in sections:
        lines.extend([f"## {section['title']}", ""])
        if section.get("accepted"):
            lines.extend(["### Accepted", ""])
            for item in section["accepted"]:
                lines.append(f"- {item}")
            lines.append("")
        if section.get("deferred"):
            lines.extend(["### Deferred", ""])
            for item in section["deferred"]:
                lines.append(f"- {item}")
            lines.append("")
        if section.get("rules"):
            lines.extend(["### Operating Rules", ""])
            for index, item in enumerate(section["rules"], start=1):
                lines.append(f"{index}. {item}")
            lines.append("")
    return "\n".join(lines) + "\n"


def build_repo_interfaces_markdown(sections: Sequence[dict[str, Any]]) -> str:
    """Render the minimal stable interfaces across current experiment seams."""
    lines = [
        "# Interfaces",
        "",
        "Only the surfaces that survived comparison should be treated as stable.",
        "",
    ]
    for section in sections:
        lines.extend([f"## {section['title']}", ""])
        lines.extend(["### Stable Core", ""])
        if section.get("stableInterfaceIntro"):
            lines.extend([section["stableInterfaceIntro"], ""])
        if section.get("stableInterfaceCode"):
            lines.extend(["```python", section["stableInterfaceCode"].rstrip(), "```", ""])
        if section.get("experimentContract"):
            lines.extend(["### Experiment Contract", ""])
            for item in section["experimentContract"]:
                lines.append(f"- {item}")
            lines.append("")
        if section.get("comparableInputs"):
            lines.extend(["### Comparable Inputs", ""])
            for item in section["comparableInputs"]:
                lines.append(f"- {item}")
            lines.append("")
        if section.get("boundary"):
            lines.extend(["### Boundary", ""])
            for item in section["boundary"]:
                lines.append(f"- {item}")
            lines.append("")
    return "\n".join(lines) + "\n"


def write_repo_experiment_docs(
    sections: Sequence[dict[str, Any]],
    *,
    docs_dir: str | Path,
) -> dict[str, str]:
    """Write the shared experiment-process docs."""
    docs_path = Path(docs_dir)
    docs_path.mkdir(parents=True, exist_ok=True)
    outputs = {
        "experiments": docs_path / "experiments.md",
        "decisions": docs_path / "decisions.md",
        "interfaces": docs_path / "interfaces.md",
    }
    outputs["experiments"].write_text(build_repo_experiments_markdown(sections), encoding="utf-8")
    outputs["decisions"].write_text(build_repo_decisions_markdown(sections), encoding="utf-8")
    outputs["interfaces"].write_text(build_repo_interfaces_markdown(sections), encoding="utf-8")
    return {name: str(path) for name, path in outputs.items()}
