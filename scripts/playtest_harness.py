"""
Playtest harness scaffolding for the immersive atmospheric game.

This script is designed to orchestrate Randall-aligned scenarios once a
runnable game binary or CLI is available. It supports dry-run output for
environments where the game executable is not yet present.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class Scenario:
    """A Randall-aligned playtest scenario definition."""

    key: str
    title: str
    description: str
    randall_refs: List[str]
    capture_modes: List[str]
    notes: str = ""


SCENARIOS: List[Scenario] = [
    Scenario(
        key="hadley_cell",
        title="Hadley Cell and ITCZ",
        description=(
            "Lat–height slice through the ITCZ with accompanying lon–height slice "
            "showing subsidence flanks; verify overturning circulation and moisture tilt."
        ),
        randall_refs=["Ch. 6, Walker and Hadley circulations"],
        capture_modes=["pilot", "satellite", "lat-height", "lon-height"],
        notes="Use PV and RH overlays; capture intersection tracer clarity.",
    ),
    Scenario(
        key="jet_streak",
        title="Jet Streak and Tropopause Fold",
        description=(
            "Cross-sections through an upper-level jet streak; highlight ascent/descent "
            "couplets and tropopause structure along the intersection line."
        ),
        randall_refs=["Ch. 7, Midlatitude dynamics"],
        capture_modes=["pilot", "satellite", "lat-height", "lon-height"],
        notes="Show ageostrophic circulation loop on both slices.",
    ),
    Scenario(
        key="frontal_cross",
        title="Frontal Ascent Band",
        description=(
            "Surface front with moisture tilt; ensure the same ascent band is visible in "
            "both orthogonal slices and in the 3D inset."
        ),
        randall_refs=["Ch. 8, Fronts and baroclinic zones"],
        capture_modes=["pilot", "satellite", "lat-height", "lon-height"],
        notes="Verify cross-variable overlays (T contours over RH shading).",
    ),
    Scenario(
        key="convective_column",
        title="Moist Convective Tower",
        description=(
            "Localized deep convection; capture tower growth and anvils while comparing "
            "CAPE/CIN along the intersection line."
        ),
        randall_refs=["Ch. 4–5, Convection and clouds"],
        capture_modes=["pilot", "satellite", "lat-height", "lon-height"],
        notes="Record updraft visualization in both slices with intersection tracer.",
    ),
]


def build_command(game_cli: Path, scenario: Scenario, output_dir: Path, run: bool) -> List[str]:
    """
    Construct the CLI invocation for the given scenario.

    The concrete flags must match the eventual game binary. This placeholder
    uses generic options for clarity.
    """
    return [
        str(game_cli),
        "--scenario",
        scenario.key,
        "--capture-dir",
        str(output_dir / scenario.key),
        "--capture-modes",
        ",".join(scenario.capture_modes),
        "--randall-source",
        "https://sunandclimate.wordpress.com/wp-content/uploads/2009/05/randall.pdf",
        *(["--run"] if run else ["--dry-run"]),
    ]


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_scenarios(selection: Iterable[str]) -> List[Scenario]:
    if not selection or "all" in selection:
        return SCENARIOS
    selected_keys = set(selection)
    resolved = [s for s in SCENARIOS if s.key in selected_keys]
    missing = selected_keys.difference({s.key for s in resolved})
    if missing:
        raise ValueError(f"Unknown scenario keys: {', '.join(sorted(missing))}")
    return resolved


def run_harness(game_cli: Path, scenarios: List[Scenario], output_dir: Path, execute: bool) -> int:
    ensure_output_dir(output_dir)
    for scenario in scenarios:
        ensure_output_dir(output_dir / scenario.key)
        command = build_command(game_cli, scenario, output_dir, execute)
        print(f"\n[PLAYTEST] {scenario.title}")
        print(f"Description : {scenario.description}")
        print(f"Randall refs: {', '.join(scenario.randall_refs)}")
        print(f"Notes       : {scenario.notes}")
        print(f"Command     : {' '.join(command)}")
        if execute:
            try:
                subprocess.run(command, check=True)
            except FileNotFoundError:
                print(f"ERROR: Game CLI not found at {game_cli}.", file=sys.stderr)
                return 1
            except subprocess.CalledProcessError as exc:
                print(f"ERROR: Command failed with exit code {exc.returncode}.", file=sys.stderr)
                return exc.returncode
    return 0


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run or preview Randall-aligned playtest scenarios for the immersive "
            "atmospheric game."
        )
    )
    parser.add_argument(
        "--game-cli",
        type=Path,
        required=True,
        help="Path to the game CLI or executable that accepts scenario arguments.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("playtest_artifacts"),
        help="Directory to store captured screenshots/GIFs.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["all"],
        help="Scenario keys to run (default: all). Options: all, "
        + ", ".join(sorted(s.key for s in SCENARIOS)),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the game CLI instead of printing dry-run commands.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        scenarios = load_scenarios(args.scenarios)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return run_harness(args.game_cli, scenarios, args.output_dir, args.execute)


if __name__ == "__main__":
    raise SystemExit(main())
