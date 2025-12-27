# Game Playtest Harness (Randall-Aligned)

This guide describes how to exercise the immersive atmospheric game via an
automated playtest harness. It is aligned with **Randall (2005) “Atmosphere,
Clouds and Climate”** (<https://sunandclimate.wordpress.com/wp-content/uploads/2009/05/randall.pdf>),
which is the authoritative source for formulas, cross-sections, and schematics.

## What the harness does

- Enumerates canonical scenarios (Hadley cell, jet streak, frontal cross,
  convective column) tied to Randall chapters.
- Builds CLI commands for both pilot and satellite perspectives plus the
  synchronized lat–height and lon–height cross-sections.
- Captures screenshots/GIFs per scenario for regression review and user-visible
  evidence.
- Defaults to dry-run mode so it can be used before a game binary exists; set
  `--execute` once the CLI is available.

## Prerequisites

- A game CLI/executable that accepts scenario identifiers, capture directory,
  and capture modes (pilot, satellite, lat-height, lon-height).
- Installed Python 3.8+.
- Storage for artifacts (defaults to `playtest_artifacts/`).

## Running the harness

From the repo root:

```bash
python scripts/playtest_harness.py \
  --game-cli /path/to/game_binary \
  --output-dir playtest_artifacts \
  --scenarios all \
  --execute
```

Dry-run (command preview only):

```bash
python scripts/playtest_harness.py --game-cli /path/to/game_binary
```

Select specific scenarios:

```bash
python scripts/playtest_harness.py \
  --game-cli /path/to/game_binary \
  --scenarios hadley_cell jet_streak \
  --execute
```

## Scenarios (all derived from Randall)

- **hadley_cell** — Randall Ch. 6. Lat–height through ITCZ ascent plus
  lon–height showing subsidence flanks; PV + RH overlays and intersection tracer
  clarity.
- **jet_streak** — Randall Ch. 7. Upper-level jet streak with ascent/descent
  couplets and tropopause fold along the intersection line.
- **frontal_cross** — Randall Ch. 8. Surface front with moisture tilt; ascent
  band visible in both slices and the 3D inset; T contours over RH shading.
- **convective_column** — Randall Ch. 4–5. Deep convection tower growth and
  anvils; CAPE/CIN comparisons along the intersection line and updraft
  visualization.

## Artifacts to collect

- Pilot view screenshots/GIFs during flight through clouds and cross-section
  planes.
- Satellite view composites with slice planes visible.
- Lat–height and lon–height slice captures with intersection tracer, overlays,
  and annotations enabled.
- Logs of commands executed and any warnings or errors.

## Validation checklist

- Intersection tracer appears identically in both slices and the 3D inset.
- Cross-variable overlays match Randall schematics (temperature, moisture,
  winds, PV).
- Expected circulation signatures are visible (Hadley overturning, jet
  couplets, frontal ascent, convective updrafts).
- Capture modes produce files in the scenario-specific directories under
  `playtest_artifacts/`.

## Extending the harness

- Add new scenarios to `scripts/playtest_harness.py` with `randall_refs` and
  capture modes.
- Integrate GPU frame-time and streaming latency logs from the game CLI to
  enforce performance budgets.
- Add golden-image comparison for key slices and perspectives once captured
  baselines exist.
