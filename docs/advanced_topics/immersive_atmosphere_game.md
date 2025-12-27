# Immersive Atmospheric Circulation Game Specification

This document translates the WeatherFlow learning studio into a fully fledged
game experience that teaches the general circulation of the atmosphere through
immersive, highly realistic visuals. It emphasizes photorealistic clouds,
fluid camera work (pilot and satellite modes), intuitive visualization of
key variables, and synchronized cross-sections that make 3D spatial
relationships obvious. **All scientific formulations, reference cross-sections,
and conceptual depictions must align with the material in
Randall (2005) “Atmosphere, Clouds and Climate”**  
<https://sunandclimate.wordpress.com/wp-content/uploads/2009/05/randall.pdf>
is the authoritative source for equations, scaling, schematic layouts, and
variable relationships used in this game.

## Experience pillars

- **Photoreal volumetric clouds:** Physically inspired volumetric rendering
  (ray-marching with temporal reprojection, soft shadows, and multi-scatter
  approximations) tuned for “fly through” fidelity, with impostors for distant
  views.
- **Dual perspective mastery:** Seamless swap between pilot view (inside the
  atmosphere) and satellite view (planet-scale) with continuous time, lighting,
  and atmosphere scattering for horizon realism.
- **God-like inspection:** Players can pause, scrub time, and interrogate any
  field—temperature, moisture, winds (u/v/w), vorticity, PV, stability indices
  —without losing spatial context.
- **Synchronized orthogonal cross-sections:** Latitude–height and
  longitude–height planes are always linked, with a bright intersection tracer
  visible in both slices and in 3D.
- **Narrative progression through science:** Missions and unlocks map to key
  atmospheric concepts (Hadley/Ferrel/Polar cells, jets, fronts, PV thinking,
  ENSO), gradually introducing tooling and visualization depth.

## Cross-section system (core to spatial intuition)

- **Permanent twin slices:** One lat–height and one lon–height slice are shown
  side-by-side; both are mirrored as translucent planes in the 3D world.
- **Intersection tracer:** The line where the planes meet is rendered as a
  high-contrast emissive polyline (e.g., neon yellow) in both slices and in 3D.
  Flow arrows animate along it using local u/v/w to tie the views together.
- **Linked interactions:** Dragging or rotating a slice in 3D repositions its
  on-screen slice and updates the intersection on the companion slice.
  Hovering any point shows pins in both slices and on the globe, with readouts
  (lat/lon/height, T, q, u/v/w, PV, RH, vertical motion).
- **Shared scales:** Vertical axes are locked across slices; horizontal axes can
  be synchronized or decoupled for detail. Aspect ratio helpers prevent “flat”
  slices.
- **Cross-variable overlays:** Each slice supports dual fields (e.g., RH shading
  + temperature contours, or PV shading + wind barbs). Toggling “match other
  slice variables” harmonizes the stories across orientations.
- **Isentropic and terrain-aware slicing:** Optionally rotate a slice onto an
  isentropic surface; terrain and ocean intersection is shaded, with drop
  shadows on slopes for depth cues.
- **Preset ribbons:** Minimap ribbons show both slice lines on a world map;
  clicking the map repositions slices. Presets include “Jet & Tropopause,”
  “Front Cross,” “Tropical Cell,” and “Convective Column.”

## 3D volumetric context

- **Volumetric rendering:** Ray-marched cloud volumes with aerial perspective,
  god rays, and light-shaft scattering; distant clouds swap to billboards or
  volume impostors with temporal reprojection.
- **Isosurfaces and ribbons:** PV=2, RH=90%, or vorticity magnitude isosurfaces
  can be toggled; ribbons/streamtubes seeded along the intersection line trace
  3D flow and appear as projected vectors in both slices.
- **Satellite composites:** Globe overlays mimic VIS/IR/WV composites; slices
  show where cloud tops intersect, tying the orbital view to the cross-sections.

## Interaction and progression

- **Controls:** Free-flight (atmospheric and orbital), time dilation, freeze and
  inspect, and “focus follow” pins that jump between 3D and both slices.
- **Missions as learning beats:**
  - *Jet anatomy:* Lat–height through jet core and lon–height across a trough;
    highlight ageostrophic circulation loops and ascent/descent along the
    intersection.
  - *Front cross:* Align slices through a surface front; color vertical motion
    and moisture tilt to show the same ascent band in both slices.
  - *Tropical cell:* Capture Hadley ascent and subsidence flanks; streamlines
    converge on the intersection line.
  - *Convective column:* Drop sondes along the intersection; compare CAPE/CIN
    while tower growth animates in 3D.

## Visualization rules

- **Colormaps:** Harmonized but distinct per variable; consistent intersection
  color across all views. Colorblind-safe palettes are mandatory.
- **Depth cues:** Subtle parallax and lighting on slice planes; drop shadows
  where planes meet terrain/ocean; thin outlines to keep slices legible against
  bright cloud fields.
- **Temporal coherence:** Data updates tween during time scrubs; volumetric
  LODs ramp smoothly to avoid popping.

## Data and performance scaffolding

- **Field backbone:** GPU-side 3D textures for T, q, u, v, w, PV, RH, cloud
  phase; compute shaders generate slice textures on demand.
- **Level of detail:** Downsampled volumes during drag; refine after release.
  Asynchronous streaming for tiles; impostors for distant volumes; FSR/DLSS/TSR
  supported where applicable.
- **Planet-scale continuity:** Curved-Earth horizon, atmosphere scattering, and
  LOD-aware tiling to keep global-to-local zoom coherent.

## Testing and validation (keep the experience reliable)

- **Unit-level rendering tests:** Shader unit tests for slice sampling,
  intersection line stability, colormap application, and billboard/impostor
  swaps under different camera distances.
- **Interaction tests:** Automated scenarios for dragging/rotating slices,
  hover-to-pin synchronization, preset loading, and time scrubbing without
  popping.
- **Scientific consistency checks:** Golden-slice comparisons for canonical
  cases (Hadley cell, jet streak, frontal zone, moist convection) to verify
  that both slices and the 3D inset agree on ascent/descent, moisture, and wind
  structure.
- **Performance budgets:** Profiling thresholds for frame time (60–120 fps
  targets by platform), volumetric LOD shifts, and streaming latency.
- **Accessibility QA:** Automated contrast checks, colorblind theme snapshots,
  and configurable marker sizes/line thickness for intersection and probes.
- **Capture and share:** Built-in snapshot and clip capture for both slices and
  the 3D view to aid playtesting and regression tracking.
- **End-to-end playtest harness:** A scripted scenario that loads canonical
  Randall-aligned scenes (e.g., Hadley cell and jet streak cases), flies
  through the atmosphere, toggles cross-sections, and records screenshots/GIFs
  from both the pilot and satellite perspectives. Use these artifacts to
  confirm expected structures and to provide user-visible evidence that the
  experience is working as designed. A starter harness is scaffolded in
  `scripts/playtest_harness.py`; it enumerates Randall-sourced scenarios and
  builds CLI commands for capture once the game executable is available.

This specification should be treated as the source of truth for building the
immersive atmospheric circulation game layer on top of WeatherFlow. It centers
the player’s sense of spatial understanding—especially how orthogonal
cross-sections relate—while enforcing a rigorous, testable workflow so that
features remain verifiably correct as the game evolves.
