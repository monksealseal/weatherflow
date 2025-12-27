import { useMemo, useState } from 'react';
import { ExperimentResult } from '../api/types';

interface Props {
  result: ExperimentResult | null;
}

type FlightMode = 'atmosphere' | 'orbital';

interface ConceptNode {
  title: string;
  description: string;
  unlocks: string;
}

interface ProbeOption {
  key: string;
  title: string;
  description: string;
  uiHint: string;
}

interface Achievement {
  title: string;
  description: string;
  progress: number;
}

const conceptRoadmap: ConceptNode[] = [
  {
    title: 'Hadley / Ferrel / Polar cells',
    description: 'Track overturning strength and angular momentum transport across cells.',
    unlocks: 'Mass streamfunction ribbons & overturning diagnostics'
  },
  {
    title: 'Jet streams & streaks',
    description: 'Stitch together upper-level streaks with PV filaments to steer free-flight.',
    unlocks: 'Jet entrance/exit overlays with automated streak seeding'
  },
  {
    title: 'Fronts & baroclinic zones',
    description: 'Blend thermal wind balance checks with cross-sections during freeze/inspect.',
    unlocks: 'Dynamic cross-section scaffold & frontogenesis heatmaps'
  },
  {
    title: 'Cyclogenesis',
    description: 'Tie vorticity budgets to surface pressure deepening along the mission track.',
    unlocks: 'Surface pressure tendency probes with vort max alerts'
  },
  {
    title: 'ENSO teleconnections',
    description: 'Overlay tropical forcing with jet displacement to unlock orbital re-entries.',
    unlocks: 'Walker/ENSO anomaly curtains with jet response templates'
  }
];

const probeOptions: ProbeOption[] = [
  {
    key: 'sondes',
    title: 'Drop sondes',
    description: 'Drop sondes along the track to log vertical profiles and moisture jumps.',
    uiHint: 'Long-press to seed a burst; drag to stack along the jet core.'
  },
  {
    key: 'crossSections',
    title: 'Cross-sections',
    description: 'Pull draggable cross-sections that respect the current flight orientation.',
    uiHint: 'Shift + drag to pivot the slice; double-click to pin during freeze.'
  },
  {
    key: 'streamlines',
    title: 'Streamline seeds',
    description: 'Inject streamline seeds in shear zones to watch jet coupling and exit dynamics.',
    uiHint: 'Hover to preview pathlines; click to bake them into the overlay.'
  },
  {
    key: 'tracers',
    title: 'Tracer overlays',
    description: 'Shade PV or theta-e curtains to expose tropopause folds and frontal slopes.',
    uiHint: 'Toggle opacity with the inspect scrubber for quick comparisons.'
  }
];

const achievementTemplates = [
  {
    title: 'PV thinker',
    description: 'Tie free-flight maneuvers to PV inversion cues and filament strength.'
  },
  {
    title: 'Thermodynamic cartographer',
    description: 'Diagnose stability with thermo diagrams while riding time-dilated orbits.'
  },
  {
    title: 'Vorticity budgeteer',
    description: 'Close column budgets and reward balanced trajectories through storms.'
  }
];

const clamp = (value: number, min: number, max: number): number => Math.min(max, Math.max(min, value));

function ExplorationPlayground({ result }: Props): JSX.Element {
  const [flightMode, setFlightMode] = useState<FlightMode>('atmosphere');
  const [timeScale, setTimeScale] = useState(1.25);
  const [freezeInspect, setFreezeInspect] = useState(false);
  const [probeStates, setProbeStates] = useState<Record<string, boolean>>({
    sondes: true,
    crossSections: true,
    streamlines: false,
    tracers: false
  });

  const lastTime = result?.prediction.times.at(-1) ?? 1;
  const missionHorizon = useMemo(() => {
    const dilated = lastTime * timeScale;
    return Math.round(dilated * 10) / 10;
  }, [lastTime, timeScale]);

  const normalizedSkill = useMemo(() => {
    const validationMetrics = result?.validation.metrics ?? [];
    const lastValidation = validationMetrics[validationMetrics.length - 1];
    const valLoss = lastValidation?.valLoss ?? null;
    const baseSkill = valLoss !== null ? clamp(1 / (1 + valLoss), 0.25, 0.95) : 0.45;
    const epochBonus = clamp((result?.config.training.epochs ?? 0) / 30, 0, 0.2);
    return clamp(baseSkill + epochBonus, 0.25, 0.99);
  }, [result]);

  const conceptProgress = useMemo(
    () =>
      conceptRoadmap.map((concept, index) => {
        const progress = clamp((normalizedSkill + index * 0.07) * 100, 30, 100);
        const unlocked = progress >= 45 + index * 4;
        return { ...concept, progress: Math.round(progress), unlocked };
      }),
    [normalizedSkill]
  );

  const achievements: Achievement[] = useMemo(() => {
    const activeProbes = Object.values(probeStates).filter(Boolean).length;
    const freezeBonus = freezeInspect ? 0.1 : 0;
    return achievementTemplates.map((achievement, index) => {
      const base = normalizedSkill + freezeBonus + activeProbes * 0.03;
      const adjusted = clamp(base + index * 0.05, 0.25, 1);
      return { ...achievement, progress: Math.round(adjusted * 100) };
    });
  }, [freezeInspect, normalizedSkill, probeStates]);

  const toggleProbe = (key: string) => {
    setProbeStates((current) => ({ ...current, [key]: !current[key] }));
  };

  const activeProbeCount = Object.values(probeStates).filter(Boolean).length;
  const flightLabel =
    flightMode === 'orbital'
      ? 'Orbital free-flight: horizon-locked cameras and re-entry windows'
      : 'Atmospheric free-flight: terrain-following autopilot and gust locks';

  return (
    <section className="section-card experience-card">
      <div className="section-heading">
        <div>
          <h2>Immersive mission prototyping</h2>
          <p>
            Prototype free-flight controls, time dilation, freeze/inspect workflows, probes, and
            achievement scaffolding alongside the existing experiment results.
          </p>
        </div>
        <div className="accent-pill">
          {flightMode === 'orbital' ? 'Orbital' : 'Atmosphere'} mode • {missionHorizon}× timeline
        </div>
      </div>

      <div className="playground-grid">
        <div className="experience-panel">
          <div className="panel-heading">
            <h3>Free-flight lab</h3>
            <span className="hint-text">{flightLabel}</span>
          </div>
          <div className="mode-toggle">
            <button
              type="button"
              className={`mode-pill ${flightMode === 'atmosphere' ? 'active' : ''}`}
              onClick={() => setFlightMode('atmosphere')}
            >
              Atmospheric
            </button>
            <button
              type="button"
              className={`mode-pill ${flightMode === 'orbital' ? 'active' : ''}`}
              onClick={() => setFlightMode('orbital')}
            >
              Orbital
            </button>
          </div>

          <label className="control-slider">
            <div>
              <strong>Time dilation</strong>
              <p className="hint-text">
                Stretch or compress mission playback to keep diagnostics legible.
              </p>
            </div>
            <div className="slider-stack">
              <input
                type="range"
                min={0.5}
                max={3}
                step={0.05}
                value={timeScale}
                onChange={(event) => setTimeScale(Number(event.target.value))}
              />
              <span className="slider-value">{timeScale.toFixed(2)}×</span>
            </div>
          </label>

          <label className="toggle-row">
            <input
              type="checkbox"
              checked={freezeInspect}
              onChange={(event) => setFreezeInspect(event.target.checked)}
            />
            <div>
              <strong>Freeze &amp; inspect</strong>
              <p className="hint-text">
                Pause at any timestep to scrub PV and thermal fields; pin cross-sections during
                inspection.
              </p>
            </div>
          </label>
        </div>

        <div className="experience-panel">
          <div className="panel-heading">
            <h3>Concept roadmap &amp; unlocks</h3>
            <span className="hint-text">Progression through dynamical concepts &amp; missions.</span>
          </div>
          <div className="progress-list">
            {conceptProgress.map((concept) => (
              <div key={concept.title} className="progress-row">
                <div className="progress-header">
                  <div>
                    <strong>{concept.title}</strong>
                    <p className="hint-text">{concept.description}</p>
                  </div>
                  <span className={`unlock-pill ${concept.unlocked ? 'unlocked' : ''}`}>
                    {concept.unlocked ? 'Unlocked' : 'Locked'}
                  </span>
                </div>
                <div className="progress-track">
                  <div className="progress-fill" style={{ width: `${concept.progress}%` }} />
                </div>
                <p className="unlock-text">Unlocks: {concept.unlocks}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="experience-panel">
          <div className="panel-heading">
            <h3>Interactive probes &amp; overlays</h3>
            <span className="hint-text">
              Toggle sondes, cross-sections, streamline seeds, and tracer overlays.
            </span>
          </div>
          <ul className="probe-grid">
            {probeOptions.map((probe) => (
              <li
                key={probe.key}
                className={`probe-card ${probeStates[probe.key] ? 'active' : ''}`}
                data-testid={`probe-${probe.key}`}
              >
                <div className="probe-header">
                  <label className="probe-toggle">
                    <input
                      type="checkbox"
                      checked={probeStates[probe.key]}
                      onChange={() => toggleProbe(probe.key)}
                    />
                    <strong>{probe.title}</strong>
                  </label>
                  <span className="accent-badge">{probe.uiHint}</span>
                </div>
                <p className="hint-text">{probe.description}</p>
              </li>
            ))}
          </ul>
          <p className="hint-text">
            Active overlays: {activeProbeCount} • Time-dilated playback spans {missionHorizon} model
            units.
          </p>
        </div>

        <div className="experience-panel">
          <div className="panel-heading">
            <h3>Narrative achievements</h3>
            <span className="hint-text">
              Reward PV thinking, thermo diagrams, and vorticity budgets as learners progress.
            </span>
          </div>
          <div className="progress-list">
            {achievements.map((achievement) => (
              <div key={achievement.title} className="progress-row">
                <div className="progress-header">
                  <div>
                    <strong>{achievement.title}</strong>
                    <p className="hint-text">{achievement.description}</p>
                  </div>
                  <span className="slider-value">{achievement.progress}%</span>
                </div>
                <div className="progress-track compact">
                  <div className="progress-fill accent" style={{ width: `${achievement.progress}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

export default ExplorationPlayground;
