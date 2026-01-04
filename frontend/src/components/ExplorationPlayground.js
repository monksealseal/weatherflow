import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useMemo, useState } from 'react';
const conceptRoadmap = [
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
const probeOptions = [
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
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
function ExplorationPlayground({ result }) {
    const [flightMode, setFlightMode] = useState('atmosphere');
    const [timeScale, setTimeScale] = useState(1.25);
    const [freezeInspect, setFreezeInspect] = useState(false);
    const [probeStates, setProbeStates] = useState({
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
    const conceptProgress = useMemo(() => conceptRoadmap.map((concept, index) => {
        const progress = clamp((normalizedSkill + index * 0.07) * 100, 30, 100);
        const unlocked = progress >= 45 + index * 4;
        return { ...concept, progress: Math.round(progress), unlocked };
    }), [normalizedSkill]);
    const achievements = useMemo(() => {
        const activeProbes = Object.values(probeStates).filter(Boolean).length;
        const freezeBonus = freezeInspect ? 0.1 : 0;
        return achievementTemplates.map((achievement, index) => {
            const base = normalizedSkill + freezeBonus + activeProbes * 0.03;
            const adjusted = clamp(base + index * 0.05, 0.25, 1);
            return { ...achievement, progress: Math.round(adjusted * 100) };
        });
    }, [freezeInspect, normalizedSkill, probeStates]);
    const toggleProbe = (key) => {
        setProbeStates((current) => ({ ...current, [key]: !current[key] }));
    };
    const activeProbeCount = Object.values(probeStates).filter(Boolean).length;
    const flightLabel = flightMode === 'orbital'
        ? 'Orbital free-flight: horizon-locked cameras and re-entry windows'
        : 'Atmospheric free-flight: terrain-following autopilot and gust locks';
    return (_jsxs("section", { className: "section-card experience-card", children: [_jsxs("div", { className: "section-heading", children: [_jsxs("div", { children: [_jsx("h2", { children: "Immersive mission prototyping" }), _jsx("p", { children: "Prototype free-flight controls, time dilation, freeze/inspect workflows, probes, and achievement scaffolding alongside the existing experiment results." })] }), _jsxs("div", { className: "accent-pill", children: [flightMode === 'orbital' ? 'Orbital' : 'Atmosphere', " mode \u2022 ", missionHorizon, "\u00D7 timeline"] })] }), _jsxs("div", { className: "playground-grid", children: [_jsxs("div", { className: "experience-panel", children: [_jsxs("div", { className: "panel-heading", children: [_jsx("h3", { children: "Free-flight lab" }), _jsx("span", { className: "hint-text", children: flightLabel })] }), _jsxs("div", { className: "mode-toggle", children: [_jsx("button", { type: "button", className: `mode-pill ${flightMode === 'atmosphere' ? 'active' : ''}`, onClick: () => setFlightMode('atmosphere'), children: "Atmospheric" }), _jsx("button", { type: "button", className: `mode-pill ${flightMode === 'orbital' ? 'active' : ''}`, onClick: () => setFlightMode('orbital'), children: "Orbital" })] }), _jsxs("label", { className: "control-slider", children: [_jsxs("div", { children: [_jsx("strong", { children: "Time dilation" }), _jsx("p", { className: "hint-text", children: "Stretch or compress mission playback to keep diagnostics legible." })] }), _jsxs("div", { className: "slider-stack", children: [_jsx("input", { type: "range", min: 0.5, max: 3, step: 0.05, value: timeScale, onChange: (event) => setTimeScale(Number(event.target.value)) }), _jsxs("span", { className: "slider-value", children: [timeScale.toFixed(2), "\u00D7"] })] })] }), _jsxs("label", { className: "toggle-row", children: [_jsx("input", { type: "checkbox", checked: freezeInspect, onChange: (event) => setFreezeInspect(event.target.checked) }), _jsxs("div", { children: [_jsx("strong", { children: "Freeze & inspect" }), _jsx("p", { className: "hint-text", children: "Pause at any timestep to scrub PV and thermal fields; pin cross-sections during inspection." })] })] })] }), _jsxs("div", { className: "experience-panel", children: [_jsxs("div", { className: "panel-heading", children: [_jsx("h3", { children: "Concept roadmap & unlocks" }), _jsx("span", { className: "hint-text", children: "Progression through dynamical concepts & missions." })] }), _jsx("div", { className: "progress-list", children: conceptProgress.map((concept) => (_jsxs("div", { className: "progress-row", children: [_jsxs("div", { className: "progress-header", children: [_jsxs("div", { children: [_jsx("strong", { children: concept.title }), _jsx("p", { className: "hint-text", children: concept.description })] }), _jsx("span", { className: `unlock-pill ${concept.unlocked ? 'unlocked' : ''}`, children: concept.unlocked ? 'Unlocked' : 'Locked' })] }), _jsx("div", { className: "progress-track", children: _jsx("div", { className: "progress-fill", style: { width: `${concept.progress}%` } }) }), _jsxs("p", { className: "unlock-text", children: ["Unlocks: ", concept.unlocks] })] }, concept.title))) })] }), _jsxs("div", { className: "experience-panel", children: [_jsxs("div", { className: "panel-heading", children: [_jsx("h3", { children: "Interactive probes & overlays" }), _jsx("span", { className: "hint-text", children: "Toggle sondes, cross-sections, streamline seeds, and tracer overlays." })] }), _jsx("ul", { className: "probe-grid", children: probeOptions.map((probe) => (_jsxs("li", { className: `probe-card ${probeStates[probe.key] ? 'active' : ''}`, "data-testid": `probe-${probe.key}`, children: [_jsxs("div", { className: "probe-header", children: [_jsxs("label", { className: "probe-toggle", children: [_jsx("input", { type: "checkbox", checked: probeStates[probe.key], onChange: () => toggleProbe(probe.key) }), _jsx("strong", { children: probe.title })] }), _jsx("span", { className: "accent-badge", children: probe.uiHint })] }), _jsx("p", { className: "hint-text", children: probe.description })] }, probe.key))) }), _jsxs("p", { className: "hint-text", children: ["Active overlays: ", activeProbeCount, " \u2022 Time-dilated playback spans ", missionHorizon, " model units."] })] }), _jsxs("div", { className: "experience-panel", children: [_jsxs("div", { className: "panel-heading", children: [_jsx("h3", { children: "Narrative achievements" }), _jsx("span", { className: "hint-text", children: "Reward PV thinking, thermo diagrams, and vorticity budgets as learners progress." })] }), _jsx("div", { className: "progress-list", children: achievements.map((achievement) => (_jsxs("div", { className: "progress-row", children: [_jsxs("div", { className: "progress-header", children: [_jsxs("div", { children: [_jsx("strong", { children: achievement.title }), _jsx("p", { className: "hint-text", children: achievement.description })] }), _jsxs("span", { className: "slider-value", children: [achievement.progress, "%"] })] }), _jsx("div", { className: "progress-track compact", children: _jsx("div", { className: "progress-fill accent", style: { width: `${achievement.progress}%` } }) })] }, achievement.title))) })] })] })] }));
}
export default ExplorationPlayground;
