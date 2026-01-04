export function formatSeconds(seconds) {
    if (Number.isNaN(seconds) || !Number.isFinite(seconds)) {
        return 'n/a';
    }
    if (seconds < 1) {
        return `${(seconds * 1000).toFixed(0)} ms`;
    }
    if (seconds < 60) {
        return `${seconds.toFixed(1)} s`;
    }
    const minutes = Math.floor(seconds / 60);
    const remainder = seconds % 60;
    return `${minutes}m ${remainder.toFixed(0)}s`;
}
