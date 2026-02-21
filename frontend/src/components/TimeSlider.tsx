import { useState, useEffect, useRef, useCallback } from 'react';

interface TimeSliderProps {
  value: number;
  max: number;
  onChange: (value: number) => void;
  label?: string;
  formatLabel?: (value: number) => string;
}

export default function TimeSlider({
  value,
  max,
  onChange,
  label = 'Forecast Hour',
  formatLabel,
}: TimeSliderProps) {
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(500); // ms per frame
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const step = useCallback(() => {
    onChange(value >= max ? 0 : value + 1);
  }, [value, max, onChange]);

  useEffect(() => {
    if (playing) {
      intervalRef.current = setInterval(step, speed);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [playing, speed, step]);

  const togglePlay = () => setPlaying((p) => !p);

  const stepBack = () => {
    setPlaying(false);
    onChange(Math.max(0, value - 1));
  };

  const stepForward = () => {
    setPlaying(false);
    onChange(Math.min(max, value + 1));
  };

  const cycleSpeed = () => {
    const speeds = [1000, 500, 250, 100];
    const idx = speeds.indexOf(speed);
    setSpeed(speeds[(idx + 1) % speeds.length]);
  };

  const displayValue = formatLabel ? formatLabel(value) : `${label}: ${value}`;

  return (
    <div className="time-slider">
      <div className="time-slider__controls">
        <button className="time-slider__btn" onClick={stepBack} title="Step back">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M10 2L4 7l6 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>

        <button className="time-slider__btn time-slider__btn--play" onClick={togglePlay} title={playing ? 'Pause' : 'Play'}>
          {playing ? (
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <rect x="3" y="2" width="3" height="10" rx="1" fill="currentColor"/>
              <rect x="8" y="2" width="3" height="10" rx="1" fill="currentColor"/>
            </svg>
          ) : (
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path d="M3 2l9 5-9 5V2z" fill="currentColor"/>
            </svg>
          )}
        </button>

        <button className="time-slider__btn" onClick={stepForward} title="Step forward">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M4 2l6 5-6 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>

        <button
          className="time-slider__btn time-slider__speed"
          onClick={cycleSpeed}
          title="Change speed"
        >
          {speed <= 100 ? '4x' : speed <= 250 ? '3x' : speed <= 500 ? '2x' : '1x'}
        </button>
      </div>

      <div className="time-slider__track-wrapper">
        <input
          type="range"
          className="time-slider__track"
          min={0}
          max={max}
          value={value}
          onChange={(e) => {
            setPlaying(false);
            onChange(Number(e.target.value));
          }}
        />
        <div
          className="time-slider__fill"
          style={{ width: max > 0 ? `${(value / max) * 100}%` : '0%' }}
        />
      </div>

      <span className="time-slider__label">{displayValue}</span>
    </div>
  );
}
