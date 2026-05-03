import { useRef } from "react";

interface Props {
  min: number;
  max: number;
  step: number;
  valueLow: number;
  valueHigh: number;
  disabled?: boolean;
  onChange: (low: number, high: number) => void;
}

export default function RangeSlider({
  min, max, step,
  valueLow, valueHigh,
  disabled = false,
  onChange,
}: Props) {
  const trackRef = useRef<HTMLDivElement>(null);

  // Convert a value to a percentage position along the track
  function pct(v: number) {
    return ((v - min) / (max - min)) * 100;
  }

  function handleLow(e: React.ChangeEvent<HTMLInputElement>) {
    const next = parseFloat(e.target.value);
    // Low thumb must not exceed high thumb
    onChange(Math.min(next, valueHigh - step), valueHigh);
  }

  function handleHigh(e: React.ChangeEvent<HTMLInputElement>) {
    const next = parseFloat(e.target.value);
    // High thumb must not go below low thumb
    onChange(valueLow, Math.max(next, valueLow + step));
  }

  const lowPct  = pct(valueLow);
  const highPct = pct(valueHigh);

  return (
    <div
      className={`range-slider-root ${disabled ? "range-slider-disabled" : ""}`}
      ref={trackRef}
    >
      {/* Track background */}
      <div className="range-slider-track">
        {/* Filled region between the two thumbs */}
        <div
          className="range-slider-fill"
          style={{
            left:  `${lowPct}%`,
            width: `${highPct - lowPct}%`,
          }}
        />
      </div>

      {/* Low thumb input */}
      <input
        type="range"
        className="range-slider-input range-slider-low"
        min={min}
        max={max}
        step={step}
        value={valueLow}
        disabled={disabled}
        onChange={handleLow}
      />

      {/* High thumb input */}
      <input
        type="range"
        className="range-slider-input range-slider-high"
        min={min}
        max={max}
        step={step}
        value={valueHigh}
        disabled={disabled}
        onChange={handleHigh}
      />
    </div>
  );
}