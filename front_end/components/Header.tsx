import { Trash2, ChevronRight } from "lucide-react";
import { STEP_ORDER, STEP_LABELS } from "../types";
import type { Step } from "../types";

interface Props {
  step: Step;
  sessionId: string | null;
  onReset: () => void;
}

const stepIdx: Record<Step, number> = {
  "upload-model":   0,
  "upload-dataset": 1,
  "inference":      2,
  "analysis":       3,
};

export default function Header({ step, sessionId, onReset }: Props) {
  return (
    <header className="header">
      <div className="header-inner">
        <div className="logo">
          <span className="logo-mark">NN</span>
          <span className="logo-text">Analyzer</span>
        </div>

        <nav className="steps">
          {STEP_ORDER.map((s, i) => (
            <div
              key={s}
              className={[
                "step",
                stepIdx[step] >= i ? "step-done" : "",
                step === s ? "step-active" : "",
              ].join(" ")}
            >
              <span className="step-num">{i + 1}</span>
              <span className="step-label">{STEP_LABELS[s]}</span>
              {i < STEP_ORDER.length - 1 && (
                <ChevronRight size={12} className="step-arrow" />
              )}
            </div>
          ))}
        </nav>

        {sessionId && (
          <button className="btn btn-ghost" onClick={onReset}>
            <Trash2 size={14} /> Reset
          </button>
        )}
      </div>
    </header>
  );
}