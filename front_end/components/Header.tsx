import { Trash2, ChevronRight, Home, BookOpen, BarChart2 } from "lucide-react";
import { STEP_ORDER, STEP_LABELS } from "../types";
import type { Step, Page } from "../types";

interface Props {
  step: Step;
  sessionId: string | null;
  currentPage: Page;
  onNavigate: (page: Page) => void;
  onReset: () => void;
}

const stepIdx: Record<Step, number> = {
  "upload-model":   0,
  "upload-dataset": 1,
  "inference":      2,
  "analysis":       3,
};

export default function Header({ step, sessionId, currentPage, onNavigate, onReset }: Props) {
  return (
    <header className="header">
      <div className="header-inner">

        {/* Logo — always navigates home */}
        <button className="logo logo-btn" onClick={() => onNavigate("home")}>
          <span className="logo-mark">NN</span>
          <span className="logo-text">Analyzer</span>
        </button>

        {/* Step progress — only shown on home page */}
        {currentPage === "home" && (
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
        )}

        {/* Spacer when not on home */}
        {currentPage !== "home" && <div style={{ flex: 1 }} />}

        {/* Page nav links */}
        <nav className="page-nav">
          <button
            className={`page-nav-btn ${currentPage === "home" ? "page-nav-active" : ""}`}
            onClick={() => onNavigate("home")}
          >
            <Home size={13} /> Home
          </button>
          <button
            className={`page-nav-btn ${currentPage === "instructions" ? "page-nav-active" : ""}`}
            onClick={() => onNavigate("instructions")}
          >
            <BookOpen size={13} /> Instructions
          </button>
          <button
            className={`page-nav-btn ${currentPage === "interpreting" ? "page-nav-active" : ""}`}
            onClick={() => onNavigate("interpreting")}
          >
            <BarChart2 size={13} /> Interpreting Results
          </button>
        </nav>

        {/* Reset — only shown when a session is active */}
        {sessionId && currentPage === "home" && (
          <button className="btn btn-ghost" onClick={onReset}>
            <Trash2 size={14} /> Reset
          </button>
        )}
      </div>
    </header>
  );
}