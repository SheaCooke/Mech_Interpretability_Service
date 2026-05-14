import { BarChart3 } from "lucide-react";
import type { InferenceSummary } from "../types";

interface ClassStats {
  total: number;
  correct: number;
  accuracy: number;
}

interface Props {
  summary: InferenceSummary;
}

function sortedClassEntries(
  perClass: Record<string, ClassStats>
): [string, ClassStats][] {
  const entries = Object.entries(perClass) as [string, ClassStats][];
  const allNumeric = entries.every(([k]) => !isNaN(Number(k)));
  if (allNumeric) {
    return entries.sort(([a], [b]) => Number(a) - Number(b));
  }
  return entries.sort(([a], [b]) => a.localeCompare(b));
}

export default function SummaryPanel({ summary }: Props) {
  return (
    <div className="panel">
      <h3 className="panel-title">
        <BarChart3 size={14} /> Inference Summary
      </h3>

      <div className="kv-grid">
        <span className="kv-key">Records</span>
        <span className="kv-val">{summary.total_records.toLocaleString()}</span>

        {summary.has_labels && (
          <>
            <span className="kv-key">Accuracy</span>
            <span className="kv-val accent">
              {((summary.accuracy ?? 0) * 100).toFixed(2)}%
            </span>

            <span className="kv-key">Correct</span>
            <span className="kv-val">{summary.correct?.toLocaleString()}</span>

            <span className="kv-key">Incorrect</span>
            <span className="kv-val">{summary.incorrect?.toLocaleString()}</span>
          </>
        )}
      </div>

      {summary.per_class_accuracy && (
        <div className="class-grid">
          {sortedClassEntries(summary.per_class_accuracy).map(
            ([cls, stats]) => (
              <div key={cls} className="class-cell">
                {/* cls is always a string (JSON key) — display as-is.
                    For integer labels like 0/1/2 it shows "0"/"1"/"2";
                    for string labels like "setosa" it shows the name directly. */}
                <span className="class-label">{cls}</span>
                <div className="class-bar-track">
                  <div
                    className="class-bar-fill"
                    style={{ width: `${stats.accuracy * 100}%` }}
                  />
                </div>
                <span className="class-pct">
                  {(stats.accuracy * 100).toFixed(0)}%
                </span>
              </div>
            )
          )}
        </div>
      )}
    </div>
  );
}