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
          {(Object.entries(summary.per_class_accuracy) as [string, ClassStats][]).map(
            ([cls, stats]) => (
              <div key={cls} className="class-cell">
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