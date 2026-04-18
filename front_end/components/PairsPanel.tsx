import { GitBranch } from "lucide-react";
import type { SimilarPair } from "../types";

const MAX_DISPLAY = 200;

interface Props {
  pairs: SimilarPair[];
}

export default function PairsPanel({ pairs }: Props) {
  return (
    <div className="panel">
      <h3 className="panel-title">
        <GitBranch size={14} /> Similar Activation Pairs ({pairs.length})
      </h3>

      {pairs.length === 0 ? (
        <p className="empty-msg">No pairs found below threshold.</p>
      ) : (
        <div className="pairs-table">
          <div className="pairs-header">
            <span>Record A</span>
            <span>Record B</span>
            <span>Label A</span>
            <span>Label B</span>
            <span>Distance</span>
          </div>

          {pairs.slice(0, MAX_DISPLAY).map((p, i) => (
            <div
              key={i}
              className={`pairs-row ${p.label_a !== p.label_b ? "pairs-mismatch" : ""}`}
            >
              <span className="mono">{p.id_a}</span>
              <span className="mono">{p.id_b}</span>
              <span>{p.label_a ?? "—"}</span>
              <span>{p.label_b ?? "—"}</span>
              <span className="dist">{p.distance.toFixed(4)}</span>
            </div>
          ))}

          {pairs.length > MAX_DISPLAY && (
            <p className="pairs-overflow">
              Showing first {MAX_DISPLAY} of {pairs.length} pairs.
            </p>
          )}
        </div>
      )}
    </div>
  );
}