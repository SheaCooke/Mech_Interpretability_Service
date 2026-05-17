import { useState, useMemo } from "react";
import { GitBranch, ChevronLeft, ChevronRight, Search, X } from "lucide-react";
import type { SimilarPair } from "../types";

const PAGE_SIZE = 50;

interface Props {
  pairs: SimilarPair[];
}

export default function PairsPanel({ pairs }: Props) {
  const [page,   setPage]   = useState(0);
  const [filter, setFilter] = useState("");

  // Filter pairs where the search term appears in either record id
  const filtered = useMemo(() => {
    const term = filter.trim().toLowerCase();
    if (!term) return pairs;
    return pairs.filter(
      p =>
        p.id_a.toLowerCase().includes(term) ||
        p.id_b.toLowerCase().includes(term)
    );
  }, [pairs, filter]);

  // Reset to page 0 whenever the filter changes
  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const safePage   = Math.min(page, totalPages - 1);
  const pageStart  = safePage * PAGE_SIZE;
  const pageRows   = filtered.slice(pageStart, pageStart + PAGE_SIZE);

  function handleFilterChange(val: string) {
    setFilter(val);
    setPage(0);
  }

  return (
    <div className="panel pairs-panel">
      {/* Header */}
      <div className="pairs-panel-header">
        <h3 className="panel-title">
          <GitBranch size={14} />
          Similar Activation Pairs
          <span className="pairs-count">
            {filtered.length !== pairs.length
              ? `${filtered.length} of ${pairs.length}`
              : pairs.length}
          </span>
        </h3>

        {/* Search input */}
        <div className="pairs-search-wrap">
          <Search size={12} className="pairs-search-icon" />
          <input
            className="pairs-search-input"
            placeholder="Filter by record id…"
            value={filter}
            onChange={e => handleFilterChange(e.target.value)}
          />
          {filter && (
            <button
              className="pairs-search-clear"
              onClick={() => handleFilterChange("")}
              aria-label="Clear filter"
            >
              <X size={11} />
            </button>
          )}
        </div>
      </div>

      {filtered.length === 0 ? (
        <p className="empty-msg">
          {pairs.length === 0
            ? "No pairs found in the selected range."
            : `No pairs match "${filter}".`}
        </p>
      ) : (
        <>
          {/* Scrollable table */}
          <div className="pairs-scroll-container">
            <div className="pairs-table">
              <div className="pairs-header">
                <span>Record A</span>
                <span>Record B</span>
                <span>Label A</span>
                <span>Label B</span>
                <span>Distance</span>
              </div>

              {pageRows.map((p, i) => (
                <div
                  key={pageStart + i}
                  className={`pairs-row ${p.label_a !== p.label_b ? "pairs-mismatch" : ""}`}
                >
                  <span className="mono pairs-id">
                    {highlightMatch(p.id_a, filter)}
                  </span>
                  <span className="mono pairs-id">
                    {highlightMatch(p.id_b, filter)}
                  </span>
                  <span>{p.label_a ?? "—"}</span>
                  <span>{p.label_b ?? "—"}</span>
                  <span className="dist">{p.distance.toFixed(4)}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="pairs-pagination">
              <button
                className="pairs-page-btn"
                onClick={() => setPage(p => Math.max(0, p - 1))}
                disabled={safePage === 0}
                aria-label="Previous page"
              >
                <ChevronLeft size={13} />
              </button>

              {/* Page number pills */}
              <div className="pairs-page-pills">
                {buildPageRange(safePage, totalPages).map((item, i) =>
                  item === "…" ? (
                    <span key={`ellipsis-${i}`} className="pairs-page-ellipsis">…</span>
                  ) : (
                    <button
                      key={item}
                      className={`pairs-page-pill ${item === safePage ? "pairs-page-pill-active" : ""}`}
                      onClick={() => setPage(item as number)}
                    >
                      {(item as number) + 1}
                    </button>
                  )
                )}
              </div>

              <button
                className="pairs-page-btn"
                onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
                disabled={safePage === totalPages - 1}
                aria-label="Next page"
              >
                <ChevronRight size={13} />
              </button>

              <span className="pairs-page-info">
                {pageStart + 1}–{Math.min(pageStart + PAGE_SIZE, filtered.length)}{" "}
                of {filtered.length}
              </span>
            </div>
          )}
        </>
      )}
    </div>
  );
}

/**
 * Highlight the matched substring within a record id string.
 * Returns either a plain string or a JSX element with a highlighted span.
 */
function highlightMatch(text: string, term: string): React.ReactNode {
  if (!term.trim()) return text;
  const idx = text.toLowerCase().indexOf(term.toLowerCase());
  if (idx === -1) return text;
  return (
    <>
      {text.slice(0, idx)}
      <mark className="pairs-highlight">{text.slice(idx, idx + term.length)}</mark>
      {text.slice(idx + term.length)}
    </>
  );
}

/**
 * Build a compact page number range to display in the pagination bar.
 * Always shows first, last, current, and the pages immediately adjacent
 * to the current page. Gaps are represented by "…".
 *
 * e.g. for page 5 of 20: [0, "…", 4, 5, 6, "…", 19]
 */
function buildPageRange(
  current: number,
  total: number
): (number | "…")[] {
  if (total <= 7) {
    return Array.from({ length: total }, (_, i) => i);
  }

  const always = new Set([0, total - 1, current, current - 1, current + 1]);
  const pages  = Array.from({ length: total }, (_, i) => i)
    .filter(i => always.has(i) && i >= 0 && i < total);

  const result: (number | "…")[] = [];
  for (let k = 0; k < pages.length; k++) {
    if (k > 0 && pages[k] - pages[k - 1] > 1) result.push("…");
    result.push(pages[k]);
  }
  return result;
}