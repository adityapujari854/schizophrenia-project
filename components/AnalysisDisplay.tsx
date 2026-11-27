// frontend/components/AnalysisDisplay.tsx
import React from "react";

type Clause = {
  type?: string;
  text?: string;
  sentence_index?: number;
};

type Analysis = {
  document_id?: string;
  classification?: {
    label?: string | null;
    counts?: Record<string, number>;
    scores?: Record<string, number>;
  };
  entities?: {
    dates?: string[];
    money?: string[];
    parties?: string[];
    percentages?: string[];
    persons?: string[];
    orgs?: string[];
  };
  clauses?: Clause[];
  summary?: string | null;
  summary_full?: string | null;
  sentence_count?: number;
  sentenceCount?: number;
  [k: string]: any;
};

const small = (v?: number) => {
  if (v === undefined || v === null) return "—";
  try {
    return Number(v).toFixed(3);
  } catch {
    return String(v);
  }
};

function SectionRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ display: "flex", gap: 12, marginBottom: 8 }}>
      <div style={{ minWidth: 120, fontWeight: 600, color: "#333" }}>{label}</div>
      <div style={{ flex: 1 }}>{children}</div>
    </div>
  );
}

function ListInline({ items }: { items?: string[] }) {
  if (!items || items.length === 0) return <span style={{ color: "#777" }}>—</span>;
  return (
    <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
      {items.map((it, i) => (
        <span
          key={i}
          style={{
            background: "#fff",
            border: "1px solid #e6eefc",
            padding: "4px 8px",
            borderRadius: 6,
            color: "#0b63d6",
            fontSize: 13,
            whiteSpace: "pre-wrap",
          }}
        >
          {it}
        </span>
      ))}
    </div>
  );
}

export default function AnalysisDisplay({ analysis }: { analysis: Analysis }) {
  if (!analysis) return null;

  const classification = analysis.classification ?? { label: null, counts: {}, scores: {} };
  const entities = Object.assign(
    { dates: [], money: [], parties: [], percentages: [], persons: [], orgs: [] },
    analysis.entities ?? {}
  );
  const clauses: Clause[] = Array.isArray(analysis.clauses) ? analysis.clauses : [];

  const scores = classification.scores ?? {};
  const scoreEntries = Object.entries(scores).sort((a, b) => (Number(b[1]) as number) - (Number(a[1]) as number));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12 }}>
        <div>
          <h3 style={{ margin: 0, fontSize: 18 }}>Analysis Summary</h3>
          <div style={{ color: "#555", marginTop: 6 }}>
            ID: <strong style={{ color: "#111" }}>{analysis.document_id ?? "—"}</strong>{" "}
            · Sentences: <strong>{analysis.sentence_count ?? analysis.sentenceCount ?? "—"}</strong>
          </div>
        </div>

        <div style={{ textAlign: "right" }}>
          <div style={{ fontSize: 12, color: "#666" }}>Top label</div>
          <div style={{ fontWeight: 700, fontSize: 16, color: "#0b63d6" }}>
            {classification.label ?? "—"}
          </div>
        </div>
      </div>

      <div
        style={{
          background: "#fff",
          border: "1px solid #e6eefc",
          padding: 12,
          borderRadius: 8,
          boxShadow: "0 1px 0 rgba(12,20,40,0.03)",
        }}
      >
        <SectionRow label="Classification">
          <div>
            <div style={{ marginBottom: 8 }}>
              <strong>Counts:</strong>{" "}
              {classification.counts && Object.keys(classification.counts).length > 0 ? (
                <span style={{ color: "#111" }}>
                  {Object.entries(classification.counts)
                    .map(([k, v]) => `${k}: ${v}`)
                    .join(" • ")}
                </span>
              ) : (
                <span style={{ color: "#777" }}>—</span>
              )}
            </div>

            <div>
              <strong>Scores:</strong>{" "}
              {scoreEntries.length > 0 ? (
                <div style={{ display: "flex", gap: 8, marginTop: 6, flexWrap: "wrap" }}>
                  {scoreEntries.map(([k, v]) => (
                    <div
                      key={k}
                      style={{
                        background: "#f1f7ff",
                        padding: "6px 10px",
                        borderRadius: 6,
                        border: "1px solid #d9e9ff",
                        display: "flex",
                        gap: 8,
                        alignItems: "center",
                        minWidth: 140,
                      }}
                    >
                      <div style={{ flex: 1 }}>
                        <div style={{ fontSize: 13, color: "#0b63d6", fontWeight: 600 }}>{k}</div>
                        <div style={{ fontSize: 12, color: "#333" }}>{small(Number(v))}</div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <span style={{ color: "#777" }}>—</span>
              )}
            </div>
          </div>
        </SectionRow>

        <hr style={{ border: "none", borderTop: "1px solid #f0f3fb", margin: "12px 0" }} />

        <SectionRow label="Entities">
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
            <div>
              <div style={{ fontSize: 13, color: "#444", marginBottom: 6 }}>Dates</div>
              <ListInline items={entities.dates} />
            </div>
            <div>
              <div style={{ fontSize: 13, color: "#444", marginBottom: 6 }}>Money</div>
              <ListInline items={entities.money} />
            </div>

            <div>
              <div style={{ fontSize: 13, color: "#444", marginBottom: 6 }}>Parties</div>
              <ListInline items={entities.parties} />
            </div>
            <div>
              <div style={{ fontSize: 13, color: "#444", marginBottom: 6 }}>Percentages</div>
              <ListInline items={entities.percentages} />
            </div>

            <div>
              <div style={{ fontSize: 13, color: "#444", marginBottom: 6 }}>Persons</div>
              <ListInline items={entities.persons} />
            </div>
            <div>
              <div style={{ fontSize: 13, color: "#444", marginBottom: 6 }}>Orgs</div>
              <ListInline items={entities.orgs} />
            </div>
          </div>
        </SectionRow>

        <hr style={{ border: "none", borderTop: "1px solid #f0f3fb", margin: "12px 0" }} />

        <SectionRow label="Clauses">
          <div>
            {clauses.length === 0 ? (
              <div style={{ color: "#777" }}>No clauses detected</div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {clauses.map((c, i) => (
                  <div
                    key={i}
                    style={{
                      background: "#fff",
                      border: "1px solid #eef6ff",
                      padding: 10,
                      borderRadius: 8,
                    }}
                  >
                    <div style={{ fontSize: 13, fontWeight: 700, color: "#0b63d6" }}>
                      {c.type ?? "clause"}
                      <span style={{ marginLeft: 8, fontSize: 12, color: "#666", fontWeight: 500 }}>
                        (sentence #{c.sentence_index ?? "—"})
                      </span>
                    </div>
                    <div style={{ marginTop: 6, color: "#222", whiteSpace: "pre-wrap" }}>{c.text ?? "—"}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </SectionRow>

        <hr style={{ border: "none", borderTop: "1px solid #f0f3fb", margin: "12px 0" }} />

        <SectionRow label="Summary">
          <div>
            {analysis.summary ? (
              <div style={{ color: "#111", lineHeight: 1.5, whiteSpace: "pre-wrap" }}>{analysis.summary}</div>
            ) : (
              <div style={{ color: "#777" }}>No summary available</div>
            )}

            {analysis.summary_full && analysis.summary_full !== analysis.summary && (
              <details style={{ marginTop: 8 }}>
                <summary style={{ cursor: "pointer", color: "#0b63d6" }}>View full summary</summary>
                <div style={{ marginTop: 8, color: "#333", whiteSpace: "pre-wrap" }}>{analysis.summary_full}</div>
              </details>
            )}
          </div>
        </SectionRow>
      </div>
    </div>
  );
}
