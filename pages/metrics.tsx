// frontend/pages/metrics.tsx
import React, { useEffect, useState } from "react";
import Head from "next/head";
import { apiFetch, BASE_URL } from "../lib/api";

type ModelMeta = {
  type?: string;
  trained_at?: string | number | null;
  accuracy?: number | null;
  f1?: number | null;
  n_samples?: number | null;
  n_classes?: number | null;
  classes?: string[];
  saved_paths?: string[];
  raw?: any;
};

type TrainingEntry = {
  model_name?: string;
  trained_at?: string; // ISO
  accuracy?: number | null;
  f1?: number | null;
};

type MetricsResponse = {
  active_model?: string | null;
  generated_at?: string | null;
  processed_document_count?: number;
  recent_processed_files?: string[];
  training_label_counts?: Record<string, number>;
  models?: Record<string, ModelMeta>;
  training_history?: TrainingEntry[];
  charts?: Record<string, string>;
};

function fmtDate(iso?: string | number | null) {
  if (!iso) return "-";
  try {
    const d = new Date(iso as any);
    return d.toLocaleString();
  } catch {
    return String(iso);
  }
}

const MetricsPage: React.FC = () => {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    async function load() {
      try {
        const res = await apiFetch("/metrics/");
        if (!mounted) return;
        setMetrics(res as MetricsResponse);
      } catch (err: any) {
        setError(err?.message ? String(err.message) : String(err));
      } finally {
        setLoading(false);
      }
    }
    load();
    return () => {
      mounted = false;
    };
  }, []);

  if (loading) return <div style={{ padding: 24 }}>Loading metrics…</div>;
  if (error) return <div style={{ padding: 24, color: "crimson" }}>Error: {error}</div>;

  const models = metrics?.models ?? {};

  const Sparkline: React.FC<{ history?: TrainingEntry[] }> = ({ history }) => {
    if (!history || history.length === 0) return <div>No history</div>;
    const points = history
      .filter((h) => h.accuracy != null || h.f1 != null)
      .map((h) => ({ t: new Date(h.trained_at || 0).getTime(), acc: h.accuracy ?? null, f1: h.f1 ?? null }));
    if (points.length === 0) return <div>No numeric history available.</div>;
    const xs = points.map((p) => p.t);
    const accs = points.map((p) => p.acc);
    const f1s = points.map((p) => p.f1);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const width = 360;
    const height = 80;
    const pad = 6;
    const vals = [...accs.filter((v): v is number => v != null), ...f1s.filter((v): v is number => v != null)];
    const minV = Math.min(...vals);
    const maxV = Math.max(...vals);
    const toX = (t: number) => pad + ((t - minX) / Math.max(1, maxX - minX)) * (width - pad * 2);
    const toY = (v: number) => height - pad - ((v - minV) / Math.max(1e-6, maxV - minV)) * (height - pad * 2);

    const pathPoints = (arr: Array<number | null>) =>
      arr
        .map((v, i) => (v == null ? null : `${toX(xs[i]).toFixed(1)},${toY(v).toFixed(1)}`))
        .filter((s) => s != null)
        .join(" ");

    const accPath = pathPoints(accs);
    const f1Path = pathPoints(f1s);

    return (
      <svg width={width} height={height} style={{ display: "block" }}>
        <rect x={0} y={0} width={width} height={height} fill="#fff" stroke="#eee" />
        {accPath && <polyline points={accPath} stroke="#2b8cbe" strokeWidth={2} fill="none" />}
        {f1Path && <polyline points={f1Path} stroke="#f46d43" strokeWidth={2} fill="none" />}
      </svg>
    );
  };

  // small inline SVG placeholder data URI (used when a chart fails to load)
  const CHART_PLACEHOLDER =
    "data:image/svg+xml;utf8," +
    encodeURIComponent(
      `<svg xmlns='http://www.w3.org/2000/svg' width='400' height='120'><rect width='100%' height='100%' fill='#fff' stroke='#eee'/><text x='50%' y='50%' dominant-baseline='middle' text-anchor='middle' fill='#999' font-family='Arial' font-size='14'>Chart unavailable</text></svg>`
    );

  function buildChartUrl(pathOrUrl?: string) {
    if (!pathOrUrl) return "";
    if (/^https?:\/\//i.test(pathOrUrl)) return pathOrUrl;
    return `${BASE_URL}${pathOrUrl.startsWith("/") ? "" : "/"}${pathOrUrl}`;
  }

  return (
    <div style={{ maxWidth: 1000, margin: "2rem auto", padding: 24 }}>
      <Head>
        <title>Metrics & Monitoring</title>
      </Head>

      <h1>Metrics & Monitoring</h1>
      <p>Service-level model, dataset and processed-document metrics.</p>

      {/* Main chart (accuracy/f1) */}
      {metrics?.charts?.accuracy_f1 && (
        <div style={{ marginTop: 12, padding: 12, background: "#fff", borderRadius: 6, textAlign: "center" }}>
          <h4 style={{ marginTop: 0 }}>Accuracy / F1 over time</h4>
          {/* ensure full URL for img src */}
          {(() => {
            const mainChartUrl = buildChartUrl(metrics.charts.accuracy_f1);
            return (
              <>
                <img
                  src={mainChartUrl}
            alt="accuracy-f1"
            style={{ maxWidth: "100%", height: "auto", border: "1px solid #eee" }}
            onError={(e: React.SyntheticEvent<HTMLImageElement, Event>) => {
              const t = e.currentTarget as HTMLImageElement;
              console.warn("Metrics chart failed to load:", t.src);
              // set to a small inline SVG placeholder so we don't display an unrelated project image
              if (!t.dataset.fallback) {
                t.dataset.fallback = "1";
                t.src = CHART_PLACEHOLDER;
              }
            }}
                />
                <div style={{ marginTop: 6, fontSize: 12, color: "#666" }}>
                  Chart URL: {mainChartUrl ? <a href={mainChartUrl} target="_blank" rel="noreferrer">Open chart</a> : <em>not available</em>}
                </div>
              </>
            );
          })()}
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 12 }}>
        <div style={{ padding: 12, background: "#f9f9f9", borderRadius: 6 }}>
          <strong>Active model</strong>
          <div style={{ marginTop: 6, fontWeight: 600 }}>{metrics?.active_model ?? "(none)"}</div>
          <div style={{ marginTop: 6, color: "#666", fontSize: 13 }}>Last updated: {fmtDate(metrics?.generated_at ?? null)}</div>
        </div>

        <div style={{ padding: 12, background: "#f4f8ff", borderRadius: 6 }}>
          <strong>Processed documents</strong>
          <div style={{ marginTop: 6, fontWeight: 600 }}>{metrics?.processed_document_count ?? 0}</div>
          <div style={{ marginTop: 8, color: "#666", fontSize: 13 }}>Showing recent {metrics?.recent_processed_files?.length ?? 0}</div>
        </div>
      </div>

      {/* Models table */}
      <section style={{ marginTop: 18, padding: 12, background: "#fff", borderRadius: 6 }}>
        <h3 style={{ marginTop: 0 }}>Models</h3>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
                <tr style={{ textAlign: "left", borderBottom: "2px solid #eee" }}>
                  <th style={{ padding: 8 }}>Name</th>
                  <th style={{ padding: 8 }}>Chart</th>
                  <th style={{ padding: 8 }}>Samples</th>
                  <th style={{ padding: 8 }}>Classes</th>
                  <th style={{ padding: 8 }}>Type</th>
                  <th style={{ padding: 8 }}>Trained At</th>
                  <th style={{ padding: 8 }}>Accuracy</th>
                  <th style={{ padding: 8 }}>F1</th>
                  <th style={{ padding: 8 }}>Classes</th>
                  <th style={{ padding: 8 }}>Artifacts</th>
                </tr>
            </thead>
            <tbody>
              {Object.entries(models).length === 0 && (
                <tr>
                  <td colSpan={9} style={{ padding: 12 }}>
                    No models found
                  </td>
                </tr>
              )}
              {Object.entries(models).map(([name, meta]) => (
                <tr key={name} style={{ borderBottom: "1px solid #f1f1f1" }}>
                  <td style={{ padding: 8, fontWeight: name === metrics?.active_model ? 700 : 500 }}>{name}</td>
                  <td style={{ padding: 8 }}>
                    {/* per-model thumbnail chart - ensure full backend URL */}
                    {(() => {
                      const modelChart = buildChartUrl(`/metrics/chart/accuracy_f1.png?model=${encodeURIComponent(name)}`);
                      return (
                        <>
                          <img
                            src={modelChart}
                            alt={`${name}-chart`}
                            style={{ width: 120, height: 48, objectFit: "cover", border: "1px solid #eee" }}
                            onError={(e: React.SyntheticEvent<HTMLImageElement, Event>) => {
                              const t = e.currentTarget as HTMLImageElement;
                              console.warn(`Model chart failed to load for ${name}:`, t.src);
                              if (!t.dataset.fallback) {
                                t.dataset.fallback = "1";
                                t.src = CHART_PLACEHOLDER;
                              }
                            }}
                          />
                          <div style={{ marginTop: 6, fontSize: 11, color: "#666" }}>
                            <a href={modelChart} target="_blank" rel="noreferrer">Open model chart</a>
                          </div>
                        </>
                      );
                    })()}
                  </td>
                  <td style={{ padding: 8 }}>{meta?.n_samples ?? meta?.raw?.n_samples ?? "-"}</td>
                  <td style={{ padding: 8 }}>{meta?.n_classes ?? meta?.raw?.n_classes ?? (Array.isArray(meta?.classes) ? meta.classes.length : "-")}</td>
                  <td style={{ padding: 8 }}>{meta?.type ?? "?"}</td>
                  <td style={{ padding: 8 }}>{typeof meta?.trained_at === "number" ? new Date((meta.trained_at as number) * 1000).toLocaleString() : fmtDate(meta?.trained_at ?? null)}</td>
                  <td style={{ padding: 8 }}>{meta?.accuracy != null ? (meta.accuracy as number).toFixed(3) : "-"}</td>
                  <td style={{ padding: 8 }}>{meta?.f1 != null ? (meta.f1 as number).toFixed(3) : "-"}</td>
                  <td style={{ padding: 8, maxWidth: 220 }}>{Array.isArray(meta?.classes) ? (meta.classes || []).slice(0, 6).join(", ") : "-"}</td>
                  <td style={{ padding: 8 }}>
                    {Array.isArray(meta?.saved_paths) && meta.saved_paths.length > 0 ? (
                      <div style={{ fontSize: 13 }}>
                        {(meta.saved_paths || []).slice(0, 2).map((s, i) => (
                          <div key={i} style={{ color: "#0366d6", wordBreak: "break-all" }}>
                            {s}
                          </div>
                        ))}
                        {meta.saved_paths.length > 2 && <div style={{ color: "#666", fontSize: 12 }}>+{meta.saved_paths.length - 2} more</div>}
                      </div>
                    ) : (
                      <span style={{ color: "#666" }}>-</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Per-class evaluation for selected model */}
      <section style={{ marginTop: 18, padding: 12, background: "#fff", borderRadius: 6 }}>
        <h3 style={{ marginTop: 0 }}>Per-class evaluation</h3>
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          <div>Choose model:</div>
          <select value={selectedModel ?? ""} onChange={(e) => setSelectedModel(e.target.value || null)}>
            <option value="">-- select model --</option>
            {Object.keys(models).map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>

        {selectedModel && models && models[selectedModel] && (
          <div style={{ marginTop: 12, overflowX: "auto" }}>
            <h4 style={{ marginTop: 0 }}>{selectedModel} — per-class report</h4>
            {models[selectedModel].raw && models[selectedModel].raw.report ? (
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr style={{ textAlign: "left", borderBottom: "1px solid #eee" }}>
                    <th style={{ padding: 8 }}>Class</th>
                    <th style={{ padding: 8 }}>Precision</th>
                    <th style={{ padding: 8 }}>Recall</th>
                    <th style={{ padding: 8 }}>F1-score</th>
                    <th style={{ padding: 8 }}>Support</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(models[selectedModel].raw.report)
                    .filter(([k]) => k !== "accuracy" && k !== "macro avg" && k !== "weighted avg")
                    .map(([cls, vals]: any) => (
                      <tr key={cls} style={{ borderBottom: "1px solid #f6f6f6" }}>
                        <td style={{ padding: 8 }}>{cls}</td>
                        <td style={{ padding: 8 }}>{Number(vals?.precision ?? vals?.["precision"])?.toFixed?.(3) ?? "-"}</td>
                        <td style={{ padding: 8 }}>{Number(vals?.recall ?? vals?.["recall"])?.toFixed?.(3) ?? "-"}</td>
                        <td style={{ padding: 8 }}>{Number(vals?.["f1-score"] ?? vals?.f1 ?? vals?.["f1_score"])?.toFixed?.(3) ?? "-"}</td>
                        <td style={{ padding: 8 }}>{vals?.support ?? "-"}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            ) : (
              <div style={{ padding: 8 }}>No per-class report available for this model.</div>
            )}
          </div>
        )}
      </section>

      {/* Training history sparkline */}
      {metrics?.training_history && metrics.training_history.length > 0 && (
        <section style={{ marginTop: 18, padding: 12, background: "#fff", borderRadius: 6 }}>
          <h3 style={{ marginTop: 0 }}>Training history (accuracy / f1)</h3>
          <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
            <div style={{ flex: 1 }}>
              <Sparkline history={metrics.training_history} />
            </div>
          </div>
        </section>
      )}

      {/* Training label counts */}
      <section style={{ marginTop: 18, padding: 12, background: "#fff", borderRadius: 6 }}>
        <h3 style={{ marginTop: 0 }}>Training label counts</h3>
        {(!metrics?.training_label_counts || Object.keys(metrics.training_label_counts || {}).length === 0) && (
          <div style={{ padding: 8 }}>No training CSV found or no labels present.</div>
        )}
        {metrics?.training_label_counts && Object.keys(metrics.training_label_counts).length > 0 && (
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            {Object.entries(metrics.training_label_counts).map(([lbl, cnt]) => (
              <div key={lbl} style={{ padding: 8, background: "#f4f8ff", borderRadius: 6 }}>
                <div style={{ fontWeight: 700 }}>{lbl}</div>
                <div style={{ color: "#666" }}>{cnt}</div>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Processed docs preview */}
      <section style={{ marginTop: 18, padding: 12, background: "#fff", borderRadius: 6 }}>
        <h3 style={{ marginTop: 0 }}>Recent processed documents</h3>
        {(!metrics?.recent_processed_files || metrics.recent_processed_files.length === 0) && <div style={{ padding: 8 }}>No processed documents found.</div>}
        {metrics?.recent_processed_files && metrics.recent_processed_files.length > 0 && (
          <div style={{ display: "grid", gap: 8 }}>
            {metrics.recent_processed_files.map((fn) => (
              <div key={fn} style={{ padding: 8, border: "1px solid #f1f1f1", borderRadius: 6 }}>
                <div style={{ fontWeight: 600 }}>{fn}</div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
};

export default MetricsPage;
