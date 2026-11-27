// frontend/components/AnalyzeResult.tsx
import React from "react";

function PrettyJSON({ obj }: { obj: any }) {
  return (
    <pre style={{ whiteSpace: "pre-wrap", wordBreak: "break-word", background: "#f7f7f7", padding: 12, borderRadius: 6 }}>
      {JSON.stringify(obj, null, 2)}
    </pre>
  );
}

export default function AnalyzeResult({ data }: { data: any }) {
  if (!data) return null;
  return (
    <div style={{ border: "1px solid #eee", padding: 12, borderRadius: 8 }}>
      <h3>Analyze Result</h3>
      <p>
        <strong>Document ID:</strong> {data.document_id}
      </p>

      <section style={{ marginTop: 8 }}>
        <h4>Classification</h4>
        <div>
          <strong>Label:</strong> {data.classification?.label ?? "—"}
        </div>
        <div>
          <strong>Counts:</strong> {JSON.stringify(data.classification?.counts ?? {})}
        </div>
        <div>
          <strong>Scores:</strong> {JSON.stringify(data.classification?.scores ?? {})}
        </div>
      </section>

      <section style={{ marginTop: 8 }}>
        <h4>Entities</h4>
        <div>
          <strong>Money:</strong> {JSON.stringify(data.entities?.money ?? [])}
        </div>
        <div>
          <strong>Dates:</strong> {JSON.stringify(data.entities?.dates ?? [])}
        </div>
        <div>
          <strong>Parties:</strong> {JSON.stringify(data.entities?.parties ?? [])}
        </div>
      </section>

      <section style={{ marginTop: 8 }}>
        <h4>Clauses</h4>
        {Array.isArray(data.clauses) && data.clauses.length ? (
          <ul>
            {data.clauses.map((c: any, idx: number) => (
              <li key={idx}>
                <strong>{c.type}</strong>: {c.text} <em>({c.sentence_index})</em>
              </li>
            ))}
          </ul>
        ) : (
          <div>None detected</div>
        )}
      </section>

      <section style={{ marginTop: 8 }}>
        <h4>Summary</h4>
        <div>{data.summary ?? "—"}</div>
      </section>

      <section style={{ marginTop: 8 }}>
        <h4>Raw JSON</h4>
        <PrettyJSON obj={data} />
      </section>
    </div>
  );
}
