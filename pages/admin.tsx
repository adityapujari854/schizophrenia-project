import React, { useState, useEffect } from "react";
import Layout from "../components/layout";
import { apiFetch } from "../lib/api";

export default function Page() {
  const [file, setFile] = useState<File | null>(null);
  const [modelName, setModelName] = useState<string>("baseline");
  const [background, setBackground] = useState<boolean>(true);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<any>(null);
  const [result, setResult] = useState<any>(null);
  const [models, setModels] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // list models on mount
    (async () => {
      try {
        const m = await apiFetch("/train/models", { method: "GET" });
        if (m && m.models) setModels(m.models || []);
      } catch (e: any) {
        // ignore
      }
    })();
  }, []);

  function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files && e.target.files[0];
    setFile(f || null);
  }

  async function submitTrain() {
    setError(null);
    setStatus(null);
    setResult(null);
    setJobId(null);

    if (!file) {
      setError("Please choose a CSV file to upload.");
      return;
    }
    if (!file.name.toLowerCase().endsWith(".csv")) {
      setError("Only CSV files are accepted. Ensure it has 'text' and 'label' columns.");
      return;
    }

    setLoading(true);
    try {
      const form = new FormData();
      form.append("file", file, file.name);

      const query = `?background=${background ? "true" : "false"}&model_name=${encodeURIComponent(
        modelName
      )}`;
      const res = await apiFetch(`/train/upload${query}`, { method: "POST", body: form, raw: true, timeoutMs: 120000 });

      if (background) {
        // expect { job_id, status_url, result_url }
        if (res && res.job_id) {
          setJobId(res.job_id);
          pollTrainStatus(res.job_id);
        } else {
          setError("Background training accepted but no job_id returned.");
        }
      } else {
        // synchronous result
        setResult(res?.result ?? res);
      }
    } catch (err: any) {
      setError(err?.data ?? err?.message ?? String(err));
    } finally {
      setLoading(false);
    }
  }

  async function pollTrainStatus(id: string) {
    setStatus(null);
    setResult(null);
    const deadline = Date.now() + 1000 * 60 * 10; // 10 minutes max
    while (Date.now() < deadline) {
      try {
        const s = await apiFetch(`/train/status/${id}`, { method: "GET", timeoutMs: 10000 });
        setStatus(s);
        if (s && s.status === "completed") {
          const r = await apiFetch(`/train/result/${id}`, { method: "GET", timeoutMs: 120000 });
          setResult(r);
          break;
        }
        if (s && s.status === "failed") {
          const r = await apiFetch(`/train/result/${id}`, { method: "GET", timeoutMs: 10000 });
          setResult(r);
          break;
        }
      } catch (e: any) {
        // transient errors ignored
      }
      // backoff
      await new Promise((r) => setTimeout(r, 2000));
    }
  }

  return (
    <Layout>
      <h1>Admin — Training</h1>
      <p>Only admin users should see this page. Use it to upload labeled CSVs and train classifiers.</p>

      <div style={{ margin: "1rem 0", padding: 12, border: "1px solid #ddd", borderRadius: 6 }}>
        <label>
          Training CSV (columns: text,label):
          <input type="file" accept=".csv" onChange={onFileChange} style={{ display: "block", marginTop: 8 }} />
        </label>

        <label style={{ display: "block", marginTop: 8 }}>
          Model name:
          <input value={modelName} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setModelName(e.target.value)} style={{ marginLeft: 8 }} />
        </label>

        <label style={{ display: "block", marginTop: 8 }}>
          <input type="checkbox" checked={background} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setBackground(e.target.checked)} /> Run in background (recommended)
        </label>

        <div style={{ marginTop: 12 }}>
          <button onClick={submitTrain} disabled={loading} style={{ padding: "0.5rem 1rem" }}>
            {loading ? "Submitting..." : "Upload & Train"}
          </button>
        </div>

        {error && (
          <div style={{ color: "crimson", marginTop: 12 }}>
            <strong>Error:</strong> {String(error)}
          </div>
        )}

        {jobId && (
          <div style={{ marginTop: 12 }}>
            <strong>Job ID:</strong> {jobId}
            <div>
              <em>Polling status...</em>
            </div>
          </div>
        )}

        {status && (
          <div style={{ marginTop: 12 }}>
            <h4>Status</h4>
            <pre style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(status, null, 2)}</pre>
          </div>
        )}

        {result && (
          <div style={{ marginTop: 12 }}>
            <h4>Result</h4>
            <pre style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(result, null, 2)}</pre>
          </div>
        )}
      </div>

      <div style={{ marginTop: 16 }}>
        <h3>Available Models</h3>
        <p>Models discovered in the backend models directory:</p>
        <ul>
          {models.length === 0 && <li>No models found</li>}
          {models.map((m) => (
            <li key={m.model_name}>
              <strong>{m.model_name}</strong>
              {m.has_metadata ? (
                <pre style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(m.metadata, null, 2)}</pre>
              ) : (
                " — no metadata"
              )}
            </li>
          ))}
        </ul>
      </div>
    </Layout>
  );
}
