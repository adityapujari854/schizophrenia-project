// frontend/components/UploadAndAnalyze.tsx
import React, { useState } from "react";
import { apiFetch } from "../lib/api";
import AnalyzeResult from "./AnalyzeResult";
import ProgressBar from "./ProgressBar";

type AnalyzeResponse = {
  document_id: string;
  filename?: string;
  processed_text_path?: string;
  sentences_path?: string;
  text_length?: number;
  sentence_count?: number;
  classification?: any;
  entities?: any;
  clauses?: any;
  summary?: string | null;
  summary_full?: any;
  summary_translated?: string | null;
};

export default function UploadAndAnalyze() {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);

  function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    setError(null);
    setResult(null);
    const f = e.target.files && e.target.files[0];
    if (f) setFile(f);
  }

  async function handleUpload() {
    setError(null);
    if (!file) {
      setError("Please choose a file to upload first.");
      return;
    }

    setUploading(true);
    setProgress(5);
    try {
      const fd = new FormData();
      fd.append("file", file, file.name);

      // Use fetch directly to track progress; apiFetch wrapper can be used if you don't need progress events
      const xhr = new XMLHttpRequest();
      const url = (process.env.NEXT_PUBLIC_API_BASE_URL || "") + "/upload/";
      xhr.open("POST", url, true);
      const apiKey = process.env.NEXT_PUBLIC_API_KEY || "";
      if (apiKey) {
        xhr.setRequestHeader("x-api-key", apiKey);
      }

      xhr.upload.onprogress = function (e) {
        if (e.lengthComputable) {
          const pct = Math.round((e.loaded / e.total) * 100);
          setProgress(Math.max(5, pct));
        }
      };

      xhr.onload = function () {
        setUploading(false);
        setProgress(100);
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const data = JSON.parse(xhr.responseText);
            // Trigger analyze automatically
            if (data && data.document_id) {
              runAnalyze(data.document_id);
            } else {
              setError("Upload succeeded but no document_id returned.");
            }
          } catch (err) {
            setError("Upload succeeded but response parse failed.");
          }
        } else {
          let msg = `Upload failed: ${xhr.status} ${xhr.statusText}`;
          try {
            const body = JSON.parse(xhr.responseText);
            msg += ` - ${JSON.stringify(body)}`;
          } catch {}
          setError(msg);
        }
      };

      xhr.onerror = function () {
        setUploading(false);
        setError("Network error during upload.");
      };

      xhr.send(fd);
    } catch (err: any) {
      setUploading(false);
      setError(err?.message || String(err));
    }
  }

  async function runAnalyze(documentId: string) {
    setAnalyzing(true);
    setResult(null);
    setError(null);
    try {
      const payload = {
        document_id: documentId,
        model_name: "baseline",
        include_summary: true,
        summary_language: null,
        chunk_size_sentences: 50,
        overlap_sentences: 5,
      };
      // Submit background job
      const submit = (await apiFetch("/analyze/submit", {
        method: "POST",
        body: payload,
        headers: { "Content-Type": "application/json" },
        timeoutMs: 10000,
      })) as { job_id: string; status_url?: string };

      if (!submit || !submit.job_id) {
        throw new Error("Failed to submit analysis job")
      }

      // poll status until completed
      const pollUrl = `/analyze/status/${submit.job_id}`;
      let attempts = 0;
      let lastStage = "queued";
      while (attempts < 60) {
        attempts += 1;
        try {
          const status = (await apiFetch(pollUrl, { method: "GET", timeoutMs: 10000 })) as any;
          if (status && status.status) {
            if (status.status === "completed") {
              // fetch result
              const result = (await apiFetch(`/analyze/result/${submit.job_id}`, { method: "GET", timeoutMs: 120000 })) as AnalyzeResponse;
              setResult(result);
              break;
            }

            if (status.status === "failed") {
              throw new Error(status.error || "Analysis job failed");
            }

            // update progress based on stage
            if (status.stage && status.stage !== lastStage) {
              lastStage = status.stage;
              if (status.stage === "classification") setProgress(40);
              else if (status.stage === "extraction") setProgress(65);
              else if (status.stage === "summary") setProgress(85);
              else setProgress(Math.min(90, progress + 5));
            }
          }
        } catch (err) {
          // ignore transient polling errors
        }

        // wait a bit (exponential backoff up to ~5s)
        await new Promise((r) => setTimeout(r, Math.min(5000, 200 * attempts)));
      }
    } catch (err: any) {
      setError(err?.message || JSON.stringify(err));
    } finally {
      setAnalyzing(false);
      setProgress(0);
    }
  }

  return (
    <div style={{ maxWidth: 820, margin: "1rem auto", padding: 16 }}>
      <h2>Upload & Analyze Document</h2>
      <p style={{ marginTop: 0 }}>
        Choose a PDF / DOCX / TXT file. The backend will extract text, preprocess, run classification, extraction, summarization and return results.
      </p>

      <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 12 }}>
        <input type="file" accept=".pdf,.docx,.txt" onChange={onFileChange} />
        <button onClick={handleUpload} disabled={!file || uploading} style={{ padding: ".5rem 1rem" }}>
          {uploading ? "Uploading..." : "Upload & Analyze"}
        </button>
      </div>

      {uploading && <ProgressBar percent={progress} />}

      {analyzing && (
        <div style={{ marginTop: 12 }}>
          <em>Analyzing document — this may take a few seconds...</em>
        </div>
      )}

      {error && (
        <div style={{ marginTop: 12, color: "crimson" }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div style={{ marginTop: 16 }}>
          <AnalyzeResult data={result} />
        </div>
      )}
    </div>
  );
}
