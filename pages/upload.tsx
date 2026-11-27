// frontend/pages/upload.tsx
import React, { useState } from "react";
import Head from "next/head";
import AnalysisDisplay from "../components/AnalysisDisplay";
import { apiUpload, apiAnalyze } from "../lib/api";

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<string>("");
  const [uploadResult, setUploadResult] = useState<any | null>(null);
  const [analysisResult, setAnalysisResult] = useState<any | null>(null);
  const [showRaw, setShowRaw] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFile(e.target.files?.[0] ?? null);
    setUploadResult(null);
    setAnalysisResult(null);
    setStatus("");
  };

  const handleUpload = async () => {
    if (!file) {
      setStatus("⚠️ Please select a file first.");
      return;
    }

    setStatus("📤 Uploading...");
    setLoading(true);
    setUploadResult(null);
    setAnalysisResult(null);

    try {
      // upload file
      const res = await apiUpload(file);
      setUploadResult(res);
      setStatus("✅ Upload successful! Starting analysis (this may take a minute)...");

      // Pass a longer timeout to apiAnalyze (120s). Adjust upward if needed.
      const analysis = await apiAnalyze(res.document_id, {
        include_summary: true,
        summary_language: "en",
        chunk_size_sentences: 50,
        overlap_sentences: 5,
        model_name: "t5-base", // example: switch model here if you want t5-base
        timeoutMs: 120000, // 120 seconds
      });

      setAnalysisResult(analysis);
      setStatus("✅ Analysis complete!");
    } catch (err: any) {
      console.error("Upload error:", err);
      const message =
        err?.data ??
        (err?.statusText ? `${err.status} ${err.statusText}` : err?.message) ??
        String(err);
      setStatus(`❌ Upload/analysis error: ${message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 900, margin: "2rem auto", padding: 24 }}>
      <Head>
        <title>Upload & Analyze Document</title>
      </Head>

      <h1>📄 Upload & Analyze Document</h1>
      <p>
        Upload a <b>PDF, DOCX, or TXT</b> legal document. The backend will extract text, classify, summarize,
        detect clauses/entities and optionally index for semantic search.
      </p>

      <input
        type="file"
        onChange={handleFileChange}
        accept=".pdf,.docx,.txt"
        style={{
          border: "1px solid #ccc",
          padding: "10px",
          width: "100%",
          marginBottom: "1rem",
        }}
        aria-label="Select document to upload"
      />

      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <button
          onClick={handleUpload}
          disabled={!file || loading}
          style={{
            padding: "10px 16px",
            backgroundColor: "#0070f3",
            color: "white",
            border: "none",
            borderRadius: "6px",
            cursor: !file || loading ? "not-allowed" : "pointer",
            opacity: !file || loading ? 0.6 : 1,
          }}
        >
          {loading ? "Working…" : "Upload & Analyze"}
        </button>

        <button
          onClick={() => {
            setFile(null);
            setUploadResult(null);
            setAnalysisResult(null);
            setStatus("");
          }}
          style={{
            padding: "8px 12px",
            backgroundColor: "#eef2ff",
            color: "#0b63d6",
            border: "1px solid #c7ddff",
            borderRadius: 6,
            cursor: "pointer",
          }}
        >
          Reset
        </button>

        <button
          onClick={() => window.location.href = '/metrics'}
          style={{
            padding: "8px 12px",
            backgroundColor: "#fff8e6",
            color: "#8a5a00",
            border: "1px solid #ffe7b5",
            borderRadius: 6,
            cursor: "pointer",
          }}
        >
          View Metrics
        </button>

        <label style={{ marginLeft: "auto", display: "flex", gap: 8, alignItems: "center" }}>
          <input type="checkbox" checked={showRaw} onChange={(e) => setShowRaw(e.target.checked)} />
          <span style={{ fontSize: 13 }}>Show raw JSON (debug)</span>
        </label>
      </div>

      <p style={{ marginTop: "1rem", color: "#555" }}>{status}</p>

      {/* Upload Result (friendly summary) */}
      {uploadResult && (
        <section
          style={{
            background: "#f9f9f9",
            padding: "1rem",
            borderRadius: "6px",
            marginTop: "1.5rem",
          }}
        >
          <h3>Upload Details</h3>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
            <div><strong>Document ID</strong><div>{uploadResult.document_id}</div></div>
            <div><strong>Filename</strong><div>{uploadResult.filename}</div></div>
            <div><strong>Saved Path</strong><div style={{ wordBreak: "break-all" }}>{uploadResult.saved_path}</div></div>
            <div><strong>Text length</strong><div>{uploadResult.text_length ?? "—"}</div></div>
            <div><strong>Sentence count</strong><div>{uploadResult.sentence_count ?? "—"}</div></div>
            {uploadResult.meta && <div><strong>Meta</strong><div>{Object.entries(uploadResult.meta).map(([k,v]) => `${k}: ${v}`).join(" • ")}</div></div>}
          </div>

          {showRaw && (
            <>
              <h4 style={{ marginTop: 12 }}>Raw upload JSON</h4>
              <pre style={{ whiteSpace: "pre-wrap", maxHeight: 300, overflow: "auto" }}>{JSON.stringify(uploadResult, null, 2)}</pre>
            </>
          )}
        </section>
      )}

      {/* Analysis Result (presentational) */}
      {analysisResult && (
        <section
          style={{
            background: "#f4f8ff",
            padding: "1rem",
            borderRadius: "6px",
            marginTop: "1.5rem",
          }}
        >
          <AnalysisDisplay analysis={analysisResult} />

          {showRaw && (
            <>
              <h4 style={{ marginTop: 12 }}>Raw analysis JSON</h4>
              <pre style={{ whiteSpace: "pre-wrap", maxHeight: 400, overflow: "auto" }}>{JSON.stringify(analysisResult, null, 2)}</pre>
            </>
          )}
        </section>
      )}
    </div>
  );
}
