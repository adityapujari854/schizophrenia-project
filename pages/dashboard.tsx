// frontend/pages/dashboard.tsx
import React, { useState } from "react";
import { useSession } from "next-auth/react";
import AccessDenied from "../components/access-denied";
import { uploadFile, apiFetch } from "../lib/api";

const Dashboard: React.FC = () => {
  const { data: session, status } = useSession();
  const [file, setFile] = useState<File | null>(null);
  const [responseData, setResponseData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string>("");

  // Handle file selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  // Handle upload + analyze pipeline
  const handleUploadAndAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMsg("");
    if (!file) {
      setErrorMsg("Please select a file first.");
      return;
    }
    setLoading(true);

    try {
  // 1️⃣ Upload file to FastAPI backend
  const uploadRes = await uploadFile(file);
      console.log("📤 Uploaded:", uploadRes);

      // 2️⃣ Run analysis on uploaded document
      const body = {
        document_id: uploadRes.document_id,
        model_name: "baseline",
        include_summary: true,
      };

      const analyzeRes = await apiFetch("/analyze/run", {
        method: "POST",
        body,
      });

      console.log("🧠 Analysis Result:", analyzeRes);
      setResponseData(analyzeRes);
    } catch (err: any) {
      console.error("❌ Error:", err);
      setErrorMsg(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  if (status === "unauthenticated") {
    return (
      <>
        <AccessDenied />
      </>
    );
  }

  return (
    <>
      <div className="titre">
        <div className="first-word">Legal Document Analyzer:</div>
        <div className="complete-phrase">
          <span>Smart Contract Analysis & Summarization</span>
        </div>
      </div>

      <div className="dashboard">
        <form onSubmit={handleUploadAndAnalyze}>
          <label htmlFor="file-upload" className="drop-container">
            <span className="drop-title">Drop a file here</span> or
            <input
              id="file-upload"
              type="file"
              name="file"
              accept=".pdf,.docx,.txt"
              onChange={handleFileChange}
              required
            />
          </label>

          <button className="custom-btn btn-8" type="submit">
            {loading ? "Analyzing..." : "Upload & Analyze"}
          </button>
        </form>

        {errorMsg && <div style={{ color: "red" }}>{errorMsg}</div>}

        {responseData && (
          <div className="code-container">
            <section className="augs bg" data-augmented-ui>
              <input className="title" value="Analysis Result" readOnly />
              <div className="code highcontrast-dark">
                <textarea
                  className="code-textarea"
                  rows={12}
                  value={
                    responseData.summary ||
                    responseData.summary_full ||
                    JSON.stringify(responseData.classification, null, 2)
                  }
                  readOnly
                />
              </div>
            </section>

            <div className="ccode highcontrast-dark">
              <h3>📑 Document Info</h3>
              <pre>
                {JSON.stringify(
                  {
                    label: responseData.classification?.label,
                    clauses: responseData.clauses?.length || 0,
                    entities: {
                      dates: responseData.entities?.dates?.length || 0,
                      money: responseData.entities?.money?.length || 0,
                      parties: responseData.entities?.parties?.length || 0,
                    },
                  },
                  null,
                  2
                )}
              </pre>
            </div>

            {responseData.summary_translated && (
              <div className="ccode highcontrast-dark">
                <h3>🌍 Translated Summary</h3>
                <p>{responseData.summary_translated}</p>
              </div>
            )}
          </div>
        )}

        {loading && <div className="loader">Processing...</div>}
      </div>
    </>
  );
};

export default Dashboard;
