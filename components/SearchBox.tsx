// frontend/components/SearchBox.tsx
import React, { useEffect, useState } from "react";

type ResultItem = {
  document_id?: string;
  chunk_id?: string;
  text: string;
  score: number;
  meta?: Record<string, any> | null;
};

export default function SearchBox() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState<number>(5);
  const [docId, setDocId] = useState<string>("");
  const [results, setResults] = useState<ResultItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [indexedDocs, setIndexedDocs] = useState<string[]>([]);
  const [indexing, setIndexing] = useState(false);
  const [indexDocId, setIndexDocId] = useState("");

  useEffect(() => {
    // fetch indexed documents (calls list_indexed_documents via /search/status by scanning folder)
    // We'll hit /search/query with empty query or create a small API - for now try /search/status on known files
    fetch("/search/status/list") // this endpoint doesn't exist; fallback below
      .then((r) => {
        if (!r.ok) throw new Error("no list endpoint");
        return r.json();
      })
      .then((data: any) => {
        // expecting { documents: [...] } - but our backend has list_indexed_documents() only
        if (Array.isArray(data)) {
          setIndexedDocs(data);
        } else if (data.documents) {
          setIndexedDocs(data.documents);
        }
      })
      .catch(() => {
        // fallback: try to call existing backend route via programmatic import (not possible here)
        // So we'll leave indexedDocs empty until user indexes.
      });
  }, []);

  const runQuery = async () => {
    if (!query || !query.trim()) return;
    setLoading(true);
    setResults([]);
    try {
      const payload: any = { query, top_k: topK };
      if (docId) payload.document_id = docId;
      const resp = await fetch("/search/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body = await resp.json();
      if (!resp.ok) {
        console.error("Query failed", body);
        alert(body.detail || "Query failed");
      } else {
        setResults(body.results || []);
      }
    } catch (err) {
      console.error(err);
      alert("Search request failed");
    } finally {
      setLoading(false);
    }
  };

  const doIndex = async (documentId: string) => {
    if (!documentId) {
      alert("Enter document_id (filename without extension) to index from processed folder");
      return;
    }
    setIndexing(true);
    try {
      const resp = await fetch("/search/index", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ document_id: documentId }),
      });
      const body = await resp.json();
      if (!resp.ok) {
        alert(body.detail || "Indexing failed");
      } else {
        alert("Index created: " + JSON.stringify(body.result || body));
        // add to indexed list
        setIndexedDocs((s) => (s.includes(documentId) ? s : [...s, documentId]));
      }
    } catch (err) {
      console.error(err);
      alert("Indexing request failed");
    } finally {
      setIndexing(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto p-4">
      <div className="mb-4 p-4 bg-white rounded shadow">
        <h2 className="text-lg font-semibold mb-2">Semantic Search</h2>

        <div className="mb-2">
          <label className="block text-sm font-medium text-gray-700">Document (optional)</label>
          <input
            value={docId}
            onChange={(e) => setDocId(e.target.value)}
            placeholder="document_id (optional)"
            className="mt-1 block w-full rounded border px-3 py-2"
          />
          <small className="text-gray-500">Leave empty to search across all indexed documents</small>
        </div>

        <div className="mb-2">
          <label className="block text-sm font-medium text-gray-700">Query</label>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter natural language query"
            className="mt-1 block w-full rounded border px-3 py-2"
          />
        </div>

        <div className="flex items-center gap-2 mb-3">
          <label className="text-sm">Top K</label>
          <input
            type="number"
            value={topK}
            min={1}
            max={50}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="w-20 rounded border px-2 py-1"
          />
          <button onClick={runQuery} disabled={loading} className="ml-auto bg-indigo-600 text-white px-4 py-2 rounded">
            {loading ? "Searching..." : "Search"}
          </button>
        </div>

        <div className="mt-2">
          <h3 className="font-medium">Index a processed document</h3>
          <div className="flex gap-2 mt-2">
            <input
              value={indexDocId}
              onChange={(e) => setIndexDocId(e.target.value)}
              placeholder="document_id (filename without .txt)"
              className="flex-1 rounded border px-3 py-2"
            />
            <button onClick={() => doIndex(indexDocId)} disabled={indexing} className="bg-green-600 text-white px-4 py-2 rounded">
              {indexing ? "Indexing..." : "Index"}
            </button>
          </div>
          <p className="text-sm text-gray-500 mt-1">This will index <code>backend/data/processed/{'{document_id}.txt'}</code></p>
        </div>
      </div>

      <div className="bg-white rounded shadow p-4">
        <h3 className="font-semibold mb-2">Results</h3>
        {!results.length && <p className="text-gray-500">No results yet.</p>}
        <ul className="space-y-3">
          {results.map((r, idx) => (
            <li key={`${r.document_id}-${r.chunk_id}-${idx}`} className="p-3 border rounded">
              <div className="flex items-baseline justify-between">
                <div className="text-sm text-gray-600">
                  <strong>{r.document_id}</strong> • chunk {r.chunk_id}
                </div>
                <div className="text-sm text-gray-700">score: {r.score.toFixed(3)}</div>
              </div>
              <p className="mt-2 text-sm text-gray-800">{r.text}</p>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
