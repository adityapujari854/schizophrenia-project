// frontend/components/ProgressBar.tsx
import React from "react";

export default function ProgressBar({ percent = 0 }: { percent: number }) {
  const p = Math.max(0, Math.min(100, percent));
  return (
    <div style={{ width: "100%", background: "#eee", borderRadius: 8, overflow: "hidden", marginTop: 8 }}>
      <div
        style={{
          width: `${p}%`,
          height: 10,
          background: p > 66 ? "#2ecc71" : p > 33 ? "#f1c40f" : "#3498db",
          transition: "width 200ms ease",
        }}
      />
    </div>
  );
}
