// src/App.js
import React, { useState } from "react";
import "./App.css"; // import styles (we'll define background in App.css)

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const API_URL = "https://movie-analysis-api.onrender.com/predict";

  async function handleSubmit(e) {
    e.preventDefault();
    setResult(null);
    if (!text.trim()) return;
    setLoading(true);
    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setResult({ error: "Error contacting API" });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-container">
      <div className="content-box">
        <h2>Sentiment Analysis Demo</h2>
        <form onSubmit={handleSubmit}>
          <textarea
            rows="6"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Type text to check sentiment..."
          />
          <div>
            <button type="submit" disabled={loading}>
              {loading ? "Analyzing..." : "Analyze Sentiment"}
            </button>
          </div>
        </form>

        {result && (
          <div className="result-box">
            {result.error ? (
              <div style={{ color: "red" }}>{result.error}</div>
            ) : (
              <>
                <div><strong>Prediction:</strong> {result.label}</div>
                {result.score !== null && result.score !== undefined && (
                  <div><strong>Confidence:</strong> {(result.score * 100).toFixed(1)}%</div>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
