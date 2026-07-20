"use client";

import { useState, useRef } from "react";

export default function Home() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);
  const abortControllerRef = useRef(null);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (!selected) return;

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setPredictions(null);
    setError(null);
    setLoading(false);
  };

  const handleReset = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    setFile(null);
    setPreview(null);
    setPredictions(null);
    setError(null);
    setLoading(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleClassify = async () => {
    if (!file) return;

    const controller = new AbortController();
    abortControllerRef.current = controller;

    setLoading(true);
    setError(null);
    setPredictions(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

      const response = await fetch(`${apiUrl}/predict`, {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error("Prediction failed. Please try again.");
      }

      const data = await response.json();
      setPredictions(data.predictions);
    } catch (err) {
      if (err.name === "AbortError") {
        return;
      }
      setError(err.message);
    } finally {
      if (abortControllerRef.current === controller) {
        setLoading(false);
      }
    }
  };

  const topPrediction = predictions ? predictions[0] : null;

  const confidenceInfo = (conf) => {
    if (conf > 0.7) return { text: "HIGH CONFIDENCE", color: "#00e5a0" };
    if (conf > 0.4) return { text: "MEDIUM CONFIDENCE", color: "#ffb020" };
    return { text: "LOW CONFIDENCE", color: "#ff5c5c" };
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-gray-50 p-6">
      <div className="w-full max-w-lg bg-white rounded-2xl shadow-md p-8">
        <h1 className="text-2xl font-bold text-center text-gray-800 mb-1">
          🔍 Neural Lens
        </h1>
        <p className="text-sm text-center text-gray-500 mb-6">
          Vision analysis engine — MobileNetV2 · 1,000+ object categories
        </p>

        {!preview && (
          <label className="flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-xl p-8 cursor-pointer hover:border-gray-400 transition">
            <span className="text-sm text-gray-500">
              Click to select an image
            </span>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
            />
          </label>
        )}

        {preview && (
          <div className="relative mt-1 bg-black rounded-xl p-3 overflow-hidden">
            <span className="absolute top-2 left-2 w-6 h-6 border-t-[3px] border-l-[3px] rounded-tl-sm" style={{ borderColor: "#00e5a0" }} />
            <span className="absolute top-2 right-2 w-6 h-6 border-t-[3px] border-r-[3px] rounded-tr-sm" style={{ borderColor: "#00e5a0" }} />
            <span className="absolute bottom-2 left-2 w-6 h-6 border-b-[3px] border-l-[3px] rounded-bl-sm" style={{ borderColor: "#00e5a0" }} />
            <span className="absolute bottom-2 right-2 w-6 h-6 border-b-[3px] border-r-[3px] rounded-br-sm" style={{ borderColor: "#00e5a0" }} />

            <img
              src={preview}
              alt="Preview"
              className="w-full h-56 object-cover rounded-lg"
            />

            {loading && (
              <div
                className="scan-line absolute left-4 right-4 h-0.5"
                style={{ backgroundColor: "#00e5a0", boxShadow: "0 0 8px #00e5a0" }}
              />
            )}

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
            />
          </div>
        )}

        {loading && (
          <p
            className="scan-pulse mt-3 text-center font-mono text-xs tracking-widest"
            style={{ color: "#00e5a0" }}
          >
            ◉ SCANNING...
          </p>
        )}

        <button
          onClick={handleClassify}
          disabled={!file || loading}
          className="mt-6 w-full py-3 bg-gray-900 text-white font-medium rounded-xl disabled:opacity-40 disabled:cursor-not-allowed hover:bg-gray-800 transition cursor-pointer"
        >
          {loading ? "Analyzing..." : "Classify Image"}
        </button>

        {error && (
          <p className="mt-4 text-sm text-red-600 text-center">{error}</p>
        )}

        {topPrediction && (
          <>
            <div className="mt-6 bg-black rounded-xl p-5 text-center">
              <p className="font-mono text-[10px] tracking-[0.2em] text-gray-500">
                TOP MATCH
              </p>
              <p className="text-2xl font-bold text-white mt-1">
                {topPrediction.label}
              </p>
              <p
                className="font-mono text-xs mt-2 tracking-wide"
                style={{ color: confidenceInfo(topPrediction.confidence).color }}
              >
                {confidenceInfo(topPrediction.confidence).text} —{" "}
                {(topPrediction.confidence * 100).toFixed(1)}%
              </p>
            </div>

            <div className="mt-6">
              <p className="font-mono text-[10px] tracking-[0.2em] text-gray-400 mb-3">
                TOP 5 MATCHES
              </p>
              <div className="space-y-3">
                {predictions.map((p) => (
                  <div key={p.rank}>
                    <div className="flex justify-between text-sm mb-1 font-mono">
                      <span className="text-gray-700">
                        #{p.rank} {p.label}
                      </span>
                      <span className="text-gray-500">
                        {(p.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full h-1.5 bg-gray-100 rounded-full overflow-hidden">
                      <div
                        className="h-full"
                        style={{
                          width: `${p.confidence * 100}%`,
                          backgroundColor: confidenceInfo(p.confidence).color,
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}

        {(file || predictions) && (
          <button
            onClick={handleReset}
            className="mt-6 w-full py-2 text-sm text-gray-600 border border-gray-300 rounded-xl hover:bg-gray-50 transition cursor-pointer"
          >
            Try Another
          </button>
        )}

        <p className="mt-8 text-xs text-center text-gray-400">
          Powered by MobileNetV2 · Trained on ImageNet · For educational purposes
        </p>
      </div>
    </main>
  );
}