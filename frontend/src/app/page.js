"use client";

import { useState, useRef } from "react";

export default function Home() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (!selected) return;
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setPredictions(null);
    setError(null);
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setPredictions(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleClassify = async () => {
    if (!file) return;

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
      });

      if (!response.ok) {
        throw new Error("Prediction failed. Please try again.");
      }

      const data = await response.json();
      setPredictions(data.predictions);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const topPrediction = predictions ? predictions[0] : null;

  const confidenceLabel = (conf) => {
    if (conf > 0.7) return { text: "High Confidence", color: "text-green-700 bg-green-50" };
    if (conf > 0.4) return { text: "Medium Confidence", color: "text-blue-700 bg-blue-50" };
    return { text: "Low Confidence", color: "text-yellow-700 bg-yellow-50" };
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-gray-50 p-6">
      <div className="w-full max-w-lg bg-white rounded-2xl shadow-md p-8">
        <h1 className="text-2xl font-bold text-center text-gray-800 mb-2">
          🔍 Neural Lens
        </h1>
        <p className="text-sm text-center text-gray-500 mb-6">
          Upload any image — our ResNet50 model will identify what's in it,
          across 1,000+ object categories.
        </p>

        <label className="flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-xl p-6 cursor-pointer hover:border-gray-400 transition">
          <span className="text-sm text-gray-500 mb-2">
            {file ? file.name : "Click to select an image"}
          </span>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="hidden"
          />
        </label>

        {preview && (
          <img
            src={preview}
            alt="Preview"
            className="mt-4 w-full h-56 object-cover rounded-xl"
          />
        )}

        <button
          onClick={handleClassify}
          disabled={!file || loading}
          className="mt-6 w-full py-3 bg-gray-900 text-white font-medium rounded-xl disabled:opacity-40 disabled:cursor-not-allowed hover:bg-gray-800 transition cursor-pointer"
        >
          {loading ? "Analyzing through 50 layers..." : "Classify Image"}
        </button>

        {error && (
          <p className="mt-4 text-sm text-red-600 text-center">{error}</p>
        )}

        {topPrediction && (
          <>
            <div className="mt-6 p-5 rounded-xl bg-gray-50 border border-gray-200 text-center">
              <p className="text-xs uppercase tracking-wide text-gray-400 mb-1">
                Top Prediction
              </p>
              <p className="text-2xl font-bold text-gray-800">
                {topPrediction.label}
              </p>
              <span
                className={`inline-block mt-2 px-3 py-1 rounded-full text-xs font-medium ${
                  confidenceLabel(topPrediction.confidence).color
                }`}
              >
                {confidenceLabel(topPrediction.confidence).text} —{" "}
                {(topPrediction.confidence * 100).toFixed(1)}%
              </span>
            </div>

            <div className="mt-6">
              <p className="text-xs uppercase tracking-wide text-gray-400 mb-3">
                Top 5 Predictions
              </p>
              <div className="space-y-3">
                {predictions.map((p) => (
                  <div key={p.rank}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-700">
                        #{p.rank} {p.label}
                      </span>
                      <span className="text-gray-500">
                        {(p.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full h-2 bg-gray-100 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gray-800"
                        style={{ width: `${p.confidence * 100}%` }}
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
          Powered by ResNet50 · Trained on ImageNet · For educational purposes
        </p>
      </div>
    </main>
  );
}