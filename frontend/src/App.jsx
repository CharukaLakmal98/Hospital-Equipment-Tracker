import { useState } from "react";

export default function App() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");

  const askAgent = async () => {
    if (!query.trim()) return;
    const res = await fetch("http://localhost:8000/ask-agent/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const data = await res.json();
    setResponse(data.response);
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4">
      <div className="bg-white shadow-lg rounded-xl max-w-md w-full p-8 space-y-6">
        <h1 className="text-3xl font-bold text-blue-600 text-center">
          Hospital Equipment Tracker
        </h1>

        <div className="space-y-4">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask about equipment..."
            className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 text-amber-950"
          />

          <button
            onClick={askAgent}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 rounded-lg shadow-md transition duration-200"
          >
            Ask
          </button>
        </div>

        {response && (
          <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 mt-4">
            <h2 className="text-lg font-semibold text-gray-700 mb-2">
              Agent Response:
            </h2>
            <p className="text-gray-800">{response}</p>
          </div>
        )}
      </div>
    </div>
  );
}
