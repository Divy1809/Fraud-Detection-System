import { useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:8000/upload-image/", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      setResult(data);
    } catch (error) {
      console.error(error);
      alert("Request failed");
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h1>Refund Fraud Detection</h1>

      <input
        type="file"
        accept="image/jpeg, image/png"
        onChange={(e) => setFile(e.target.files[0])}
      />

      <br /><br />

      <button onClick={handleUpload}>Upload</button>

      <br /><br />

      {result && (
        <div>
          <h3>Status: {result.status}</h3>
          {result.reason && <p>Reason: {result.reason}</p>}
        </div>
      )}
    </div>
  );
}

export default App;