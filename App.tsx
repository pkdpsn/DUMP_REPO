import { useEffect, useState } from "react";
import Streamdown from "streamdown";

export default function App() {
  const [text, setText] = useState("");

  useEffect(() => {
    const evtSource = new EventSource("http://localhost:3001/stream");

    evtSource.onmessage = (event) => {
      setText((prev) => prev + event.data);
    };

    evtSource.addEventListener("end", () => {
      evtSource.close();
    });

    return () => {
      evtSource.close();
    };
  }, []);

  return (
    <div style={{ padding: 20 }}>
      <h2>Streaming Markdown (Streamdown)</h2>

      <Streamdown>{text}</Streamdown>

      {/* cursor */}
      <span style={{ opacity: 0.5 }}>▌</span>
    </div>
  );
}App.tsx
