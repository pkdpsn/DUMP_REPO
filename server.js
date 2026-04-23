import express from "express";

const app = express();

app.get("/stream", async (req, res) => {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  const text = `## Streaming Demo

This is **streaming markdown** using Streamdown.

- Item 1
- Item 2

\`\`\`python
def hello():
    print("hi")
\`\`\`

And some more text coming in...
`;

  for (let i = 0; i < text.length; i++) {
    const chunk = text[i];

    res.write(`data: ${chunk}\n\n`);
    await new Promise((r) => setTimeout(r, 20));
  }

  res.write("event: end\ndata: done\n\n");
  res.end();
});

app.listen(3001, () => {
  console.log("Server running on http://localhost:3001");
});
