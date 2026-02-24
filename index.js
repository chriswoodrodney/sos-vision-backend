const express = require("express");
const cors = require("cors");
const { ImageAnnotatorClient } = require("@google-cloud/vision");

const app = express();
app.use(cors());
app.use(express.json({ limit: "10mb" }));

const client = new ImageAnnotatorClient();

// health check
app.get("/", (req, res) => {
  res.send("Vision backend running");
});

app.post("/api/scan", async (req, res) => {
  try {
    const { imageBase64 } = req.body;

    if (!imageBase64) {
      return res.status(400).json({ error: "No image provided" });
    }

    const [result] = await client.annotateImage({
      image: { content: imageBase64 },
      features: [
        { type: "LABEL_DETECTION", maxResults: 5 },
        { type: "TEXT_DETECTION", maxResults: 5 }
      ]
    });

    const labels = result.labelAnnotations?.map(l => l.description) || [];
    const text = result.textAnnotations?.[0]?.description || "";

    let placement = "unknown";
    if (labels.some(l => l.toLowerCase().includes("medicine"))) {
      placement = "medical cabinet";
    } else if (labels.some(l => l.toLowerCase().includes("food"))) {
      placement = "food storage";
    }

    res.json({ labels, text, placement });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Vision API failed" });
  }
});

const PORT = 8080;
app.listen(PORT, () => {
  console.log(`Vision backend listening on port ${PORT}`);
});
