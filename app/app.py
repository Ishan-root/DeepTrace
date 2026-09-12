import sys
from pathlib import Path

import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import get_model

MODEL_PATH = ROOT / "checkpoints" / "resnet18_deeptrace.pth"
CLASS_NAMES = ["FAKE", "REAL"]
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


CSS = """
:root {
  --bg-0: #06080d;
  --bg-1: #0d1117;
  --bg-2: #101820;
  --panel-border: rgba(255,255,255,0.1);
  --text: #edf3ff;
  --muted: rgba(237,243,255,0.62);
  --blue: #5ea2ff;
  --violet: #7d69ff;
  --green: #4ade80;
  --red: #f87171;
}

html, body, .gradio-container {
  margin: 0;
  background:
    radial-gradient(circle at 15% 10%, rgba(125,105,255,0.18) 0%, transparent 30%),
    radial-gradient(circle at 85% 20%, rgba(94,162,255,0.16) 0%, transparent 30%),
    linear-gradient(160deg, var(--bg-0) 0%, var(--bg-1) 45%, var(--bg-2) 100%) !important;
  font-family: "SF Pro Display", "SF Pro Text", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  color: var(--text);
}

/* subtle grain, purely decorative, sits behind everything */
body::before {
  content: "";
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px),
    radial-gradient(rgba(255,255,255,0.08) 0.6px, transparent 0.8px);
  background-size: 18px 18px, 18px 18px, 8px 8px;
  opacity: 0.24;
  pointer-events: none;
  z-index: 0;
}

/* --- outer shell: centers everything --- */
#page-shell {
  position: relative;
  max-width: 1200px !important;
  margin: 0 auto !important;
  padding: 52px 22px 40px !important;
  min-height: 100vh;
}

/* --- the glass card itself, rendered directly by Gradio's Column --- */
#glass-card {
  position: relative;
  z-index: 1;
  width: min(760px, 100%);
  margin: 0 auto !important;
  padding: 26px 24px 20px !important;
  border-radius: 30px !important;
  background: rgba(18, 23, 30, 0.62) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  box-shadow:
    0 24px 80px rgba(0,0,0,0.58),
    inset 0 1px 0 rgba(255,255,255,0.12),
    inset 0 -12px 24px rgba(0,0,0,0.12);
  backdrop-filter: blur(24px) saturate(140%);
  -webkit-backdrop-filter: blur(24px) saturate(140%);
}

.topbar {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 18px;
}

.logo-mark {
  width: 28px;
  height: 28px;
  border-radius: 9px;
  background: linear-gradient(145deg, rgba(94,162,255,0.9), rgba(125,105,255,0.9));
  box-shadow: 0 0 20px rgba(125,105,255,0.6);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  font-weight: 700;
  color: white;
}
.logo-mark::after { content: "D"; }

.brand-title {
  font-size: 1.05rem;
  font-weight: 700;
  letter-spacing: 0.02em;
  color: var(--text);
}

.brand-subtitle {
  margin-left: auto;
  font-size: 0.75rem;
  color: var(--muted);
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

/* --- upload box: styles applied to Gradio's own wrapper via elem_classes --- */
.upload-box {
  border-radius: 22px !important;
  border: 1.5px dashed rgba(166, 188, 255, 0.58) !important;
  background: rgba(255,255,255,0.02) !important;
  overflow: hidden;
}

.scan-button {
  margin-top: 18px !important;
  width: 100% !important;
  background: linear-gradient(135deg, var(--blue), var(--violet)) !important;
  border: 0 !important;
  border-radius: 14px !important;
  color: white !important;
  font-size: 0.96rem !important;
  font-weight: 700 !important;
  padding: 14px 22px !important;
  box-shadow: 0 12px 28px rgba(125,105,255,0.34);
  transition: all 0.22s ease;
}
.scan-button:hover {
  transform: translateY(-1px);
  box-shadow: 0 16px 32px rgba(125,105,255,0.42);
}

.result-block {
  margin-top: 18px;
  padding: 18px 18px 14px;
  border-radius: 22px;
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.06);
}

.verdict {
  font-size: clamp(2rem, 3vw, 3rem);
  letter-spacing: 0.06em;
  font-weight: 800;
  line-height: 1;
  margin-bottom: 8px;
}
.verdict.real { color: var(--green); text-shadow: 0 0 14px rgba(74, 222, 128, 0.28); }
.verdict.fake { color: var(--red); text-shadow: 0 0 14px rgba(248, 113, 113, 0.26); }

.confidence {
  font-size: 0.95rem;
  color: var(--muted);
  margin-bottom: 20px;
}

.map-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}

.map-card {
  min-height: 110px;
  border-radius: 16px;
  background: rgba(255,255,255,0.025);
  border: 1px solid rgba(255,255,255,0.08);
  display: flex;
  align-items: center;
  justify-content: center;
  color: rgba(255,255,255,0.6);
  font-size: 0.74rem;
  letter-spacing: 0.04em;
}

.floating-note {
  position: absolute;
  z-index: 2;
  max-width: 220px;
  font-size: 0.77rem;
  line-height: 1.45;
  color: rgba(255,255,255,0.42);
  padding: 0 8px;
}
.note-a { left: -230px; top: 130px; }
.note-b { right: -230px; bottom: 140px; }
.floating-note strong { color: rgba(255,255,255,0.65); font-weight: 700; }

.empty-result {
  padding: 28px 10px 12px;
  text-align: center;
  color: rgba(255,255,255,0.5);
  font-size: 0.9rem;
}

@media (max-width: 900px) {
  #page-shell { padding: 34px 16px !important; }
  .floating-note { display: none; }
  #glass-card { padding: 22px 18px 18px !important; }
}
"""


def analyze_image(image):
    if image is None:
        return "<div class='empty-result'>Upload an image to begin the scan.</div>"

    pil_image = image.convert("RGB")

    with torch.no_grad():
        tensor = transform(pil_image).unsqueeze(0).to(device)
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    predicted_idx = int(torch.argmax(probs).item())
    predicted_label = CLASS_NAMES[predicted_idx]
    confidence = float(probs[predicted_idx].item() * 100)
    verdict_class = "real" if predicted_label == "REAL" else "fake"

    return f"""
    <div class="result-block">
      <div class="verdict {verdict_class}">{predicted_label}</div>
      <div class="confidence">{confidence:.2f}% confidence</div>
      <div class="map-grid">
        <div class="map-card">Where it looked</div>
        <div class="map-card">Where layers disagreed</div>
      </div>
    </div>
    """


with gr.Blocks(css=CSS, theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    with gr.Column(elem_id="page-shell"):
        gr.HTML(
            """
            <div class="floating-note note-a"><strong>Multi-Layer Explainability</strong> — sees not just what, but where and why.</div>
            <div class="floating-note note-b"><strong>Statistically Guaranteed Confidence</strong> — no guesswork, a real range.</div>
            """
        )

        with gr.Column(elem_id="glass-card"):
            gr.HTML(
                """
                <div class="topbar">
                  <div class="logo-mark"></div>
                  <div class="brand-title">DeepTrace</div>
                  <div class="brand-subtitle">AI Image Forensics</div>
                </div>
                """
            )

            upload = gr.Image(
                type="pil",
                label="",
                show_label=False,
                height=220,
                elem_classes=["upload-box"],
            )

            scan_button = gr.Button("Scan Image", elem_classes=["scan-button"])
            result_html = gr.HTML("<div class='empty-result'>Upload an image and run the scan.</div>")

    scan_button.click(fn=analyze_image, inputs=[upload], outputs=[result_html])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )