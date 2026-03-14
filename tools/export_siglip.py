"""
Export google/siglip-base-patch16-256-multilingual to two ONNX files:
  - siglip_onnx/vision_model.onnx  (pixel_values → image_embeds)
  - siglip_onnx/text_model.onnx    (input_ids → text_embeds)

Also copies tokenizer files for Go consumption.
"""

import os, json, torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoProcessor

MODEL_ID = "google/siglip-base-patch16-256-multilingual"
OUT_DIR = "siglip_onnx"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading model …")
model = AutoModel.from_pretrained(MODEL_ID)
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# ── Save tokenizer files ──────────────────────────────────────────────────
tokenizer.save_pretrained(OUT_DIR)
processor.save_pretrained(OUT_DIR)
print("Tokenizer files saved to", OUT_DIR)

# ── Inspect model output names ────────────────────────────────────────────
# SigLIP's forward returns image_embeds & text_embeds when given both inputs.
# We export each encoder separately.

# ── Vision model ─────────────────────────────────────────────────────────
class VisionEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vision = model.vision_model
        self.head   = model.visual_projection if hasattr(model, 'visual_projection') else None

    def forward(self, pixel_values):
        out = self.vision(pixel_values=pixel_values)
        pooled = out.pooler_output          # (B, hidden)
        if self.head is not None:
            pooled = self.head(pooled)
        return pooled

vision_enc = VisionEncoder(model)
vision_enc.eval()

dummy_img = torch.randn(1, 3, 256, 256)
vision_out = vision_enc(dummy_img)
emb_dim = vision_out.shape[-1]
print(f"Vision emb_dim = {emb_dim}")

print("Exporting vision model …")
torch.onnx.export(
    vision_enc,
    (dummy_img,),
    os.path.join(OUT_DIR, "vision_model.onnx"),
    input_names=["pixel_values"],
    output_names=["image_embeds"],
    dynamic_axes={"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}},
    opset_version=18,
)
print("  ✓ vision_model.onnx")

# ── Text model ───────────────────────────────────────────────────────────
class TextEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.text = model.text_model
        self.head = model.text_projection if hasattr(model, 'text_projection') else None

    def forward(self, input_ids):
        out = self.text(input_ids=input_ids)
        pooled = out.pooler_output
        if self.head is not None:
            pooled = self.head(pooled)
        return pooled

text_enc = TextEncoder(model)
text_enc.eval()

MAX_LEN = 64   # SigLIP default
dummy_ids = torch.randint(0, 250000, (1, MAX_LEN), dtype=torch.long)
text_out = text_enc(dummy_ids)
text_emb_dim = text_out.shape[-1]
print(f"Text emb_dim = {text_emb_dim}")

print("Exporting text model …")
torch.onnx.export(
    text_enc,
    (dummy_ids,),
    os.path.join(OUT_DIR, "text_model.onnx"),
    input_names=["input_ids"],
    output_names=["text_embeds"],
    dynamic_axes={"input_ids": {0: "batch"}, "text_embeds": {0: "batch"}},
    opset_version=18,
)
print("  ✓ text_model.onnx")

# ── Quick sanity check with ONNX Runtime ──────────────────────────────────
import onnxruntime as ort

sess_v = ort.InferenceSession(os.path.join(OUT_DIR, "vision_model.onnx"))
sess_t = ort.InferenceSession(os.path.join(OUT_DIR, "text_model.onnx"))

# Use processor to preprocess a dummy image
from PIL import Image
dummy_pil = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
pv = processor(images=dummy_pil, return_tensors="np")["pixel_values"]
ie = sess_v.run(None, {"pixel_values": pv.astype(np.float32)})[0]
print(f"  Vision  ONNX output shape: {ie.shape}")

enc = tokenizer("hello world", padding="max_length", max_length=MAX_LEN, truncation=True, return_tensors="np")
te = sess_t.run(None, {"input_ids": enc["input_ids"].astype(np.int64)})[0]
print(f"  Text    ONNX output shape: {te.shape}")

# ── Save config for Go ───────────────────────────────────────────────────
logit_scale = model.logit_scale.item()
logit_bias  = model.logit_bias.item() if hasattr(model, 'logit_bias') and model.logit_bias is not None else 0.0

go_config = {
    "emb_dim":      int(emb_dim),
    "img_size":     256,
    "max_text_len": MAX_LEN,
    "logit_scale":  logit_scale,
    "logit_bias":   logit_bias,
    "mean":         [0.5, 0.5, 0.5],
    "std":          [0.5, 0.5, 0.5],
}
with open(os.path.join(OUT_DIR, "go_config.json"), "w") as f:
    json.dump(go_config, f, indent=2)
print(f"\ngo_config.json: {json.dumps(go_config, indent=2)}")
print("\nDone!")
