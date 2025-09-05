from fastapi import FastAPI
from pydantic import BaseModel
import os
import onnxruntime as ort
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.onnx")
TOKENIZER_DIR = os.getenv("TOKENIZER_DIR", "artifacts/hf")

app = FastAPI(title="MLOps Trilha Mínima API", version="0.1.0")

class InferenceRequest(BaseModel):
    text: str

@app.on_event("startup")
def load():
    global _sess, _tok
    _tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    _sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: InferenceRequest):
    enc = _tok(req.text, return_tensors="np", truncation=True, max_length=128)
    inputs = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    logits = _sess.run(["logits"], inputs)[0]
    logits_t = torch.from_numpy(logits)
    probs = F.softmax(logits_t, dim=-1).squeeze().tolist()
    pred = int(logits.argmax(axis=-1).item())
    return {"pred": pred, "probs": probs}
