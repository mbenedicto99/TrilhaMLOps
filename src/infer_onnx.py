import argparse
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import torch.nn.functional as F
import torch

def predict(text: str, model_path: str = "artifacts/model.onnx", tokenizer_dir: str = "artifacts/hf"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    enc = tokenizer(text, return_tensors="np", truncation=True, max_length=128)
    inputs = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    logits = sess.run(["logits"], inputs)[0]
    logits_t = torch.from_numpy(logits)
    probs = F.softmax(logits_t, dim=-1).squeeze().tolist()
    pred = int(np.argmax(logits, axis=-1).item())
    return {"pred": pred, "probs": probs}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()
    print(predict(args.text))
