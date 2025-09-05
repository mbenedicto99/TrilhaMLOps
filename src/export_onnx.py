import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def export(model_dir="artifacts/hf", out_path="artifacts/model.onnx", opset=13):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    dummy = tokenizer("hello world", return_tensors="pt", truncation=True, max_length=128)
    inputs = (dummy["input_ids"], dummy["attention_mask"])
    dynamic_axes = {"input_ids": {0: "batch", 1: "sequence"}, "attention_mask": {0: "batch", 1: "sequence"}, "logits": {0: "batch"}}

    with torch.no_grad():
        torch.onnx.export(
            model,
            args=inputs,
            f=out_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
        )
    print(f"Exported ONNX to {out_path}")

if __name__ == "__main__":
    export()
