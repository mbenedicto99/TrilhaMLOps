import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

def predict(text: str, model_dir: str = "artifacts/hf"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        out = model(**inputs).logits
        probs = F.softmax(out, dim=-1).squeeze().tolist()
        pred = int(out.argmax(dim=-1).item())
    return {"pred": pred, "probs": probs}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()
    print(predict(args.text))
