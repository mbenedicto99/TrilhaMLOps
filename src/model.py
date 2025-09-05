from typing import Any, Dict
import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

class TextClassifier(pl.LightningModule):
    def __init__(self, model_name: str, num_labels: int = 4, lr: float = 5e-5, weight_decay: float = 0.01):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.acc = MulticlassAccuracy(num_classes=num_labels, average="macro")
        self.f1 = MulticlassF1Score(num_classes=num_labels, average="macro")

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        out = self(**batch)
        preds = out.logits.argmax(dim=-1)
        acc = self.acc(preds, batch["labels"])
        f1 = self.f1(preds, batch["labels"])
        self.log_dict({"train_loss": out.loss, "train_acc": acc, "train_f1": f1}, prog_bar=True, on_step=True, on_epoch=True)
        return out.loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        out = self(**batch)
        preds = out.logits.argmax(dim=-1)
        acc = self.acc(preds, batch["labels"])
        f1 = self.f1(preds, batch["labels"])
        self.log_dict({"val_loss": out.loss, "val_acc": acc, "val_f1": f1}, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        out = self(**batch)
        preds = out.logits.argmax(dim=-1)
        acc = self.acc(preds, batch["labels"])
        f1 = self.f1(preds, batch["labels"])
        self.log_dict({"test_loss": out.loss, "test_acc": acc, "test_f1": f1}, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def on_train_end(self) -> None:
        # Salva pesos HF finetunados para export ONNX posterior
        self.model.save_pretrained("artifacts/hf")
