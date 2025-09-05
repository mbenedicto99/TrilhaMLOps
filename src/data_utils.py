import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

def get_dataloaders(model_name: str, dataset_name: str = "ag_news", max_length: int = 128, batch_size: int = 16):
    ds = load_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dl = DataLoader(tokenized["train"], batch_size=batch_size, shuffle=True, collate_fn=collator)
    test_dl = DataLoader(tokenized["test"], batch_size=batch_size, shuffle=False, collate_fn=collator)

    # Criar validação simples a partir do train (pequeno split)
    val_size = min(4000, len(tokenized["train"]))
    val_subset = torch.utils.data.Subset(tokenized["train"], range(val_size))
    val_dl = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    return train_dl, val_dl, test_dl, tokenizer
