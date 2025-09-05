import os
import random
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
from omegaconf import DictConfig, OmegaConf

from src.data_utils import get_dataloaders
from src.model import TextClassifier

def maybe_wandb_logger(cfg):
    use_wb = bool(cfg.logging.get("use_wandb", False))
    if use_wb and os.environ.get("WANDB_API_KEY"):
        import wandb
        from pytorch_lightning.loggers import WandbLogger
        wandb.login()
        return WandbLogger(project=cfg.logging.get("project", "mlops-trilha-minima"))
    return CSVLogger(save_dir="logs", name="runs")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    pl.seed_everything(seed, workers=True)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print("Config:\n", OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    train_dl, val_dl, test_dl, tokenizer = get_dataloaders(
        model_name=cfg.model.name,
        dataset_name=cfg.data.dataset_name,
        max_length=cfg.data.max_length,
        batch_size=cfg.data.batch_size,
    )

    model = TextClassifier(
        model_name=cfg.model.name,
        num_labels=cfg.model.num_labels,
        lr=cfg.model.lr,
        weight_decay=cfg.model.weight_decay,
    )

    logger = maybe_wandb_logger(cfg)
    ckpt = ModelCheckpoint(monitor="val_f1", mode="max", save_top_k=1, dirpath="artifacts", filename="model")

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        logger=logger,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        callbacks=[ckpt],
    )

    trainer.fit(model, train_dl, val_dl)
    trainer.test(model, test_dl, ckpt_path=ckpt.best_model_path if ckpt.best_model_path else None)
    print(f"Best checkpoint: {ckpt.best_model_path}")

if __name__ == "__main__":
    main()
