import argparse
import os

import yaml
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger

from models import AttentiveDeepSym
from dataset import StateActionEffectDM


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train AttentiveDeepSym model.")
    parser.add_argument("-c", "--config", help="Configuration file path", type=str, required=True)
    args = parser.parse_args()


    with open(os.path.join("..", args.config), "r") as f:
        config = yaml.safe_load(f)


    log_dir = os.path.join("../logs", config["name"])
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=log_dir,
        save_last=False, 
        save_top_k=1, 
        monitor="val_loss",
        mode="min"
    )


    logger = TensorBoardLogger("logs", name=config["name"])


    trainer = pl.Trainer(
        max_epochs=config["epoch"], 
        gradient_clip_val=10.0,
        logger=logger, 
        devices=config["devices"], 
        callbacks=[ckpt_callback]
    )

  
    model = AttentiveDeepSym(config)

  
    dm = StateActionEffectDM(
        config["dataset_name"], 
        batch_size=config["batch_size"]
    )

    trainer.fit(model, datamodule=dm)
    
    best_ckpt_path = ckpt_callback.best_model_path
    new_ckpt_path = os.path.join(log_dir, "best.ckpt")
    if best_ckpt_path != new_ckpt_path:
        os.rename(best_ckpt_path, new_ckpt_path)
