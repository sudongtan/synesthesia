import os
from pathlib import Path
from utils.numpy_encode import *
from dataloader import *
from learner import *
from config import *
import torch
from fastai.callbacks import SaveModelCallback
import argparse
import shutil

SEED = 1234567890
np.random.seed(SEED)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", type=int, default=8, help="batch size")
    parser.add_argument(
        "-i", type=int, default=2, help="number of epochs",
    )

    args = parser.parse_args()
    config = default_config()
    encode_position = True
    data_dir = Path("../dataset/midi/transformer")
    data_name = "data_sample.pkl"

    dl_tfms = [batch_position_tfm] if encode_position else []
    data = load_data(
        data_dir,
        data_name,
        bs=args.bs,
        encode_position=encode_position,
        dl_tfms=dl_tfms,
    )

    learn = music_model_learner(data, config=config.copy())

    model_name = "transformer_example"
    callbacks = [
        SaveModelCallback(
            learn, every="improvement", monitor="valid_loss", name=model_name
        )
    ]
    learn.fit_one_cycle(args.i, callbacks=callbacks)
    model_dir_old = os.path.join(data_dir, "models")
    model_path_old = os.path.join(data_dir, "models", model_name)
    model_path = os.path.join("../model", model_name)
    shutil.move(model_path_old, model_path)
    print("Model saved as ", model_path)
