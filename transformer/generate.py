import argparse
import os
from config import default_config
from dataloader import MusicDataBunch
from learner import music_model_learner
import numpy as np

SEED = 1234567890
np.random.seed(SEED)
config = default_config()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, help="local path of the image")
    parser.add_argument(
        "-n",
        type=int,
        default=1000,
        help="number of words, controling the lengths of the generated midi",
    )
    parser.add_argument(
        "-m", type=str, default="transformer", help="name of the model"
    )

    args = parser.parse_args()
    model = music_model_learner(
        data=MusicDataBunch.empty("../dataset/midi/transformer/"),
        config=config,
        pretrained_path=f"../model/{args.m}.pth",
    )
    pred, _ = model.predict_from_image(image_path=args.i, n_words=args.n)
    output_name = os.path.basename(args.i).split(".")[0] + ".MID"
    output_path = os.path.join("../output/transformer", output_name)
    stream = pred.to_stream()
    stream.write("midi", fp=output_path)
    print(f"generated midi saved to {output_path}")
