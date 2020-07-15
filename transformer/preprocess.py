import os
from pathlib import Path
from pathlib import Path
from utils.numpy_encode import *
from dataloader import *
from learner import *
import argparse


if __name__ == "__main__":
    midi_path = Path("../dataset/midi/train")
    midi_path.mkdir(parents=True, exist_ok=True)

    # Location to save dataset
    save_dir = Path("../dataset/midi/transformer")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_name = "data_sample.pkl"
    save_path = os.path.join(save_dir, save_name)

    midi_files = get_files(midi_path, ".mid", recurse=True)
    print("Number of midi files: ", len(midi_files))

    processors = [Midi2ItemProcessor()]
    data = MusicDataBunch.from_files(
        midi_files, save_dir, processors=processors, bs=2, bptt=12
    )
    data.save(save_name)
    print("Data saved as ", save_path)
