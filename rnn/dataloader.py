import os
import torch
import itertools, os
import numpy as np
from progress.bar import Bar

import config
import utils
from sequence import EventSeq, ControlSeq
from collections import defaultdict
from sklearn.utils import shuffle

# pylint: disable=E1101
# pylint: disable=W0101
SEED = 12345
np.random.seed(SEED)


class Dataset:
    def __init__(self, root, verbose=False):
        assert os.path.isdir(root), root
        paths = utils.find_files_by_extensions(root, [".data"])
        self.root = root
        self.labels = [
            "excitement",
            "contentment",
            "sadness",
            "amusement",
            "anger",
        ]
        self.midi_samples = defaultdict(list)
        self.midi_seqlens = defaultdict(list)
        self.image_samples = {}

        self.samples = []
        self.seqlens = []
        if verbose:
            paths = Bar(root).iter(list(paths))
        for path in paths:
            label = os.path.basename(path).split("_")[0]
            eventseq, controlseq = torch.load(path)
            controlseq = ControlSeq.recover_compressed_array(controlseq)
            assert len(eventseq) == len(controlseq)
            self.samples.append((eventseq, controlseq))
            self.seqlens.append(len(eventseq))
            self.midi_samples[label].append((eventseq, controlseq))
            self.midi_seqlens[label].append(len(eventseq))

        image_dir = "../dataset/image/train"
        for label in self.labels:
            class_dir = os.path.join(image_dir, label)
            names = os.listdir(class_dir)
            image_paths = [os.path.join(class_dir, name) for name in names]
            self.image_samples[label] = image_paths

        self.avglen = np.mean(self.seqlens)

    def paired_batches(self, batch_size, window_size, stride_size):
        eventseq_batch = []
        controlseq_batch = []
        image_batch = []
        label_batch = []
        s = int(batch_size / 5)
        l = batch_size - 4 * s
        sample_sizes = [s, s, s, s, l]
        np.random.shuffle(sample_sizes)

        for label, sample_size in zip(self.labels, sample_sizes):
            indeces = [
                (i, range(j, j + window_size))
                for i, seqlen in enumerate(self.midi_seqlens[label])
                for j in range(0, seqlen - window_size, stride_size)
            ]
            np.random.shuffle(indeces)
            samples = indeces[:sample_size]

            for i, r in samples:
                eventseq, controlseq = self.midi_samples[label][i]
                eventseq = eventseq[r.start : r.stop]
                controlseq = controlseq[r.start : r.stop]

                eventseq_batch.append(eventseq)
                controlseq_batch.append(controlseq)
                label_batch.append(label)

            image_batch.extend(
                np.random.choice(self.image_samples[label], sample_size)
            )

        # shuffle
        eventseq_batch, controlseq_batch, image_batch, label_batch = shuffle(
            eventseq_batch,
            controlseq_batch,
            image_batch,
            label_batch,
            random_state=SEED,
        )

        result = (
            np.stack(eventseq_batch, axis=1),
            np.stack(controlseq_batch, axis=1),
            image_batch,
            label_batch,
        )
        # print(label_batch[0], image_batch[0])
        return result

    def batches(self, batch_size, window_size, stride_size):
        indeces = [
            (i, range(j, j + window_size))
            for i, seqlen in enumerate(self.seqlens)
            for j in range(0, seqlen - window_size, stride_size)
        ]
        while True:
            eventseq_batch = []
            controlseq_batch = []
            n = 0
            for ii in np.random.permutation(len(indeces)):
                i, r = indeces[ii]
                eventseq, controlseq = self.samples[i]
                eventseq = eventseq[r.start : r.stop]
                controlseq = controlseq[r.start : r.stop]
                # print("event: ", eventseq.shape)
                eventseq_batch.append(eventseq)
                controlseq_batch.append(controlseq)
                n += 1
                if n == batch_size:
                    yield (
                        np.stack(eventseq_batch, axis=1),
                        np.stack(controlseq_batch, axis=1),
                    )
                    eventseq_batch.clear()
                    controlseq_batch.clear()
                    n = 0

    def __repr__(self):
        return (
            f'Dataset(root="{self.root}", '
            f"samples={len(self.samples)}, "
            f"avglen={self.avglen})"
        )
