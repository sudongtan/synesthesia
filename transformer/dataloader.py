"Fastai Language Model Databunch modified to work with music"
from fastai.basics import *

# from fastai.basic_data import DataBunch
from fastai.text.data import LMLabelList
from utils.transform import *
from utils.vocab import MusicVocab


class MusicDataBunch(DataBunch):
    "Create a `TextDataBunch` suitable for training a language model."

    @classmethod
    def create(
        cls,
        train_ds,
        valid_ds,
        test_ds=None,
        path: PathOrStr = ".",
        no_check: bool = False,
        bs=64,
        val_bs: int = None,
        num_workers: int = 0,
        device: torch.device = None,
        collate_fn: Callable = data_collate,
        dl_tfms: Optional[Collection[Callable]] = None,
        bptt: int = 70,
        preloader_cls=None,
        shuffle_dl=False,
        transpose_range=(0, 12),
        **kwargs
    ) -> DataBunch:
        "Create a `TextDataBunch` in `path` from the `datasets` for language modelling."
        datasets = cls._init_ds(train_ds, valid_ds, test_ds)
        preloader_cls = (
            MusicPreloader if preloader_cls is None else preloader_cls
        )
        val_bs = ifnone(val_bs, bs)
        datasets = [
            preloader_cls(
                ds,
                shuffle=(i == 0),
                bs=(bs if i == 0 else val_bs),
                bptt=bptt,
                transpose_range=transpose_range,
                **kwargs
            )
            for i, ds in enumerate(datasets)
        ]
        val_bs = bs
        dl_tfms = [
            partially_apply_vocab(tfm, train_ds.vocab)
            for tfm in listify(dl_tfms)
        ]
        dls = [
            DataLoader(d, b, shuffle=shuffle_dl)
            for d, b in zip(datasets, (bs, val_bs, val_bs, val_bs))
            if d is not None
        ]
        return cls(
            *dls,
            path=path,
            device=device,
            dl_tfms=dl_tfms,
            collate_fn=collate_fn,
            no_check=no_check
        )

    @classmethod
    def from_folder(cls, path: PathOrStr, extensions=".npy", **kwargs):
        files = get_files(path, extensions=extensions, recurse=True)
        return cls.from_files(files, path, **kwargs)

    @classmethod
    def from_files(
        cls,
        files,
        path,
        processors=None,
        split_pct=0.1,
        vocab=None,
        list_cls=None,
        **kwargs
    ):

        # print("Before processing: ")
        # print(files)
        # print()
        if vocab is None:
            vocab = MusicVocab.create()
        if list_cls is None:
            list_cls = MusicItemList
        src = (
            list_cls(items=files, path=path, processor=processors, vocab=vocab)
            .split_by_rand_pct(split_pct, seed=6)
            .label_const(label_cls=LMLabelList)
        )
        result = src.databunch(**kwargs)
        # print("After processing")
        # print(result.get(0).__dict__)
        # print()
        # print("Shape of the midi numpy: ", result.shape)
        return result

    # @classmethod
    # def load_images(self, images):
    #     model = torch.hub.load("pytorch/vision:v0.4.2", "resnet18", pretrained=True)

    #     feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    #     feature_extractor.eval()
    #     preprocess = transforms.Compose(
    #         [
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #         ]
    #     )

    #     input_batch = []
    #     for image_path in images:
    #         input_image = Image.open(image_path)
    #         input_tensor = preprocess(input_image)
    #         input_batch.append(input_tensor)
    #     input_batch = torch.stack(input_batch)

    #     #input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    #     # move the input and model to GPU for speed if available
    #     if torch.cuda.is_available():
    #         input_batch = input_batch.to("cuda")
    #         feature_extractor.to("cuda")

    #     with torch.no_grad():
    #         output = feature_extractor(input_batch)
    #     # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    #     feature_map = output[..., 0]
    #     # print('feature_map', feature_map.shape)
    #     return feature_map.transpose(1, 2)

    @classmethod
    def empty(cls, path, **kwargs):
        vocab = MusicVocab.create()
        src = MusicItemList(
            [], path=path, vocab=vocab, ignore_empty=True
        ).split_none()
        result = src.label_const(label_cls=LMLabelList).databunch()
        return result


def partially_apply_vocab(tfm, vocab):
    if "vocab" in inspect.getfullargspec(tfm).args:
        return partial(tfm, vocab=vocab)
    return tfm


class MusicItemList(ItemList):
    _bunch = MusicDataBunch

    def __init__(self, items: Iterator, vocab: MusicVocab = None, **kwargs):
        super().__init__(items, **kwargs)
        self.vocab = vocab
        self.copy_new += ["vocab"]

        # print("Music list")
        # for item in items:
        #     print("here", item)

    def get(self, i):
        o = super().get(i)
        if is_pos_enc(o):
            return MusicItem.from_idx(o, self.vocab)

        # print("Not from idx")
        # print(o)
        # print()
        return MusicItem.from_idx(o, self.vocab)


def is_pos_enc(idxenc):
    if len(idxenc.shape) == 2 and idxenc.shape[0] == 2:
        return True
    return idxenc.dtype == np.object and idxenc.shape == (2,)


class MusicItemProcessor(PreProcessor):
    "`PreProcessor` that transforms numpy files to indexes for training"

    def process_one(self, item):
        item = MusicItem.from_npenc(item, vocab=self.vocab)
        return item.to_idx()

    def process(self, ds):
        self.vocab = ds.vocab
        super().process(ds)


class OpenNPFileProcessor(PreProcessor):
    "`PreProcessor` that opens the filenames and read the texts."

    def process_one(self, item):
        return (
            np.load(item, allow_pickle=True) if isinstance(item, Path) else item
        )


class Midi2ItemProcessor(PreProcessor):
    "Skips midi preprocessing step. And encodes midi files to MusicItems"

    def process_one(self, item):
        item = MusicItem.from_file(item, vocab=self.vocab)
        return item.to_idx()

    def process(self, ds):
        self.vocab = ds.vocab
        super().process(ds)


## For npenc dataset
class MusicPreloader(Callback):
    "Transforms the tokens in `dataset` to a stream of contiguous batches for language modelling."

    class CircularIndex:
        "Handles shuffle, direction of indexing, wraps around to head tail in the ragged array as needed"

        def __init__(self, length: int, forward: bool):
            self.idx, self.forward = np.arange(length), forward

        def __getitem__(self, i):
            return self.idx[
                i % len(self.idx)
                if self.forward
                else len(self.idx) - 1 - i % len(self.idx)
            ]

        def __len__(self) -> int:
            return len(self.idx)

        def shuffle(self):
            np.random.shuffle(self.idx)

    def __init__(
        self,
        dataset: LabelList,
        lengths: Collection[int] = None,
        bs: int = 32,
        bptt: int = 70,
        backwards: bool = False,
        shuffle: bool = False,
        y_offset: int = 1,
        transpose_range=None,
        transpose_p=0.5,
        encode_position=True,
        **kwargs
    ):
        (
            self.dataset,
            self.bs,
            self.bptt,
            self.shuffle,
            self.backwards,
            self.lengths,
        ) = (dataset, bs, bptt, shuffle, backwards, lengths)
        self.vocab = self.dataset.vocab
        self.bs *= num_distrib() or 1
        self.totalToks, self.ite_len, self.idx = int(0), None, None
        self.y_offset = y_offset

        self.transpose_range, self.transpose_p = transpose_range, transpose_p
        self.encode_position = encode_position
        self.bptt_len = self.bptt

        self.allocate_buffers()  # needed for valid_dl on distributed training - otherwise doesn't get initialized on first epoch

    def __len__(self):
        if self.ite_len is None:
            if self.lengths is None:
                self.lengths = np.array([len(item) for item in self.dataset.x])
            self.totalToks = self.lengths.sum()
            self.ite_len = (
                self.bs * int(math.ceil(self.totalToks / (self.bptt * self.bs)))
                if self.item is None
                else 1
            )
        return self.ite_len

    def __getattr__(self, k: str) -> Any:
        return getattr(self.dataset, k)

    def allocate_buffers(self):
        "Create the ragged array that will be filled when we ask for items."
        if self.ite_len is None:
            len(self)
        self.idx = MusicPreloader.CircularIndex(
            len(self.dataset.x), not self.backwards
        )

        # batch shape = (bs, bptt, 2 - [index, pos]) if encode_position. Else - (bs, bptt)
        buffer_len = (3,) if self.encode_position else ()
        self.batch = np.zeros(
            (self.bs, self.bptt + self.y_offset) + buffer_len, dtype=np.int64
        )
        # print("batch size: ", self.batch.shape)
        self.batch_x, self.batch_y = (
            self.batch[:, 0 : self.bptt],
            self.batch[:, self.y_offset : self.bptt + self.y_offset],
        )
        # ro: index of the text we're at inside our datasets for the various batches
        self.ro = np.zeros(self.bs, dtype=np.int64)
        # ri: index of the token we're at inside our current text for the various batches
        self.ri = np.zeros(self.bs, dtype=np.int)

        # allocate random transpose values. Need to allocate this before hand.
        self.transpose_values = self.get_random_transpose_values()

    def get_random_transpose_values(self):
        if self.transpose_range is None:
            return None
        n = len(self.dataset)
        rt_arr = (
            torch.randint(*self.transpose_range, (n,))
            - self.transpose_range[1] // 2
        )
        mask = torch.rand(rt_arr.shape) > self.transpose_p
        rt_arr[mask] = 0
        return rt_arr

    def on_epoch_begin(self, **kwargs):
        if self.idx is None:
            # print("alocate buffer:")
            self.allocate_buffers()
        elif self.shuffle:
            # print("Shuffle")
            self.ite_len = None
            self.idx.shuffle()
            self.transpose_values = self.get_random_transpose_values()
            self.bptt_len = self.bptt
        self.idx.forward = not self.backwards

        step = self.totalToks / self.bs
        ln_rag, countTokens, i_rag = 0, 0, -1
        for i in range(0, self.bs):
            # Compute the initial values for ro and ri
            while ln_rag + countTokens <= int(step * i):
                countTokens += ln_rag
                i_rag += 1
                ln_rag = self.lengths[self.idx[i_rag]]
            self.ro[i] = i_rag
            self.ri[i] = (
                (ln_rag - int(step * i - countTokens))
                if self.backwards
                else int(step * i - countTokens)
            )
            # print("begin: ", self.ro.shape, self.ri.shape)

    # Training dl gets on_epoch_begin called, val_dl, on_epoch_end
    def on_epoch_end(self, **kwargs):
        self.on_epoch_begin()

    def __getitem__(self, k: int):

        ## TODO: add emotion
        j = k % self.bs
        if j == 0:
            if self.item is not None:
                # print("I am item")
                return self.dataset[0]
            if self.idx is None:
                # print("I am starting")
                self.on_epoch_begin()

        # print("Batch: ", self.batch.shape, self.ro.shape, self.ri.shape)
        # print("Item:  ", self.batch[j][: self.bptt_len + self.y_offset])

        self.ro[j], self.ri[j] = self.fill_row(
            not self.backwards,
            self.dataset.x,
            self.idx,
            self.batch[j][: self.bptt_len + self.y_offset],
            self.ro[j],
            self.ri[j],
            overlap=1,
            lengths=self.lengths,
        )

        # print(self.ro)
        # print(self.ri)
        result = (
            self.batch_x[j][: self.bptt_len],
            self.batch_y[j][: self.bptt_len],
        )
        return result

    def fill_row(self, forward, items, idx, row, ro, ri, overlap, lengths):
        "Fill the row with tokens from the ragged array. --OBS-- overlap != 1 has not been implemented"
        ibuf = n = 0
        ro -= 1
        while ibuf < row.shape[0]:
            ro += 1
            ix = idx[ro]

            item = items[ix]
            emotion = item._emotion
            # print("Item: ", item.__dict__)
            if self.transpose_values is not None:
                item = item.transpose(self.transpose_values[ix].item())
                item.emotion = emotion
                # print("After: ", item.__dict__)

            if self.encode_position:

                # Positions are colomn stacked with indexes. This makes it easier to keep in sync
                rag = np.stack(
                    [item.data, item.position, len(item.data) * [item.emotion]],
                    axis=1,
                )
                # print("I am ragging: ", rag.shape, item.emotion)
                # print("item: ")
                # print(item.__dict__)
                # print(rag)
            else:
                rag = item.data

            if forward:
                ri = 0 if ibuf else ri
                n = min(lengths[ix] - ri, row.shape[0] - ibuf)
                # print("no casting", row.shape, rag[ri : ri + n].shape)
                # print(rag)
                # print("current: ", rag[ri : ri + n])
                row[ibuf : ibuf + n] = rag[ri : ri + n]
            else:
                ri = lengths[ix] if ibuf else ri
                n = min(ri, row.size - ibuf)
                row[ibuf : ibuf + n] = rag[ri - n : ri][::-1]
            ibuf += n
        return ro, ri + ((n - overlap) if forward else -(n - overlap))


def batch_position_tfm(b):
    "Batch transform for training with positional encoding"
    # print("Batch")
    # print(b)
    x, y = b

    # print("Data batch:")
    # print(type(b), len(b))
    # print("X")
    # print(x)
    # print("Y")
    # print(y)

    x = {"x": x[..., 0], "pos": x[..., 1], "emo": x[..., 2]}
    # TODO: how to carry emotion here

    return x, y[..., 0]


import os


def load_image_data():
    emotion_idx = {
        "sadness": 0,
        "anger": 1,
        "excitement": 2,
        "contentment": 3,
        "amusement": 4,
    }
    result = {}

    image_dir = "../dataset/image/train"

    for e, idx in emotion_idx.items():
        class_dir = os.path.join(image_dir, e)

        names = os.listdir(class_dir)
        image_paths = [os.path.join(class_dir, name) for name in names]
        result[idx] = image_paths

    return result

