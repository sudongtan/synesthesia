from fastai.basics import *
from fastai.text.models.transformer import TransformerXL
from utils.attention_mask import rand_window_mask
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from dataloader import load_image_data
import torch
import numpy as np


model = torch.hub.load("pytorch/vision:v0.4.2", "resnet18", pretrained=True)

feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
feature_extractor.eval()
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

SEED = 12
np.random.seed(SEED)


class MusicTransformerXL(TransformerXL):
    "Exactly like fastai's TransformerXL, but with more aggressive attention mask: see `rand_window_mask`"

    def __init__(self, *args, encode_position=True, mask_steps=1, **kwargs):
        import inspect

        sig = inspect.signature(TransformerXL)
        arg_params = {k: kwargs[k] for k in sig.parameters if k in kwargs}
        super().__init__(*args, **arg_params)

        self.encode_position = encode_position
        if self.encode_position:
            self.beat_enc = BeatPositionEncoder(kwargs["d_model"])

        self.mask_steps = mask_steps
        self.image_paths = load_image_data()
        self.current_emotions = torch.ones(()).new_empty(2).long().to("cuda")
        # print("Original emotion: ", self.current_emotions)

    def forward(self, z):
        # The hidden state has to be initiliazed in the forward pass for nn.DataParallel
        x = z
        # print("Current emotions: ", self.current_emotions)

        if self.mem_len > 0 and not self.init:
            self.reset()
            self.init = True

        if "emo" in x.keys():

            emotions = x["emo"][:, 0]
            if self.current_emotions[0].size() == 0:
                self.current_emotions = emotions
                # print("Initial set of emotions: ", emotions)

            else:
                # print("1", emotions)
                # print("2", self.current_emotions)
                # print(torch.equal(emotions, self.current_emotions))
                if not torch.equal(emotions, self.current_emotions):
                    init_hidden = self.init_to_hidden(emotions)
                    self.hidden = init_hidden
                    # print(
                    #     "Emotions reset: ",
                    #     self.current_emotions,
                    #     "=>",
                    #     emotions,
                    # )
                    # print("Hidden reset  :", self.hidden[0])
                    self.current_emotions = emotions
        else:
            pass

        benc = 0
        if self.encode_position:
            x, pos = x["x"], x["pos"]
            # pos = pos.to("cuda")
            benc = self.beat_enc(pos)

        bs, x_len = x.size()
        inp = self.drop_emb(
            self.encoder(x) + benc
        )  # .mul_(self.d_model ** 0.5)
        m_len = (
            self.hidden[0].size(1)
            if hasattr(self, "hidden") and len(self.hidden[0].size()) > 1
            else 0
        )
        seq_len = m_len + x_len

        mask = (
            rand_window_mask(
                x_len,
                m_len,
                inp.device,
                max_size=self.mask_steps,
                is_eval=not self.training,
            )
            if self.mask
            else None
        )

        if m_len == 0:
            mask[..., 0, 0] = 0
        # [None,:,:None] for einsum implementation of attention
        hids = []
        pos = torch.arange(
            seq_len - 1, -1, -1, device=inp.device, dtype=inp.dtype
        )
        pos_enc = self.pos_enc(pos)
        hids.append(inp)
        for i, layer in enumerate(self.layers):
            mem = self.hidden[i] if self.mem_len > 0 else None
            inp = layer(inp, r=pos_enc, u=self.u, v=self.v, mask=mask, mem=mem)
            hids.append(inp)
        core_out = inp[:, -x_len:]
        if self.mem_len > 0:
            self._update_mems(hids)
        # print("Current hidden: ", len(self.hidden), self.hidden[0].size())
        # print()
        # print()
        return (self.hidden if self.mem_len > 0 else [core_out]), [core_out]

    def load_feature_map(self, emotions):
        emotions = emotions.tolist()
        image_paths = [
            np.random.choice(self.image_paths[emotion], 1)[0]
            for emotion in emotions
        ]
        input_batch = []
        for image_path in image_paths:
            input_image = Image.open(image_path)
            input_tensor = preprocess(input_image)
            input_batch.append(input_tensor)
        input_batch = torch.stack(input_batch)

        # input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to("cuda")
            feature_extractor.to("cuda")

        with torch.no_grad():
            output = feature_extractor(input_batch)
        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        feature_map = output[..., 0]
        # print("feature_map", feature_map.shape)
        return feature_map.transpose(1, 2)

    def load_feature_map_predict(self, image_paths):
        input_batch = []
        for image_path in image_paths:
            input_image = Image.open(image_path)
            input_tensor = preprocess(input_image)
            input_batch.append(input_tensor)
        input_batch = torch.stack(input_batch)
        if torch.cuda.is_available():
            input_batch = input_batch.to("cuda")
            feature_extractor.to("cuda")

        with torch.no_grad():
            output = feature_extractor(input_batch)
        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        feature_map = output[..., 0]
        print("feature_map", feature_map.shape)
        return feature_map.transpose(1, 2)

    def init_to_hidden(self, emotions):
        batch_size = len(emotions)
        out = self.load_feature_map(emotions)
        # out = torch.ones((batch_size, 1, 512)).to("cuda")
        # out = torch.ones(0).to("cuda")
        # print("image features:", out.size())
        return [out] * (self.n_layers + 1)

    def init_to_hidden_predict(self, image_paths):
        out = self.load_feature_map_predict(image_paths)
        # out = torch.ones((1, 1, 512)).to("cuda")
        return [out] * (self.n_layers + 1)


# Beat encoder
class BeatPositionEncoder(nn.Module):
    "Embedding + positional encoding + dropout"

    def __init__(self, emb_sz: int, beat_len=32, max_bar_len=1024):
        super().__init__()

        self.beat_len, self.max_bar_len = beat_len, max_bar_len
        self.beat_enc = nn.Embedding(beat_len, emb_sz, padding_idx=0)
        self.bar_enc = nn.Embedding(max_bar_len, emb_sz, padding_idx=0)

    def forward(self, pos):
        beat_enc = self.beat_enc(pos % self.beat_len)
        bar_pos = pos // self.beat_len % self.max_bar_len
        bar_pos[bar_pos >= self.max_bar_len] = self.max_bar_len - 1
        bar_enc = self.bar_enc((bar_pos))
        return beat_enc + bar_enc
