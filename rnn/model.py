import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torchvision import transforms

import numpy as np
from progress.bar import Bar
from config import device
from PIL import Image

# pylint: disable=E1101,E1102

model = torch.hub.load("pytorch/vision:v0.4.2", "resnet18", pretrained=True)


class PerformanceRNN(nn.Module):
    def __init__(
        self,
        event_dim,
        control_dim,
        init_dim,
        hidden_dim,
        gru_layers=3,
        gru_dropout=0.3,
    ):
        super().__init__()

        self.event_dim = event_dim
        self.control_dim = control_dim
        self.init_dim = init_dim
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers
        self.concat_dim = event_dim + 1 + control_dim
        self.input_dim = hidden_dim
        self.output_dim = event_dim

        self.primary_event = self.event_dim - 1

        self.inithid_fc = nn.Linear(init_dim, gru_layers * hidden_dim)
        self.inithid_fc_activation = nn.Tanh()

        self.event_embedding = nn.Embedding(event_dim, event_dim)
        self.concat_input_fc = nn.Linear(self.concat_dim, self.input_dim)
        self.concat_input_fc_activation = nn.LeakyReLU(0.1, inplace=True)

        self.gru = nn.GRU(
            self.input_dim,
            self.hidden_dim,
            num_layers=gru_layers,
            dropout=gru_dropout,
        )
        self.output_fc = nn.Linear(hidden_dim * gru_layers, self.output_dim)
        self.output_fc_activation = nn.Softmax(dim=-1)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_normal_(self.event_embedding.weight)
        nn.init.xavier_normal_(self.inithid_fc.weight)
        self.inithid_fc.bias.data.fill_(0.0)
        nn.init.xavier_normal_(self.concat_input_fc.weight)
        nn.init.xavier_normal_(self.output_fc.weight)
        self.output_fc.bias.data.fill_(0.0)

    def _sample_event(self, output, greedy=True, temperature=1.0):
        if greedy:
            return output.argmax(-1)
        else:
            output = output / temperature
            probs = self.output_fc_activation(output)
            return Categorical(probs).sample()

    def forward(self, event, control=None, hidden=None):
        # One step forward

        assert len(event.shape) == 2
        assert event.shape[0] == 1
        batch_size = event.shape[1]
        event = self.event_embedding(event)

        if control is None:
            default = torch.ones(1, batch_size, 1).to(device)
            control = torch.zeros(1, batch_size, self.control_dim).to(device)
        else:
            default = torch.zeros(1, batch_size, 1).to(device)
            assert control.shape == (1, batch_size, self.control_dim)

        concat = torch.cat([event, default, control], -1)
        input = self.concat_input_fc(concat)
        input = self.concat_input_fc_activation(input)

        _, hidden = self.gru(input, hidden)
        output = hidden.permute(1, 0, 2).contiguous()
        output = output.view(batch_size, -1).unsqueeze(0)
        output = self.output_fc(output)
        return output, hidden

    def get_primary_event(self, batch_size):
        return torch.LongTensor([[self.primary_event] * batch_size]).to(device)

    def init_to_hidden(self, init, images):
        # [batch_size, init_dim]
        batch_size = init.shape[0]
        out = self.load_feature_map(images)
        out = self.inithid_fc(out)
        out = self.inithid_fc_activation(out)
        out = out.view(self.gru_layers, batch_size, self.hidden_dim)
        return out

    def expand_controls(self, controls, steps):
        # [1 or steps, batch_size, control_dim]
        assert len(controls.shape) == 3
        assert controls.shape[2] == self.control_dim
        if controls.shape[0] > 1:
            assert controls.shape[0] >= steps
            return controls[:steps]
        return controls.repeat(steps, 1, 1)

    def generate(
        self,
        init,
        steps,
        events=None,
        controls=None,
        images=None,
        greedy=1.0,
        temperature=1.0,
        teacher_forcing_ratio=1.0,
        output_type="index",
        verbose=False,
    ):
        # init [batch_size, init_dim]
        # events [steps, batch_size] indeces
        # controls [1 or steps, batch_size, control_dim]

        batch_size = init.shape[0]
        assert init.shape[1] == self.init_dim
        assert steps > 0

        use_teacher_forcing = events is not None
        if use_teacher_forcing:
            assert len(events.shape) == 2
            assert events.shape[0] >= steps - 1
            events = events[: steps - 1]

        event = self.get_primary_event(batch_size)
        use_control = controls is not None
        if use_control:
            controls = self.expand_controls(controls, steps)
        hidden = self.init_to_hidden(init, images)

        outputs = []
        step_iter = range(steps)
        if verbose:
            step_iter = Bar("Generating").iter(step_iter)

        for step in step_iter:
            control = controls[step].unsqueeze(0) if use_control else None
            output, hidden = self.forward(event, control, hidden)

            use_greedy = np.random.random() < greedy
            event = self._sample_event(
                output, greedy=use_greedy, temperature=temperature
            )

            if output_type == "index":
                outputs.append(event)
            elif output_type == "softmax":
                outputs.append(self.output_fc_activation(output))
            elif output_type == "logit":
                outputs.append(output)
            else:
                assert False

            if use_teacher_forcing and step < steps - 1:  # avoid last one
                if np.random.random() <= teacher_forcing_ratio:
                    event = events[step].unsqueeze(0)

        return torch.cat(outputs, 0)

    def beam_search(
        self,
        init,
        steps,
        beam_size,
        controls=None,
        temperature=1.0,
        verbose=False,
    ):
        assert len(init.shape) == 2 and init.shape[1] == self.init_dim
        assert self.event_dim >= beam_size > 0 and steps > 0

        batch_size = init.shape[0]
        use_control = controls is not None
        if use_control:
            controls = self.expand_controls(controls, steps)

        init = torch.randn(batch_size, self.init_dim).to(device)
        hidden = self.init_to_hidden(init)
        hidden = hidden.unsqueeze(2).repeat(1, 1, beam_size, 1)
        # [gru_layers, batch_size, beam_size, hidden_dim]

        event = self.get_primary_event(batch_size)
        event = event.unsqueeze(-1).repeat(1, 1, beam_size)
        # [1, batch_size, beam_size]

        beam = torch.zeros(batch_size, beam_size, steps).long().to(device)
        score = torch.zeros(batch_size, beam_size).to(device)

        step_iter = range(steps)
        if verbose:
            step_iter = Bar("Beam Search").iter(step_iter)

        for step in step_iter:
            if use_control:
                control = controls[step].unsqueeze(0).unsqueeze(2)
                control = control.repeat(1, 1, beam_size, 1)
                # [1, batch_size, beam_size, control_dim]
                control = control.view(
                    1, batch_size * beam_size, self.control_dim
                )
            else:
                control = None

            event = event.view(1, batch_size * beam_size)
            hidden = hidden.view(
                self.gru_layers, batch_size * beam_size, self.hidden_dim
            )

            output, hidden = self.forward(event, control, hidden)
            output = self.output_fc_activation(output / temperature)

            output = output.view(1, batch_size, beam_size, self.event_dim)
            hidden = hidden.view(
                self.gru_layers, batch_size, beam_size, self.hidden_dim
            )

            top_v, top_i = torch.log(output).topk(beam_size, -1)
            # [1, batch_size, beam_size, beam_size]
            top_v += score.view(1, batch_size, beam_size, 1)
            top_v = top_v.view(
                1, batch_size, -1
            )  # [1, batch_size, beam_size * beam_size]
            top_i = top_i.view(
                1, batch_size, -1
            )  # [1, batch_size, beam_size * beam_size]

            _, bbi = top_v.topk(beam_size, -1)  # [1, batch_size, beam_size]
            bbi = bbi.view(batch_size, -1)
            bi = torch.arange(batch_size).long().view(batch_size, -1)
            i = bbi / beam_size

            score = top_v[0, bi, bbi]
            hidden = hidden[:, bi, i, :]
            beam[:, :, :step] = beam[bi, i, :step]
            event = top_i[0, bi, bbi]
            beam[bi, i, step] = event

        best = beam[torch.arange(batch_size).long(), score.argmax(-1)]
        best = best.contiguous().t()
        return best

    def load_feature_map(self, images):

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

        input_batch = []
        for image_path in images:
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
        # print('feature_map', feature_map.shape)
        return feature_map.transpose(1, 2)
