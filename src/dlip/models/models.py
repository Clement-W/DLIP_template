import torch
import torch.nn as nn
import torch.nn.functional as F
from dlip.utils.utils import instanciate_class


class LinearModel(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 n_layers: int,
                 hidden_size: int,
                 hidden_activ: str,
                 output_activ: str) -> None:
        super(LinearModel, self).__init__()

        self.l_in = nn.Linear(input_size, hidden_size)
        self.layers = [nn.Linear(hidden_size, hidden_size)
                       for _ in range(n_layers-1)]
        self.l_out = nn.Linear(hidden_size, output_size)

        self.hidden_activ = instanciate_class(hidden_activ)
        self.output_activ = instanciate_class(output_activ)

    def forward(self, inputs):  # Called when we apply the network
        x = self.hidden_activ(self.l_in(inputs))
        for layer in self.layers:
            x = self.hidden_activ(layer(x))
        x = self.output_activ(self.l_out(x))
        return x


def load_model(path_checkpoint, modelClass: torch.nn.Module, **kwargs):
    model = modelClass(**kwargs)
    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def save_model(path_checkpoint, model: torch.nn.Module):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
        },
        path_checkpoint,
    )
