import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter


class TFlogger(SummaryWriter):
    def __init__(self, log_dir=None):
        SummaryWriter.__init__(self, log_dir=log_dir)
        self.written_values = {}

    def add_scalar(self, name, value, iteration):
        if (name, iteration) in self.written_values:
            return
        else:
            self.written_values[(name, iteration)] = True

        if isinstance(value, int) or isinstance(value, float):
            SummaryWriter.add_scalar(self, name, value, iteration)

    def add_images(self, name, value, iteration):
        if (name, iteration) in self.written_values:
            return
        else:
            self.written_values[(name, iteration)] = True

        SummaryWriter.add_images(self, name, value, iteration)

    def log_params_from_omegaconf_dict(self, params):
        for param_name, element in params.items():
            self._explore_recursive(param_name, element)

    def _explore_recursive(self, parent_name, element):
        if isinstance(element, DictConfig):
            for k, v in element.items():
                if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                    self._explore_recursive(f"{parent_name}.{k}", v)
                else:
                    SummaryWriter.add_text(self, f"{parent_name}.{k}", str(v))
        elif isinstance(element, ListConfig):
            for i, v in enumerate(element):
                SummaryWriter.add_text(self, f"{parent_name}.{i}", str(v))
