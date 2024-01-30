import os

import hydra
from hydra import utils
from omegaconf import DictConfig

from dlip.data.data import load_dataset
from dlip.models.models import save_model
from dlip.models.train_model import train
from dlip.utils.logger import TFlogger
from dlip.utils.utils import instanciate_class


@hydra.main(config_path="conf", config_name="train_model")
def launch(cfg: DictConfig):

    logger = TFlogger(log_dir=os.getcwd())

    train_set, val_set = load_dataset(
        utils.get_original_cwd() + cfg.exp.data_path)

    model = instanciate_class(cfg.model.name, cfg.model.args)

    criterion = instanciate_class(cfg.train.criterion)

    optimizer = instanciate_class(
        cfg.train.optimizer.name, model.parameters(), cfg.train.optimizer.args)

    logger.log_params_from_omegaconf_dict(cfg)

    train(cfg.train.num_epochs, cfg.train.batch_size, criterion,
          optimizer, model, train_set, val_set, cfg.train.eval_freq, logger)

    # Save the checkpoint
    save_model(os.path.join(os.getcwd(), "checkpoint.pt"), model)


if __name__ == "__main__":
    launch()
