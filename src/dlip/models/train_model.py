import logging

import mlflow
import torch
from torch.utils.data import DataLoader

from dlip.models.evaluation import accuracy
from dlip.utils.logger import TFlogger


def train(num_epochs: int,
          batch_size: int,
          criterion,
          optimizer: torch.optim.Optimizer,
          model: torch.nn.Module,
          train_set: torch.utils.data.Dataset,
          val_set: torch.utils.data.Dataset,
          eval_freq: int,
          logger: TFlogger):

    train_error = []
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    model.train()

    for epoch in range(num_epochs):
        epoch_average_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            step = epoch * len(train_loader) + batch_idx
            y_pre = model(images.view(batch_size, -1))
            labels_one_hot = torch.FloatTensor(batch_size, 10)
            labels_one_hot.zero_()
            labels_one_hot.scatter_(1, labels.view(-1, 1), 1)

            loss = criterion(y_pre, labels_one_hot)  #
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_average_loss += loss.item() * batch_size / len(train_set)

            logger.add_scalar("train_error", loss.item(), step)

        train_error.append(epoch_average_loss)

        if (epoch % eval_freq == 0):
            model.eval()
            val_error = accuracy(val_set, model)
            logger.add_scalar("val_error", val_error, epoch)
            model.train()

        logging.info(
            "Epoch [{}/{}], Loss: {:.4f}".format(
                epoch + 1, num_epochs, epoch_average_loss
            )
        )

    return train_error
