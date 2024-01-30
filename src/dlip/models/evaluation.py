import torch
from torch.utils.data import DataLoader


def accuracy(dataset, model: torch.nn.Module, verbose=False):
    with torch.no_grad():
        correct = 0
        dataloader = DataLoader(dataset)
        for images, labels in dataloader:
            images = images.view(-1, 16 * 16)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum()
    accuracy = correct.item() / len(dataset)
    if (verbose):
        print(
            "Accuracy of the model : {:.2f} %".format(100 * accuracy)
        )
    return accuracy
