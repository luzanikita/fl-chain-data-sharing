"""src: An authenticated Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

INFO = 30


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def _get_weights(self):
        """Convert model params to the list of vectors."""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def _set_weights(self, parameters):
        """Set model params with the list of vectors."""
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.load_state_dict(state_dict, strict=True)

    def _train_local(self, trainloader, valloader, epochs, learning_rate, device):
        """Train the model on the training set."""
        self.to(device)  # move model to GPU if available
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        self.train()
        for _ in range(epochs):
            for batch in trainloader:
                images = batch["img"]
                labels = batch["label"]
                optimizer.zero_grad()
                criterion(self(images.to(device)), labels.to(device)).backward()
                optimizer.step()

        val_loss, val_acc = self._test(valloader, device)

        results = {
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }
        return results

    def _test(self, testloader, device):
        """Validate the model on the test set."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, loss = 0, 0.0
        with torch.no_grad():
            for batch in testloader:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)
                outputs = self(images)
                loss += criterion(outputs, labels).item()
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        accuracy = correct / len(testloader.dataset)
        loss = loss / len(testloader)
        return loss, accuracy
