import torch
import torch.nn as nn
import torch.nn.functional as F

class HDDM(nn.Module):
    def __init__(self):
        super(HDDM, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Automatically calculate the flattened size
        self.flatten_size = self._get_flatten_size((3, 128, 128))

        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)

    def _get_flatten_size(self, input_shape):
        """
        Calculate the flattened size after convolution and pooling.
        :param input_shape: Tuple (channels, height, width)
        :return: Flattened size
        """
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)  # Batch size = 1
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            return x.numel()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x