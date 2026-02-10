import torch
import torch.nn as nn
import torch.nn.functional as F




class CNN_FEMNIST(nn.Module):
    """Used for EMNIST experiments in references[1]
    Args:
        only_digits (bool, optional): If True, uses a final layer with 10 outputs, for use with the
            digits only MNIST dataset (http://yann.lecun.com/exdb/mnist/).
            If selfalse, uses 62 outputs for selfederated Extended MNIST (selfEMNIST)
            EMNIST: Extending MNIST to handwritten letters: https://arxiv.org/abs/1702.05373
            Defaluts to `True`
    Returns:
        A `torch.nn.Module`.
    """
    def __init__(self, only_digits=False):
        super(CNN_FEMNIST, self).__init__()
        self.conv2d_1 = nn.Conv2d(1, 32, kernel_size=3)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout_1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(9216, 128)
        self.dropout_2 = nn.Dropout(0.5)
        self.linear_2 = nn.Linear(128, 10 if only_digits else 62)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.dropout_1(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout_2(x)
        x = self.linear_2(x)
        # x = self.softmax(x)
        return x


class cnn_mnist(nn.Module):
    """
    Convolutional Neural Network for MNIST.

    Description:
    ------------
    A simple convolutional neural network designed for the MNIST dataset. It 
    consists of two convolutional layers, ReLU activation, max pooling, and 
    fully connected layers.

    Examples:
    ---------
    >>> model = cnn_mnist()
    >>> x = torch.randn(16, 1, 28, 28)  # Batch of 16 grayscale MNIST images
    >>> output = model(x)
    >>> print(output.shape)
    torch.Size([16, 10])
    """
    def __init__(self):
        """Initialize the model parameters."""
        super().__init__()
        self._c1 = nn.Conv2d(1, 20, 5, 1)
        self._c2 = nn.Conv2d(20, 50, 5, 1)
        self._f1 = nn.Linear(800, 500)
        self._f2 = nn.Linear(500, 10)

    def forward(self, x):
        """Perform a forward pass through the model."""
        x = F.relu(self._c1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self._c2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self._f1(x.view(-1, 800)))
        x = F.log_softmax(self._f2(x), dim=1)
        return x


class MclrLogistic(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        super(MclrLogistic, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output


# one hidden layer

class NN1(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        super(NN1, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class NN1_PCA(nn.Module):
    def __init__(self, input_dim=60, output_dim=10):
        super(NN1_PCA, self).__init__()
        self.fc1 = nn.Linear(input_dim, 200)
        self.fc2 = nn.Linear(200, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# CNN

class CNN(nn.Module):
    def __init__(self, output_dim=10, inter_dim=200):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, stride=1)
        self.conv2 = nn.Conv2d(64, 64, 5, stride=1)
        self.fc1 = nn.Linear(4 * 4 * 64, inter_dim)
        self.fc2 = nn.Linear(inter_dim, output_dim)

    def forward(self, x):
        x = torch.reshape(x, (-1, 3, 28, 28))
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 64)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


