from collections import namedtuple
from dlvc.models.pytorch import CnnClassifier
from dlvc.test import Accuracy

from dlvc.datasets.pets import PetsDataset, Subset
from dlvc.batches import BatchGenerator

import dlvc.ops as ops

import numpy as np

import torch.nn as nn
import torch.nn.functional as F

TrainedModel = namedtuple('TrainedModel', ['model', 'accuracy'])

path = "..\\..\\cifar-10-python\\cifar-10-batches-py"

# 1) Load the training, validation, and test sets as individual PetsDatasets.
train = PetsDataset(path, Subset.TRAINING)
test = PetsDataset(path, Subset.TEST)
val = PetsDataset(path, Subset.VALIDATION)

# 2) Create a BatchGenerator for each one.
op = ops.chain([
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1 / 127.5),
    ops.hwc2chw()
])

batchsize = 128
train_batches = BatchGenerator(train, batchsize, shuffle=False, op=op)
test_batches = BatchGenerator(test, batchsize, shuffle=False, op=op)
val_batches = BatchGenerator(val, batchsize, shuffle=False, op=op)
results = []

# architecture from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



net = Net()
clf = CnnClassifier(
    net,
    input_shape=(0,3,32,32),
    num_classes=2,
    lr=0.1,
    wd=0)

n_epochs = 100

for i in range(n_epochs):
    losses = []
    print("epoch %d" % i)
    for batch in train_batches:
        # train classifier

        loss = clf.train(batch.data, batch.label)
        losses.append(loss)

        # print(batch.data.dtype)
        # print(data.shape)
        # print(batch.label.shape)

    print(" train loss: %.3f +- %.3f" % (np.average(losses), np.std(losses)))
    
    accuracy = Accuracy()
    for batch in val_batches:
        pred = clf.predict(batch.data)
        accuracy.update(pred, batch.label)
    
    print(" val acc: accuracy: %.3f" % accuracy.accuracy())
        

 