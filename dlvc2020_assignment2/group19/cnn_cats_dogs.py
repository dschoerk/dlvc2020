from collections import namedtuple

from PIL.Image import Image
from dlvc.models.pytorch import CnnClassifier
from dlvc.test import Accuracy

from dlvc.datasets.pets import PetsDataset, Subset
from dlvc.batches import BatchGenerator

import dlvc.ops as ops

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import argparse

TrainedModel = namedtuple('TrainedModel', ['model', 'accuracy'])

path = "..\\..\\cifar-10-python\\cifar-10-batches-py"

parser = argparse.ArgumentParser(description='CNN Classifier')
parser.add_argument('--model', type=str, default="Net", help='model used for classification')
parser.add_argument('--no_augmentation', action='store_true', help='Use data augmentation')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
parser.add_argument('--early_stop', type=int, default=5, help='Early stop after n not improved epochs')
args = parser.parse_args()

# 1) Load the training, validation, and test sets as individual PetsDatasets.
train = PetsDataset(path, Subset.TRAINING)
test = PetsDataset(path, Subset.TEST)
val = PetsDataset(path, Subset.VALIDATION)

avg = np.mean(train.images, axis=(0, 1, 2))
# avg = 127.5

input_shape = (0, 3, 32, 32) # needs to be 0, 3, 224, 244 for pretrained torch models
if args.model == "TransferResNet":
    input_shape = (0, 3, 224, 224)

enableDebugPlots = False
# 2) Create a BatchGenerator for each one.
op = ops.chain([
    ops.debug(enableDebugPlots),
    ops.type_cast(np.float32),
    ops.add(-avg),
    ops.mul(1 / avg),
    ops.type_cast(np.float32),
    ops.resize(input_shape[2], input_shape[3]),
    ops.hwc2chw(),
])

enableDebugPlots = False
op_with_augmentation = ops.chain([
    ops.debug(enableDebugPlots),
    ops.rcrop(32, 4, 'mean'), ops.debug(enableDebugPlots),
    ops.scale_centered_crop(28, 0.05), ops.debug(enableDebugPlots),
    ops.hflip(), ops.debug(enableDebugPlots),
    ops.rotate(5), ops.debug(enableDebugPlots,True),
    ops.type_cast(np.float32),
    ops.add(-avg),
    ops.mul(1 / avg),
    ops.type_cast(np.float32),
    ops.resize(input_shape[2], input_shape[3]),
    ops.hwc2chw(),
])

if args.no_augmentation:
    op_with_augmentation = op


batchsize = 128
train_batches = BatchGenerator(train, batchsize, shuffle=True, op=op_with_augmentation)
test_batches = BatchGenerator(test, batchsize, shuffle=False, op=op)
val_batches = BatchGenerator(val, batchsize, shuffle=False, op=op)
results = []

# architecture from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class AlternativeNet(nn.Module):
    """CNN."""

    def __init__(self):
        """CNN Builder."""
        super(AlternativeNet, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
            nn.Linear(512, 2)
        )

    def forward(self, x):
        """Perform forward."""

        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)

        return x


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
        x = F.dropout(x, p=0.1)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.1)
        x = self.fc3(x)
        return x


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.m = models.resnet18(pretrained=False)
        self.m.fc = nn.Linear(self.m.fc.in_features, 2) 

    def forward(self, x):
        return self.m(x)        

class TransferResNet(nn.Module):
    def __init__(self):
        super(TransferResNet, self).__init__()
        self.m = models.resnet18(pretrained=True)
        
        for p in self.m.parameters(): # freeze all existing parameters
            p.require_grad = False
                
        self.m.fc = nn.Linear(self.m.fc.in_features, 2) # new fc layer with same number of inputs, 2 output classes


    def forward(self, x):
        return self.m(x)


net = None
if args.model == "TransferResNet":
    net = TransferResNet()
elif args.model == "AlternativeNet":
    net = AlternativeNet()
elif args.model == "Net":
    net = Net()
elif args.model == "ResNet":
    net = ResNet()

# net = AlternativeNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

clf = CnnClassifier(
    net,
    input_shape=input_shape,
    num_classes=2,
    lr=args.lr,
    wd=5e-4,
    momentum=0.3
)

n_epochs = 100
e_early_stop = args.early_stop

best_accuracy = 0
accuracies = []
last_improvement = -1

for i in range(n_epochs):
    losses = []
    print("epoch %d" % i)
    for batch in train_batches:
        # train classifier
        loss = clf.train(batch.data, batch.label)
        losses.append(loss)

    print(" train loss: %.3f +- %.3f" % (np.average(losses), np.std(losses)))

    accuracy = Accuracy()
    for batch in val_batches:
        pred = clf.predict(batch.data)
        accuracy.update(pred, batch.label)

    accuracies.append(accuracy.accuracy_value)
    print(" val acc: accuracy: %.3f" % accuracy.accuracy())

    if accuracy.accuracy_value > best_accuracy:
        best_accuracy = accuracy
        torch.save(net.state_dict(), 'model.pth')
        last_improvement = i

    if i - last_improvement > e_early_stop:
        print('early stop triggered: ({})'.format(accuracies))
        break


def evaluate_model(clf: CnnClassifier):
    accuracy = Accuracy()
    for batch in test_batches:
        pred = clf.predict(batch.data)
        accuracy.update(pred, batch.label)

    return accuracy


net.load_state_dict(torch.load('model.pth'))
clf = CnnClassifier(
    net,
    input_shape=input_shape,
    num_classes=2,
    lr=args.lr,
    wd=5e-4,
    momentum=0.9
)
model_accuracy = evaluate_model(clf)
print("Model Accuracy: {}".format(model_accuracy))
