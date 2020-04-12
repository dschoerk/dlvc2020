from collections import namedtuple

from dlvc.models.linear import LinearClassifier
from dlvc.test import Accuracy

from dlvc.datasets.pets import PetsDataset, Subset
from dlvc.batches import BatchGenerator

import dlvc.ops as ops

import numpy as np

TrainedModel = namedtuple('TrainedModel', ['model', 'accuracy'])

path = "..\cifar-10-python\cifar-10-batches-py"
# path = "C:\\Users\\tommi\Desktop\\cifar-10-python\\cifar-10-batches-py"

# 1) Load the training, validation, and test sets as individual PetsDatasets.
train = PetsDataset(path, Subset.TRAINING)
test = PetsDataset(path, Subset.TEST)
val = PetsDataset(path, Subset.VALIDATION)

# 2) Create a BatchGenerator for each one.


op = ops.chain([
    ops.vectorize(),
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1 / 127.5),
])

batchsize = 256
train_batches = BatchGenerator(train, batchsize, shuffle=False, op=op)
test_batches = BatchGenerator(test, batchsize, shuffle=False, op=op)
val_batches = BatchGenerator(val, batchsize, shuffle=False, op=op)


def train_model(lr: float, momentum: float) -> TrainedModel:
    '''
    Trains a linear classifier with a given learning rate (lr) and momentum.
    Computes the accuracy on the validation set.
    Returns both the trained classifier and accuracy.
    '''

    # 3) Complete the function train_model. It trains a linear classifier for 10 epochs 
    # and then computes the accuracy on the validation set. You can choose to train with or without Nesterov momentum.

    # TODO implement step 3

    clf = LinearClassifier(
        input_dim=32 * 32 * 3,
        num_classes=2,
        lr=lr,
        momentum=momentum,
        nesterov=False)

    n_epochs = 10
    for i in range(n_epochs):
        for batch in train_batches:
            # train classifier

            clf.train(batch.data, batch.label)

            # print(batch.data.dtype)
            # print(data.shape)
            # print(batch.label.shape)

    accuracy = Accuracy()
    for batch in val_batches:
        pred = clf.predict(batch.data)
        accuracy.update(pred, batch.label)

        # predict and update accuracy

    return TrainedModel(clf, accuracy)


def tune_parameters():
    # hyper parameter tuning with random search based on accuracy in validation set
    best = 0
    bestModel = None
    for i in range(10):

        params = np.random.random_sample(2)
        lr = params[0]
        momentum = params[1]

        print("Iteration {}: lr={}, momentum={}".format(i, lr, momentum))

        model = train_model(lr=lr, momentum=momentum)

        if model.accuracy > best:
            best = model.accuracy
            bestModel = model.model
            print(model.accuracy)

    return bestModel


def evaluate_model(clf: LinearClassifier):
    accuracy = Accuracy()
    for batch in test_batches:
        pred = clf.predict(batch.data)
        accuracy.update(pred, batch.label)

    return accuracy


# determine acurracy on test set
model = tune_parameters()

print("parameter tuning completed.")

print("evaluating model on test set.")
model_accuracy = evaluate_model(model)

print("Model Accuracy: {}".format(model_accuracy))
